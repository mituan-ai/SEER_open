"""训练并评估单次运行；另行独占设备测量效率。

运行目录内容：
    config.resolved.yaml   解析后的配置
    split.json             各划分与评测组的段数与窗口数
    normalization.json     训练集统计量
    history.csv            每轮训练损失与验证指标
    best.pt                验证最优检查点，仅含模型状态
    predictions.csv        带评测组字段的常规测试预测（SEER 为原语义下的预测）
    metrics.csv            总体及各组的 MAE / RMSE / R2
    summary.json           训练来源记录与完成标记
    efficiency.json        独占设备测量的参数量、耗时与内存

数据是逐段流，窗口在 GPU 上按末端行号切出。每轮遍历一次有标签的训练窗口：
Base 的每个 batch 为 B 个有标签窗口，损失为 L_y；
SEER 另从全部有下一步的训练窗口中均匀抽取 B 个，L = L_y + eta * L_x，L_y 只用有标签窗口，
L_x 用全部有下一步的窗口；每个样本以概率 p 施加描述池中的一个合成语义变化，
输入窗口、下一步观测与该变量受影响字段的嵌入同步替换，目标不变。
两种方法都按验证集目标 RMSE 早停并选择检查点。
"""

from __future__ import annotations

import copy
import fcntl
import json
import random
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from seer.config import RunSpec, efficiency_settings_sha256, implementation_sha256
from seer.data import Normalization, Split, StreamDataset, build_split, load_dataset, training_normalization
from seer.io import git_state, read_json, resolve_path, write_csv, write_json
from seer.methods.prior import apply_windows
from seer.methods.seer import random_visibility
from seer.models.forecaster import Forecaster, build_forecaster
from seer.semantic_artifact import load_artifact
from seer.semantic_cards import FIELDS
from seer.semantic_prior import load_pool
from seer.training.evaluation import (
    count_parameters,
    grouped_metrics,
    measure_efficiency,
    peak_rss_mb,
    regression_metrics,
)

EVALUATION_SEED = 0  # 评估使用固定随机状态，Informer 在 eval 模式下仍会采样键


def run_status(spec: RunSpec) -> str:
    """运行状态：completed（完成）、outdated（配置相同、代码不同）、stale（配置不同）、
    incomplete（未完成）、not_started（未开始）、blocked（语义产物或描述池尚未构建）。"""

    summary = spec.run_dir / "summary.json"
    if summary.exists():
        record = read_json(summary)
        if record.get("config_sha256") != spec.config_sha256:
            return "stale"
        return "completed" if record.get("implementation_sha256") == implementation_sha256() else "outdated"
    method = spec.config["method"]
    if any(key in method and method[f"{key}_sha256"] is None for key in ("artifact", "prior")):
        return "blocked"
    return "incomplete" if spec.run_dir.exists() else "not_started"


def efficiency_status(spec: RunSpec) -> str:
    """效率状态为 measured 或 pending；每次测量对应一份训练结果和一组计时设置。"""

    path = spec.run_dir / "efficiency.json"
    if not path.exists() or not (spec.run_dir / "summary.json").exists():
        return "pending"
    record = read_json(path)
    current = (
        read_json(spec.run_dir / "summary.json").get("completed_at_utc"),
        efficiency_settings_sha256(spec.config),
        implementation_sha256(),
    )
    measured = (
        record.get("run_completed_at_utc"),
        record.get("settings_sha256"),
        record.get("implementation_sha256"),
    )
    return "measured" if measured == current else "pending"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """每个运行目录只允许一个进程使用；进程退出时释放锁。"""

    lock_path = run_dir.parent / f".{run_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{run_dir} is in use by another process") from None
        yield


@contextmanager
def evaluation_rng(device: torch.device) -> Iterator[None]:
    """评估结果由模型权重确定，不改变训练随机流。"""

    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(EVALUATION_SEED)
        yield


@dataclass
class Stream:
    """设备上常驻的标准化逐步流，按窗口末端行号切窗口。"""

    values: torch.Tensor  # (T, d)
    targets: torch.Tensor  # (T, p)，无标签处为 NaN
    starts: torch.Tensor  # (T,) 所在段的起点
    stops: torch.Tensor  # (T,) 所在段的终点（不含）
    window: int

    @classmethod
    def build(cls, dataset: StreamDataset, normalization: Normalization, device: torch.device) -> Stream:
        starts = np.zeros(len(dataset.values), dtype=np.int64)
        stops = np.zeros(len(dataset.values), dtype=np.int64)
        for segment in dataset.segments:
            starts[segment.start : segment.stop] = segment.start
            stops[segment.start : segment.stop] = segment.stop
        return cls(
            torch.as_tensor(normalization.inputs(dataset.values)).to(device),
            torch.as_tensor(normalization.targets(dataset.targets)).to(device),
            torch.as_tensor(starts).to(device),
            torch.as_tensor(stops).to(device),
            dataset.window_size,
        )

    def windows(self, ends: torch.Tensor) -> torch.Tensor:
        return self.values[ends[:, None] + torch.arange(-self.window + 1, 1, device=ends.device)]

    def following(self, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """下一步观测与其是否存在（t + 1 仍在同段内）。"""

        exists = ends + 1 < self.stops[ends]
        return self.values[torch.where(exists, ends + 1, ends)], exists


class Epochs:
    """每轮一次打乱的有标签窗口末端；SEER 另附等量的均匀抽取窗口。"""

    def __init__(
        self, labelled: np.ndarray, pool: np.ndarray | None, batch_size: int, seed: int, device
    ) -> None:
        self.labelled = torch.as_tensor(labelled, device=device)
        self.pool = None if pool is None else torch.as_tensor(pool, device=device)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return -(-len(self.labelled) // self.batch_size)

    def __iter__(self) -> Iterator[torch.Tensor]:
        order = torch.randperm(len(self.labelled), generator=self.generator).to(self.labelled.device)
        for start in range(0, len(order), self.batch_size):
            ends = self.labelled[order[start : start + self.batch_size]]
            if self.pool is not None:
                draw = torch.randint(len(self.pool), (self.batch_size,), generator=self.generator)
                ends = torch.cat([ends, self.pool[draw.to(self.pool.device)]])
            yield ends


class Prior:
    """描述池：变换参数与各措辞的字段嵌入，均在设备上。

    被抽中的样本同步替换记录与卡片（池中的卡是被改变量的完整描述，保留原字段可能与变化矛盾）。
    """

    def __init__(self, pool, probability: float, seed: int, device: torch.device) -> None:
        t = pool.transforms
        self.probability = float(probability)
        self.params = {
            name: torch.as_tensor(getattr(t, name), dtype=torch.float32, device=device)
            for name in ("target", "source", "kernel", "gain", "bias", "noise", "hold")
        }
        self.fields = torch.as_tensor(pool.fields, dtype=torch.float32, device=device)  # (N, P, F, k)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    @classmethod
    def like(cls, other: Prior, seed: int) -> Prior:
        """同一描述池与概率、另一随机流（验证时每次从同一种子开始，各轮可比）。"""

        prior = copy.copy(other)
        prior.generator = torch.Generator(device=other.fields.device).manual_seed(seed)
        return prior

    def __call__(
        self, stream: Stream, ends: torch.Tensor, x: torch.Tensor, nxt: torch.Tensor, fields: torch.Tensor
    ) -> torch.Tensor:
        """就地替换被抽中样本的窗口、下一步观测与字段嵌入；返回被替换的 (样本, 变量) 单元 (B, d)。"""

        chosen = torch.rand(len(ends), device=ends.device, generator=self.generator) < self.probability
        rows, draw = self._draw(chosen)
        if rows.numel() == 0:
            return torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool, device=ends.device)
        params = {name: value[draw] for name, value in self.params.items()}
        params["phase"] = torch.randint(
            0, 1 << 20, (rows.numel(),), device=ends.device, generator=self.generator
        )
        values = apply_windows(
            stream.values, stream.starts[ends[rows]], ends[rows], stream.window, params, self.generator
        )
        target = params["target"].long()
        x[rows, :, target] = values[:, :-1]
        nxt[rows, target] = values[:, -1]
        return self._write(fields, rows, draw)

    def _draw(self, chosen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.nonzero(chosen).squeeze(1)
        draw = torch.randint(
            len(self.params["target"]), (rows.numel(),), device=chosen.device, generator=self.generator
        )
        return rows, draw

    def _write(self, fields: torch.Tensor, rows: torch.Tensor, draw: torch.Tensor) -> torch.Tensor:
        changed = torch.zeros(fields.shape[:2], dtype=torch.bool, device=fields.device)
        if rows.numel() == 0:
            return changed
        target = self.params["target"][draw].long()
        phrasing = torch.randint(
            self.fields.shape[1], (rows.numel(),), device=fields.device, generator=self.generator
        )
        fields[rows, target] = self.fields[draw, phrasing]  # 池中的卡是被改变量的完整描述
        changed[rows, target] = True
        return changed


def _device(training: dict[str, Any]) -> torch.device:
    return torch.device(training.get("device", "cuda") if torch.cuda.is_available() else "cpu")


@dataclass
class Prepared:
    dataset: StreamDataset
    split: Split
    normalization: Normalization
    model: Forecaster
    device: torch.device
    prior: Any | None


def _steps(dataset: StreamDataset, segments: np.ndarray, normalization: Normalization) -> np.ndarray:
    values = normalization.inputs(dataset.values)
    steps = [np.diff(values[dataset.segments[i].start : dataset.segments[i].stop], axis=0) for i in segments]
    return np.concatenate(steps)


def step_std(dataset: StreamDataset, segments: np.ndarray, normalization: Normalization) -> np.ndarray:
    """各段内标准化一步增量 x_{t+1} - x_t 的标准差 (d,)。"""

    return _steps(dataset, segments, normalization).std(axis=0)


def step_rms(dataset: StreamDataset, segments: np.ndarray, normalization: Normalization) -> np.ndarray:
    """各段内标准化一步增量的均方根 (d,)：观测头读取增量的预条件尺度（含确定性的恒定步长）。"""

    return np.sqrt(np.mean(_steps(dataset, segments, normalization) ** 2, axis=0))


def prepare(spec: RunSpec) -> Prepared:
    config = spec.config
    device = _device(config["training"])
    set_seed(spec.seed)
    dataset = load_dataset(config["dataset"])
    split = build_split(dataset, config["dataset"])
    normalization = training_normalization(dataset, split)
    method = config["method"]
    extra: dict[str, Any] = {}
    prior = None
    if method["name"] == "seer":
        spec_data = method["dataset"]
        fields = load_artifact(resolve_path(method["artifact"]), dataset.feature_names).fields
        observed = tuple(dataset.feature_names.index(name) for name in spec_data["observed"])
        extra = {
            "fields": fields,
            "observed": observed,
            "reader_lags": int(spec_data["reader_lags"]),
            "scale_floor": float(method["scale_floor"])
            * step_std(dataset, split.train, normalization)[list(observed)],
            "step": step_rms(dataset, split.train, normalization),
            "reference": ((FIELDS.index("computation"), FIELDS.index("dependency")), FIELDS.index("source")),
        }
        if "prior" in method:
            prior = load_pool(resolve_path(method["prior"]), dataset.feature_names)
    model = build_forecaster(
        method=method,
        backbone=config["backbone"],
        feature_names=dataset.feature_names,
        window_size=dataset.window_size,
        output_dim=len(dataset.target_names),
        **extra,
    ).to(device)
    if device.type == "cuda" and getattr(model.backbone, "fast_path", True) is False:
        raise RuntimeError(
            "Mamba CUDA kernels are unavailable, so HuggingFace would fall back to its sequential scan; "
            "run once with network access to cache the pinned kernels (seer.models.mamba_kernels)"
        )
    return Prepared(dataset, split, normalization, model, device, prior)


def run(spec: RunSpec, *, force: bool = False, log=print) -> Path:
    with _run_lock(spec.run_dir):
        status = run_status(spec)
        if status == "completed" and not force:
            log(f"[skip] {spec.name} already completed")
            return spec.run_dir
        if status == "blocked":
            raise RuntimeError(f"semantic artifact or prior pool of {spec.name} is not built yet")
        if status in ("stale", "outdated") and not force:
            what = "another config" if status == "stale" else "other code"
            raise RuntimeError(f"{spec.run_dir} holds a run with {what}; pass --force to replace it")
        if spec.run_dir.exists():
            shutil.rmtree(spec.run_dir)
        spec.run_dir.mkdir(parents=True)
        return _train_run(spec, log)


def _train_run(spec: RunSpec, log) -> Path:
    config = spec.config
    (spec.run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    training = config["training"]
    rss_start = peak_rss_mb()  # 导入完成、读取数据与构建模型之前的常驻内存峰值
    p = prepare(spec)
    write_json(spec.run_dir / "split.json", p.split.summary(p.dataset))
    write_json(
        spec.run_dir / "normalization.json",
        p.normalization.as_dict(p.dataset.feature_names, p.dataset.target_names),
    )
    stream = Stream.build(p.dataset, p.normalization, p.device)
    seer = p.model.semantic
    epochs = Epochs(
        p.dataset.ends(p.split.train, labelled=True),
        p.dataset.ends(p.split.train, has_next=True) if seer else None,
        int(training["batch_size"]),
        spec.seed,
        p.device,
    )
    prior = None
    if p.prior is not None:
        prior = Prior(p.prior, float(config["method"]["prior_probability"]), spec.seed, p.device)
    start = time.perf_counter()
    history, best_epoch = train(
        p.model,
        stream,
        epochs,
        p.dataset.ends(p.split.validation, labelled=True),
        p.normalization,
        training,
        float(config["method"].get("observation_weight", 0.0)),
        prior,
        spec.run_dir / "best.pt",
        log=log,
    )
    observation_epoch = 0
    if prior is not None:
        batch_size = int(training["batch_size"])
        train_ends = p.dataset.ends(p.split.train, has_next=True)
        transitions = Epochs(train_ends, None, batch_size, spec.seed, p.device)  # 观测头在全部训练转移上拟合
        stage, observation_epoch = fit_observation(
            p.model,
            stream,
            transitions,
            p.dataset.ends(p.split.validation, has_next=True),
            training,
            prior,
            spec.run_dir / "best.pt",
            log=log,
        )
        write_csv(spec.run_dir / "observation_history.csv", stage)
    train_seconds = time.perf_counter() - start
    write_csv(spec.run_dir / "history.csv", history)

    p.model.load_state_dict(torch.load(spec.run_dir / "best.pt", map_location=p.device))
    ends = p.dataset.ends(p.split.test, labelled=True)
    target, prediction = predict(p.model, stream, ends, p.normalization, int(training["batch_size"]))
    group_columns = list(dict.fromkeys(config["dataset"].get("eval_groups", {}).values()))
    _write_predictions(spec.run_dir / "predictions.csv", p.dataset, group_columns, ends, target, prediction)
    write_csv(spec.run_dir / "metrics.csv", grouped_metrics(p.split, p.dataset, ends, target, prediction))

    total, trainable = count_parameters(p.model)
    summary: dict[str, Any] = {
        "run": spec.name,
        "dataset": spec.dataset,
        "backbone": spec.backbone,
        "method": spec.method,
        "variant": spec.variant,
        "seed": spec.seed,
        "overrides": dict(spec.overrides),
        "config_sha256": spec.config_sha256,
        "pairing_sha256": spec.pairing_sha256,
        "implementation_sha256": implementation_sha256(),
        "best_epoch": best_epoch,
        "observation_epoch": observation_epoch,
        "epochs_run": len(history),
        "best_val_RMSE": float(history[best_epoch - 1]["val_RMSE"]),
        "train_wall_seconds": train_seconds,  # 设备可能被共享，此耗时不作计时指标
        "parameters": total,
        "trainable_parameters": trainable,
        "adapter_parameters": count_parameters(p.model.reader)[0] if seer else 0,
        "peak_rss_mb": peak_rss_mb(),
        "peak_rss_increase_mb": peak_rss_mb() - rss_start,
        "artifact": _artifact_provenance(config["method"]),
        "mamba_fast_path": getattr(p.model.backbone, "fast_path", None),
        "device": str(p.device),
        "device_name": torch.cuda.get_device_name(p.device) if p.device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "git": git_state(),
        "command": sys.argv,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(spec.run_dir / "summary.json", summary)
    log(f"[done] {spec.name} best_epoch={best_epoch} val_RMSE={summary['best_val_RMSE']:.5g}")
    return spec.run_dir


def losses(
    model: Forecaster, stream: Stream, ends: torch.Tensor, prior: Prior | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """一个 batch 的 L_y（有标签窗口的 MSE）与 L_x（观测头在全部存在单元上的平均负对数似然）。

    合成样本的窗口、下一步与该变量的卡同步替换为描述池中的一个变化；同时刻可见性逐样本从部署时的
    同一族中抽取（``random_visibility``），被检验集 U 含改动的观测变量。L_x 是增广训练分布上的似然，
    合成单元按其实际比例计入：观测头主要估计原语义下的自然读取，合成变化教文本引用落到卡片所写的来源上。
    """

    x = stream.windows(ends)
    y = stream.targets[ends]
    if not model.semantic:
        return torch.mean((model(x)[0] - y) ** 2), None
    nxt, exists = stream.following(ends)
    observed = model.observation.observed
    fields = model.fields.expand(len(ends), *model.fields.shape).clone()
    labelled = torch.isfinite(y).all(dim=1)
    changed = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool, device=x.device)
    if prior is not None:
        changed = prior(stream, ends, x, nxt, fields)
    target = nxt[:, observed]  # 增广之后：合成单元的响应是变换后的记录
    visible, _ = random_visibility(observed, x.shape[-1], changed[:, observed])
    prediction, (mean, scale) = model(x, fields, nxt, visible)
    loss_y = torch.mean((prediction[labelled] - y[labelled]) ** 2)
    nll = model.observation.nll(target, mean, scale)
    return loss_y, nll[exists].mean()


def train(
    model: Forecaster,
    stream: Stream,
    epochs: Epochs,
    validation: np.ndarray,
    normalization: Normalization,
    training: dict[str, Any],
    eta: float,
    prior: Prior | None,
    checkpoint: Path,
    *,
    log=print,
) -> tuple[list[dict[str, Any]], int]:
    """返回训练历史与已保存检查点对应的轮次。

    只有验证 RMSE 相对当前最优值的下降超过 ``early_stopping_rel_delta``，才视为改善。
    不同数据集的标签尺度相差多个数量级，绝对阈值会使部分数据集过早停止。
    """

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["min_lr"]),
    )
    clip = float(training["gradient_clip_norm"])
    margin = float(training["early_stopping_rel_delta"])
    batch_size = int(training["batch_size"])
    best, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    device = stream.values.device
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        sums = torch.zeros(2, device=device)
        for ends in epochs:
            optimizer.zero_grad(set_to_none=True)
            loss_y, loss_x = losses(model, stream, ends, prior)
            loss = loss_y if loss_x is None else loss_y + eta * loss_x
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            sums += torch.stack(
                [loss_y.detach(), torch.zeros((), device=device) if loss_x is None else loss_x.detach()]
            )
        target, prediction = predict(model, stream, validation, normalization, batch_size)
        metrics = regression_metrics(target, prediction)
        scheduler.step(metrics["RMSE"])
        row = {
            "epoch": epoch,
            "train_loss_y": float(sums[0]) / len(epochs),
            **({"train_loss_x": float(sums[1]) / len(epochs)} if model.semantic else {}),
            **{f"val_{key}": value for key, value in metrics.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if metrics["RMSE"] < best * (1.0 - margin):
            best, best_epoch, stale = metrics["RMSE"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        # 调度器读取该行显示进度
        log(
            f"epoch {epoch}/{training['max_epochs']} train_loss={row['train_loss_y']:.4g} "
            f"val_RMSE={metrics['RMSE']:.4g} best={best:.4g}@{best_epoch} "
            f"stale={stale}/{training['early_stopping_patience']} lr={row['learning_rate']:.2g}"
            + (f" loss_x={row['train_loss_x']:.4g}" if model.semantic else "")
        )
        if stale >= int(training["early_stopping_patience"]):
            log(f"early stopping at epoch {epoch}")
            break
    if not checkpoint.exists():
        raise RuntimeError("training produced no finite validation result")
    return history, best_epoch


def fit_observation(
    model: Forecaster,
    stream: Stream,
    epochs: Epochs,
    validation: np.ndarray,
    training: dict[str, Any],
    prior: Prior,
    checkpoint: Path,
    *,
    log=print,
) -> tuple[list[dict[str, Any]], int]:
    """第二阶段：冻结目标通路与语义投影，只训练观测头，按验证段（施加同一合成先验）上的 L_x 早停。

    观测头是冻结语义表示之上的条件密度模型，以自己的留出似然选择；第一阶段按目标 RMSE 选择的检查点
    常停在 L_x 仍在下降的早期。目标预测不受此阶段影响。
    """

    model.load_state_dict(torch.load(checkpoint, map_location=stream.values.device))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("observation."))
    optimizer = torch.optim.AdamW(
        model.observation.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size, clip = int(training["batch_size"]), float(training["gradient_clip_norm"])  # 同第一阶段
    ends = torch.as_tensor(validation, device=stream.values.device)
    best, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        total = 0.0
        for batch in epochs:
            optimizer.zero_grad(set_to_none=True)
            loss = losses(model, stream, batch, prior)[1]
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.observation.parameters(), clip)
            optimizer.step()
            total += float(loss.detach())
        model.eval()
        held = Prior.like(prior, EVALUATION_SEED)
        with torch.no_grad(), evaluation_rng(stream.values.device):
            value = np.mean(
                [
                    float(losses(model, stream, ends[a : a + batch_size], held)[1])
                    for a in range(0, len(ends), batch_size)
                ]
            )
        history.append({"epoch": epoch, "train_loss_x": total / len(epochs), "val_loss_x": value})
        if value < best - 1e-4:
            best, best_epoch, stale = value, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        log(
            f"observation epoch {epoch} train_loss_x={total / len(epochs):.4g} val_loss_x={value:.4g} "
            f"best={best:.4g}@{best_epoch}"
        )
        if stale >= int(training["early_stopping_patience"]):
            break
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.load_state_dict(torch.load(checkpoint, map_location=stream.values.device))
    return history, best_epoch


@torch.no_grad()
def predict(
    model: Forecaster,
    stream: Stream,
    ends: np.ndarray,
    normalization: Normalization,
    batch_size: int,
    fields: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """给定末端行号的目标与预测（原始尺度）；``fields`` (d, F, k) 替换语义。"""

    model.eval()
    index = torch.as_tensor(ends, device=stream.values.device)
    predictions = []
    with evaluation_rng(stream.values.device):
        for start in range(0, len(index), batch_size):
            batch = index[start : start + batch_size]
            semantics = None if fields is None else fields.expand(len(batch), *fields.shape)
            predictions.append(model(stream.windows(batch), semantics)[0])
    targets = stream.targets[index]
    return (
        normalization.inverse_targets(targets.cpu().numpy()),
        normalization.inverse_targets(torch.cat(predictions).cpu().numpy()),
    )


def measure(spec: RunSpec, *, log=print) -> Path:
    """从已完成运行的检查点测量效率；调用方负责保持设备独占。"""

    with _run_lock(spec.run_dir):
        if run_status(spec) != "completed":
            raise RuntimeError(f"{spec.name} is not a completed run with the current config and code")
        p = prepare(spec)
        p.model.load_state_dict(torch.load(spec.run_dir / "best.pt", map_location=p.device))
        batch_size = int(spec.config["training"]["batch_size"])
        stream = Stream.build(p.dataset, p.normalization, torch.device("cpu"))

        def batches(ends: np.ndarray) -> list[tuple[torch.Tensor, torch.Tensor]]:
            index = torch.as_tensor(ends)
            return [
                (stream.windows(index[s : s + batch_size]), stream.targets[index[s : s + batch_size]])
                for s in range(0, len(index), batch_size)
            ]

        settings = spec.config["efficiency"]
        train_ends = p.dataset.ends(p.split.train, labelled=True)
        with evaluation_rng(p.device):
            result = measure_efficiency(
                p.model,
                train_batches=batches(np.random.default_rng(0).permutation(train_ends)),
                test_batches=[x for x, _ in batches(p.dataset.ends(p.split.test, labelled=True))],
                device=p.device,
                training=spec.config["training"],
                settings=settings,
            )
        summary = read_json(spec.run_dir / "summary.json")
        # 独占设备的训练时间估计：每轮一次训练集更新与一次验证集前向，乘以实际训练轮数
        epoch_ms = (
            train_ends.size * result["train_ms_per_sample"]
            + p.dataset.ends(p.split.validation, labelled=True).size * result["inference_ms_per_sample"]
        )
        write_json(
            spec.run_dir / "efficiency.json",
            {
                **result,
                "train_time_h": summary["epochs_run"] * epoch_ms / 3.6e6,
                "adapter_parameters": summary["adapter_parameters"],
                "settings": settings,
                "settings_sha256": efficiency_settings_sha256(spec.config),
                "implementation_sha256": implementation_sha256(),
                "run_completed_at_utc": summary["completed_at_utc"],
                "device": str(p.device),
                "device_name": torch.cuda.get_device_name(p.device) if p.device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "measured_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        log(f"[measured] {spec.name} inference={result['inference_ms_per_sample']:.4f} ms/sample")
        return spec.run_dir


def _write_predictions(
    path: Path,
    dataset: StreamDataset,
    columns: list[str],
    ends: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    rows = []
    for row, (end, meta) in enumerate(zip(ends, dataset.meta(ends), strict=True)):
        item: dict[str, Any] = {"row": int(end)}
        item.update({column: meta[column] for column in columns})
        for position, name in enumerate(dataset.target_names):
            item[f"target.{name}"] = float(target[row, position])
            item[f"prediction.{name}"] = float(prediction[row, position])
        rows.append(item)
    write_csv(path, rows)


def _artifact_provenance(method: dict[str, Any]) -> dict[str, Any] | None:
    if "artifact" not in method:
        return None
    artifact = resolve_path(method["artifact"])
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    return {
        "path": str(artifact),
        "source": metadata.get("source"),
        "arrays_sha256": method.get("artifact_sha256"),
        "prior": method.get("prior"),
        "prior_sha256": method.get("prior_sha256"),
    }
