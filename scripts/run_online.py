"""在线修订评测：在训练 study 的 Base 与 SEER 检查点上运行 base、off、seer 三种评测方式（不训练）。

    uv run --extra llm python scripts/run_online.py configs/studies/smoke.yaml \
        [--dataset ...] [--backbones ...] [--seeds ...] [--modes base off seer] [--streams NAME ...]

检查点按 study 的数据集、主干、种子、``overrides`` 与 ``output_root`` 定位（须已训练完成）；
在常规、未知变化与漂移流上评测。
  base  Base 网络；off  SEER 网络，原语义；seer  SEER 网络，LLM 候选的在线修订。
指标只统计变化起点之后的有标签窗口。修订事件（触发、失配摘要、语义梯度、候选、权重轨迹）
写入 outputs/online/<dataset>/<backbone>/<variant>/seed_<seed>/<stream>.<mode>.json，
指标写入同目录的 metrics.csv。
seer 方式在证据触发后离线调用 LLM 与文本编码器（configs/env/api.local.env）；LLM 候选按提示词哈希缓存在
outputs/proposals/，字段编码缓存在 outputs/embedding_cache/。base 与 off 不调用外部模型。
``--evaluate train`` 在训练段上评测，写入上述目录下的 train/：
无变化的训练流上不应有候选被审批，用于部署前检查适应族的声明。
"""

from __future__ import annotations

import argparse
import fnmatch
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from seer.config import CONFIG_DIR, RUN_ROOT, dataset_config, dataset_spec, resolve_run
from seer.io import ROOT, load_api_env_files, load_yaml, read_csv, resolve_path, write_csv, write_json
from seer.llm import LLMConfig, patiently
from seer.revision import propose
from seer.semantic_artifact import EmbeddingConfig, EncodingCache
from seer.semantic_artifact import revise_fields as encode_revision
from seer.semantic_cards import FIELDS, cards_path, dataset_materials, load_cards
from seer.semantic_models import load_registry, model_slug
from seer.training.evaluation import grouped_metrics
from seer.training.online import (
    Reference,
    Runner,
    Settings,
    calibrate,
    cusum_threshold,
    lagged_correlation,
    monitor_reference,
    outputs,
    static_metrics,
    streams,
)
from seer.training.runner import Stream, prepare, run_status

MODES = ("base", "off", "seer")
REFERENCE_SIZE = 4000  # 训练流参照末端数（均匀抽取）：参照 NLL 与时滞相关
ONLINE_ROOT = ROOT / "outputs" / "online"


def online_dir(spec, root: Path = ONLINE_ROOT) -> Path:
    """在线评测输出与训练运行同样按主干与方法变体（含 LLM 与编码器）分目录。"""

    return root / spec.dataset / spec.backbone / spec.variant / f"seed_{spec.seed}"


def revise_fields(fields, names, revisions, cache):
    """编码修订后的语义；编码接口断网时等待恢复（``patiently``）。"""

    return patiently(lambda: encode_revision(fields, names, revisions, cache))


def load(dataset: str, backbone: str, method: str, seed: int, overrides: dict, root: Path):
    spec = resolve_run(dataset, backbone, method, seed, overrides, output_root=root)
    if run_status(spec) != "completed":
        raise SystemExit(f"{spec.name} is {run_status(spec)}; train it first")
    p = prepare(spec)
    p.model.load_state_dict(torch.load(spec.run_dir / "best.pt", map_location=p.device))
    p.model.eval()
    return spec, p


def reference(
    p, settings: Settings, lags: int, quantile: float
) -> tuple[np.ndarray, Reference, tuple[np.ndarray, float]]:
    """训练流上的支撑盒与参照统计量；正常运行流（训练与验证）上的监控水平与逐变量 CUSUM 峰值（原语义）。"""

    data = Stream.build(p.dataset, p.normalization, p.device)
    values = data.values.cpu().numpy()
    rows = np.concatenate(
        [np.arange(p.dataset.segments[i].start, p.dataset.segments[i].stop) for i in p.split.train]
    )
    box = np.quantile(values[rows], [quantile, 1 - quantile], axis=0)
    observed = p.model.observation.observed

    def calibrated(segments):
        ends = p.dataset.ends(segments, has_next=True)
        ends = ends[np.all((values[ends] >= box[0]) & (values[ends] <= box[1]), axis=1)]
        _, mean, scale = outputs(p.model, data, ends, p.model.fields, settings.batch_size)
        following = data.following(torch.as_tensor(ends, device=p.device))[0][:, observed].cpu().numpy()
        return ends, mean, scale, following, values[ends][:, observed.cpu().numpy()]

    # 监控标定：全部正常运行段（训练与验证，按序相接），验证段在网络拟合之外，延长标定长度
    normal = calibrated(np.sort(np.r_[p.split.train, p.split.validation]))[1:]
    level, peak = monitor_reference(*normal, settings)
    ends, mean, scale, following, current = calibrated(p.split.train)
    pick = np.linspace(0, len(ends) - 1, min(len(ends), REFERENCE_SIZE)).astype(int)
    nll = calibrate(mean[pick], scale[pick], following[pick], current[pick])[2].mean(0)
    correlation = lagged_correlation(values, ends[pick], lags)
    return box, Reference(nll, correlation), (level, peak)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("study", help="训练 study（检查点按其数据集、主干、种子与 overrides 定位）")
    parser.add_argument("--dataset", nargs="+")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--streams", nargs="+", default=["*"], help="流名称的通配模式")
    parser.add_argument("--evaluate", choices=["test", "train"], default="test")
    args = parser.parse_args()
    study = load_yaml(resolve_path(args.study))
    overrides = dict(study.get("overrides") or {})
    root = resolve_path(study.get("output_root", RUN_ROOT))
    raw = load_yaml(CONFIG_DIR / "revision.yaml")
    load_api_env_files()
    llms, encoders = load_registry(CONFIG_DIR / "llm.yaml"), load_registry(CONFIG_DIR / "embedding.yaml")
    llm = llms.resolve(raw["llm"])
    for dataset in args.dataset or study["datasets"]:
        for backbone in args.backbones or study["backbones"]:
            for seed in args.seeds or study["seeds"]:
                run_one(dataset, backbone, seed, args.modes, args, raw, llms, llm, encoders, overrides, root)


def run_one(dataset, backbone, seed, modes, args, raw, llms, llm, encoders, overrides, root) -> None:
    base = load(dataset, backbone, "base", seed, overrides, root)[1] if "base" in modes else None
    spec, seer = load(dataset, backbone, "seer", seed, overrides, root)
    data = spec.config["dataset"]
    event = dataset_spec(data)["event"]
    settings = Settings(
        budget=int(raw["budget"]),
        risk=float(raw["risk"]),
        threshold=float(raw["threshold"]),
        alpha=float(raw["prior"]["alpha"]),
        beta=float(raw["prior"]["beta"]),
        warmup=int(event["warmup"]),
        calibration_floor=float(raw["calibration_floor"]),
        exact_floor=float(raw["exact_floor"]),
        cusum_drift=float(raw["cusum"]["drift"]),
        cusum_cap=float(raw["cusum"]["cap"]),
        cusum_threshold=math.inf,
        batch_size=int(spec.config["training"]["batch_size"]),
    )
    method = spec.config["method"]
    materials = dataset_materials(dataset_config(dataset))
    cards = {
        c["variable_name"]: {f: c[f] for f in FIELDS}
        for c in load_cards(cards_path(dataset, method["llm"]), materials)["cards"]
    }
    cache = config = None
    if "seer" in modes:  # 只有在线修订调用外部模型；未配置 API 时在此报错
        encoder = EmbeddingConfig.from_mapping(encoders.entry(method["embedding"]))
        shared = ROOT / "outputs" / "embedding_cache" / f"{model_slug(method['embedding'])}.jsonl"
        cache = EncodingCache(shared, encoder)
        config = LLMConfig.from_mapping(llms.entry(llm))
    proposals = ROOT / "outputs" / "proposals" / dataset / model_slug(llm)
    names = seer.dataset.feature_names
    original = seer.model.fields
    quantile = float(raw["support_quantile"])
    box, ref, (level, peak) = reference(seer, settings, seer.model.reader.lags, quantile)
    threshold = cusum_threshold(peak, settings, float(raw["cusum"]["margin"]))
    settings = replace(settings, cusum_threshold=threshold, monitor_level=level)
    minutes = float(dataset_spec(data)["sampling_minutes"])
    runner = Runner(seer, cards, box, ref, settings, minutes)
    segments = {"test": seer.split.test, "train": seer.split.train}[args.evaluate]
    out = online_dir(spec)
    if args.evaluate == "train":
        out = out / "train"

    def tensor(array):
        return torch.as_tensor(array, dtype=torch.float32, device=seer.device)

    def from_llm(context):
        current = context["base"]
        record = propose(materials, current.cards, context["summary"], settings.budget, config, proposals)
        context["proposal"] = {k: record.get(k) for k in ("model", "usage", "seconds", "failed_attempts")}
        result = []
        for candidate in record["candidates"]:
            fields = revise_fields(current.fields.cpu().numpy(), names, candidate["revisions"], cache)
            revised = {v: {**text, **candidate["revisions"].get(v, {})} for v, text in current.cards.items()}
            result.append((tensor(fields), revised, candidate))
        return result

    rows = []
    for stream in streams(seer, spec.config, segments):
        if not any(fnmatch.fnmatch(stream.name, pattern) for pattern in args.streams):
            continue
        for mode in modes:
            extra = {}
            if mode in ("base", "off"):
                prepared, fields = (base, None) if mode == "base" else (seer, original)
                groups = static_metrics(prepared, stream, fields, settings.batch_size)
            else:
                result = runner.run(stream, from_llm)
                ends, prediction, events = result["ends"], result["prediction"], result["events"]
                write_json(out / f"{stream.name}.{mode}.json", {"events": events})
                extra = event_columns(events, stream, settings)
                groups = metrics_after_onset(seer, stream, ends, prediction)
            for group in groups:
                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "seed": seed,
                        "stream": stream.name,
                        "kind": stream.kind,
                        "mode": mode,
                    }
                    | group
                    | extra
                )
            overall = groups[0]
            print(
                f"{dataset:<22} {backbone:<12} seed={seed} {stream.name:<36} {mode:<6}"
                f" MAE={overall['MAE']:.4g}"
                + "".join(f" {k}={v}" for k, v in extra.items() if k in ("events", "crossed", "approved")),
                flush=True,
            )
        merge(out / "metrics.csv", rows)  # 逐流落盘，中断不丢已完成的流


def metrics_after_onset(p, stream, ends, prediction):
    segment = stream.dataset.segment_index(ends)
    onset = stream.onsets[np.searchsorted(stream.segments, segment)]
    keep = (ends >= onset) & np.isfinite(stream.dataset.targets[ends]).all(axis=1)
    target = p.normalization.inverse_targets(
        Stream.build(stream.dataset, p.normalization, p.device)
        .targets[torch.as_tensor(ends[keep], device=p.device)]
        .cpu()
        .numpy()
    )
    return grouped_metrics(
        p.split, stream.dataset, ends[keep], target, p.normalization.inverse_targets(prediction[keep])
    )


def event_columns(events, stream, settings) -> dict:
    """事件计数、越界与审批；变化流上另记最终占优候选是否修订了被改写的变量。"""

    revised = [e for e in events if e.get("candidates")]
    approved = [e for e in revised if e.get("approved")]
    last = revised[-1] if revised else None
    hit = None
    if stream.variables and last is not None and "shares" in last:
        top = last["candidates"][int(np.argmax(last["shares"]))]
        touched = set(top.get("revisions", {})) or {top.get("variable")}
        hit = bool(touched & set(stream.variables))
    return {
        "events": len(events),
        "crossed": sum(bool(e.get("crossed")) for e in revised),
        "approved": len(approved),
        "max_w": max((e.get("max_w", 0.0) for e in revised), default=0.0),
        "final_w": last.get("w", 0.0) if last else 0.0,
        "top_hits_changed_variable": hit,
        "risk": settings.risk,
    }


def merge(path, rows) -> None:
    """按 (stream, mode) 覆盖已有行。"""

    keys = {(r["stream"], r["mode"]) for r in rows}
    old = [r for r in read_csv(path) if (r["stream"], r["mode"]) not in keys] if path.exists() else []
    write_csv(path, old + rows)



if __name__ == "__main__":
    main()
