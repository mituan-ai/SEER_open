"""部署期评测：在常规、未知变化与漂移流上评测冻结检查点。

静态评测只在给定语义下前向；在线修订不改网络参数。
各流的指标只统计变化起点之后的窗口（末端行号 >= onset），
常规流统计全部测试窗口。目标指标在有标签末端上计算；观测指标为观测头在有下一步的末端上、
各观测变量下一步记录在其余变量同时刻记录下（留一）的平均高斯负对数似然（标准化尺度）。

在线修订：冻结网络在一个评测流上逐步运行。网络在固定语义下的输出只依赖截至当前的窗口，
因此各语义的输出按批预先计算；
证据、信念与触发严格按时间顺序更新：时刻 t 的预测只用截至 t 已实现的记录（证据来自 x_{u+1}, u < t）。

- 条件：观测头读取同时刻已揭示的记录（``seer.methods.seer.visibility``）。监测、失配摘要与语义梯度用留一条件；
  事件内的证据在被检验集 U（各候选字段有改动的观测变量之并，候选冻结后确定）上按变量序链式揭示，
  原语义与全部候选都在 x_{U,t+1} 给定 F_t 与 x_{-U,t+1} 的同一条件联合似然上比较。
- 支撑：窗口末端各变量的记录都落在训练分位盒内的时刻；只有支撑内时刻更新证据与监测统计量。
- 事件：流开始后积累 W 个支撑内样本时开启一次；此后原语义经前序校准的标准化残差使单边 CUSUM 越界时再开启。
  开启事件后监测重新预热 W 个样本。新事件开启时关闭旧事件：旧事件若 w >= tau 且某候选份额过半，
  视为已审批，该候选成为新的原语义。
- 候选：在事件起点 s 之前确定并冻结，证据从 s 起累积；两层权重见 ``seer.methods.evidence``。
- 预测：q_t(S_0) g(S_0) + sum_k q_t(S_k) g(S_k)。

候选来源（``propose``）为 LLM，只看事件起点之前的数据。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from seer.changes import EvalStream, build_streams
from seer.config import dataset_spec
from seer.io import read_json, resolve_path
from seer.methods.evidence import (
    Computed,
    Monitor,
    Natural,
    Prequential,
    Profile,
    belief,
    event_mass,
    features,
    innovation_floors,
    supports,
)
from seer.methods.seer import visibility
from seer.semantic_cards import FIELDS
from seer.training.evaluation import grouped_metrics
from seer.training.runner import Prepared, Stream, evaluation_rng


def streams(p: Prepared, config: dict[str, Any], segments: np.ndarray | None = None) -> list[EvalStream]:
    """常规测试段（或给定的 ``segments``）上的常规、未知变化与漂移流。"""

    data = config["dataset"]
    changes = read_json(resolve_path(data["path"]) / "changes.json")
    stats = p.normalization
    return build_streams(
        p.dataset,
        p.split.test if segments is None else segments,
        changes,
        float(dataset_spec(data)["sampling_minutes"]),
        np.asarray(stats.input_mean, dtype=np.float64),
        np.asarray(stats.input_std, dtype=np.float64),
    )


def after_onset(stream: EvalStream, *, labelled: bool = False, has_next: bool = False) -> np.ndarray:
    ends = stream.dataset.ends(stream.segments, labelled=labelled, has_next=has_next)
    onset = stream.onsets[np.searchsorted(stream.segments, stream.dataset.segment_index(ends))]
    return ends[ends >= onset]


@torch.no_grad()
def static_metrics(
    p: Prepared, stream: EvalStream, fields: torch.Tensor | None, batch_size: int
) -> list[dict[str, object]]:
    """固定语义 ``fields`` (d, F, k) 下的目标与观测指标；Base 传 None。

    整个流一行、每个评测组一行（变化起点之后的有标签窗口）；观测 NLL 只在整体行。
    """

    model = p.model.eval()
    device = p.device
    data = Stream.build(stream.dataset, p.normalization, device)
    labelled = torch.as_tensor(after_onset(stream, labelled=True), device=device)
    following = torch.as_tensor(after_onset(stream, has_next=True), device=device)
    if fields is None and model.semantic:
        fields = model.fields
    with evaluation_rng(device):
        predictions = []
        for start in range(0, len(labelled), batch_size):
            batch = labelled[start : start + batch_size]
            semantics = None if fields is None else fields.expand(len(batch), *fields.shape)
            predictions.append(model(data.windows(batch), semantics)[0])
        target = p.normalization.inverse_targets(data.targets[labelled].cpu().numpy())
        prediction = p.normalization.inverse_targets(torch.cat(predictions).cpu().numpy())
        rows = grouped_metrics(p.split, stream.dataset, labelled.cpu().numpy(), target, prediction)
        out = rows[0]
        if model.semantic:
            losses = []
            loo = leave_one_out(model)
            for start in range(0, len(following), batch_size):
                batch = following[start : start + batch_size]
                semantics = fields.expand(len(batch), *fields.shape)
                nxt, _ = data.following(batch)
                _, (mean, scale) = model(data.windows(batch), semantics, nxt, loo)
                losses.append(model.observation.nll(nxt[:, model.observation.observed], mean, scale))
            nll = torch.cat(losses)
            out["obs_NLL"] = float(nll.mean())
            names = [p.dataset.feature_names[int(j)] for j in model.observation.observed]
            for variable in stream.variables:
                if variable in names:
                    out[f"obs_NLL.{variable}"] = float(nll[:, names.index(variable)].mean())
    return rows


@dataclass(frozen=True)
class Settings:
    budget: int  # K
    risk: float  # 每个测试流的错误越界风险预算 epsilon，按事件分配（``event_mass``）
    threshold: float  # tau
    alpha: float
    beta: float
    warmup: int  # W
    calibration_floor: float  # c_min：原族校准方差乘子的下界
    exact_floor: float  # 计算族噪声尺度的下界（数值精度）：精确算出的记录可被表达
    cusum_drift: float
    cusum_cap: float
    cusum_threshold: float | np.ndarray  # 逐变量阈值 (J,)（``monitor_reference``、``cusum_threshold``）
    batch_size: int
    monitor_level: np.ndarray | None = None  # 正常运行流上的残差水平 (J,)（``monitor_reference``）


@dataclass
class Semantics:
    """一套语义：字段嵌入、卡片文本，以及在流上各末端的输出（事件起点之前为 NaN）。"""

    fields: torch.Tensor
    cards: dict[str, dict[str, str]]
    prediction: np.ndarray  # (n, p) 标准化尺度
    mean: np.ndarray  # (n, J)
    scale: np.ndarray  # (n, J)
    note: dict[str, Any] = field(default_factory=dict)
    computed: np.ndarray | None = None  # (J,)：经批准的修订声明为算出的观测记录（相对原语义）


@dataclass
class Event:
    start: int
    trigger: str
    candidates: list[Semantics]
    record: dict[str, Any]
    rho: np.ndarray
    calibration: Prequential  # (K, J)：候选的头族 [mu_k, x, 1]
    computed: Computed  # (K, |U|)：候选的计算族（记录由卡片所写的来源算出），只在 U 上
    profile: Profile  # (J,)：原语义的头族
    natural: Natural  # (|U|,)：不比正常运行更精确的来源重构，属于原语义
    mean: np.ndarray  # (K + 1, n, J)：被检验集条件下原语义与各候选的条件均值
    scale: np.ndarray  # (n, J)：原语义的尺度；语义只经条件均值进入证据，不确定性是共同的数值干扰
    log_head: np.ndarray  # (K, J)：头族的前序对数似然
    tested: np.ndarray  # (J,)：被检验集 U = 各 U_k 之并
    revised: np.ndarray  # (K, J)：候选 k 的被检验集 U_k，它修订的记录
    visible: np.ndarray  # (J, d)：U 下的同时刻可见性
    pi: float  # 本事件的初始修订质量 pi_e
    burn_in: int  # W：前 W 个样本只用于学习，证据从此后计起
    trace: list[dict[str, Any]] = field(default_factory=list)
    scored: int = 0
    burned: np.ndarray | None = None  # 前 W 个样本上候选的对数边际 (K, J)
    burned_parts: tuple[np.ndarray, np.ndarray] | None = None  # 前 W 个样本上的头族 (K, J) 与重构族 (K, |U|)

    def log_m(self) -> np.ndarray:
        """候选的两个数值族各占先验 1/2：仍是测量（网络在新卡下的预测加校准），或由所写来源算出。"""

        computed = np.zeros_like(self.log_head)
        computed[:, self.tested] = self.computed.log_marginal()
        return np.logaddexp(self.log_head, computed) - math.log(2.0)

    def log_null(self) -> np.ndarray:
        """原语义的轮廓似然：头族与训练段来源重构的并集上取上确界。"""

        null = self.profile.log_likelihood()
        null[self.tested] = np.maximum(null[self.tested], self.natural.log_likelihood())
        return null

    def scored_step(self) -> None:
        """事件打开时原模型正处于局部失配（触发由此而来），任何灵活的数值族都能在局部胜出；前 W 个样本
        只用于学习：候选以其上的后验为先验，零族的上确界只取其后的样本。以 F_W 为条件，这是新的 e 过程。"""

        self.scored += 1
        if self.scored == self.burn_in:
            self.burned = self.log_m()
            self.burned_parts = (self.log_head.copy(), self.computed.log_marginal())
            self.computed.restart()
            self.profile.reset()
            self.natural.reset()

    def belief(self):
        if self.burned is None:
            zero = np.zeros_like(self.log_head)
            return belief(zero, zero[0], self.revised, self.rho, self.pi)
        return belief(self.log_m() - self.burned, self.log_null(), self.revised, self.rho, self.pi)


def leave_one_out(model) -> torch.Tensor:
    observed = model.observation.observed
    return visibility(
        observed, len(model.fields), torch.zeros(len(observed), dtype=torch.bool, device=observed.device)
    )


@torch.no_grad()
def outputs(
    model, data: Stream, ends: np.ndarray, fields: torch.Tensor, batch_size: int, start: int = 0, tested=None
):
    """语义 ``fields`` 下第 start 个及之后各末端的目标预测与观测头输出（之前为 NaN）。

    观测头条件于被检验集 ``tested``（bool (J,)）的链式揭示；None 时为留一条件。
    """

    n, device = len(ends), fields.device
    observed = model.observation.observed
    visible = (
        leave_one_out(model)
        if tested is None
        else visibility(observed, len(model.fields), torch.as_tensor(tested, device=device))
    )
    prediction = np.full((n, model.head.out_features), np.nan)
    mean = np.full((n, len(observed)), np.nan)
    scale = np.full((n, len(observed)), np.nan)
    index = torch.as_tensor(ends, device=device)
    with evaluation_rng(device):
        for a in range(start, n, batch_size):
            batch = index[a : a + batch_size]
            nxt = data.following(batch)[0]
            y, (mu, sigma) = model(
                data.windows(batch), fields.expand(len(batch), *fields.shape), nxt, visible
            )
            prediction[a : a + len(batch)] = y.cpu().numpy()
            mean[a : a + len(batch)] = mu.cpu().numpy()
            scale[a : a + len(batch)] = sigma.cpu().numpy()
    return prediction, mean, scale


def calibrate(mean: np.ndarray, scale: np.ndarray, following: np.ndarray, current: np.ndarray):
    """逐变量的原族最大似然：theta (J, 3)、c^2 (J,) 与每步校准后的原始空间 NLL (n, J)。"""

    phi, y = features(mean, scale, following, current)
    theta = np.stack([np.linalg.lstsq(phi[:, j], y[:, j], rcond=None)[0] for j in range(y.shape[1])])
    residual = y - np.einsum("njk,jk->nj", phi, theta)
    c2 = np.mean(residual**2, axis=0)
    nll = 0.5 * np.log(2 * math.pi * c2 * scale**2) + residual**2 / (2 * c2)
    return theta, c2, nll


def semantic_gradient(
    model, data: Stream, ends: np.ndarray, fields: torch.Tensor, theta, c2, batch_size: int
) -> np.ndarray:
    """gamma_{j,f} = || d/d e_{j,f} sum_u l_u^cal(e(S_0)) ||，校准参数取事件前的最大似然。"""

    device = fields.device
    leaf = fields.detach().clone().requires_grad_(True)
    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
    c2 = torch.as_tensor(c2, dtype=torch.float32, device=device)
    observed, loo = model.observation.observed, leave_one_out(model)
    index = torch.as_tensor(ends, device=device)
    with evaluation_rng(device), torch.backends.cudnn.flags(enabled=False):  # cuDNN RNN 只在训练模式下反传
        for a in range(0, len(ends), batch_size):
            batch = index[a : a + batch_size]
            nxt = data.following(batch)[0]
            _, (mu, sigma) = model(data.windows(batch), leaf.expand(len(batch), *leaf.shape), nxt, loo)
            x = nxt[:, observed]
            center = theta[:, 0] * mu + theta[:, 1] * data.values[batch][:, observed] + theta[:, 2]
            loss = torch.sum(torch.log(sigma) + 0.5 * torch.log(c2) + (x - center) ** 2 / (2 * c2 * sigma**2))
            loss.backward()
    return leaf.grad.norm(dim=-1).cpu().numpy()  # (d, F)


@dataclass(frozen=True)
class Reference:
    """训练段上的参照：原语义下各观测变量的校准 NLL，与输入各变量间的时滞相关。"""

    nll: np.ndarray  # (J,)，在 ends 上
    correlation: np.ndarray  # (d, d, Lr)：一步增量间的时滞相关


def lagged_correlation(values: np.ndarray, rows: np.ndarray, lags: int) -> np.ndarray:
    """一步增量间的时滞相关 corr(dx_{i,t}, dx_{j,t-l})；用增量以免短窗口内的趋势主导。

    values (T, d) 标准化流；rows 为当前时刻行号（其前 lags 行在同段内）。
    """

    step = np.diff(values, axis=0, prepend=values[:1])
    now = step[rows]
    out = np.full((values.shape[1], values.shape[1], lags), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (now - now.mean(0)) / now.std(0)
        for lag in range(lags):
            past = step[rows - lag]
            b = (past - past.mean(0)) / past.std(0)
            out[:, :, lag] = a.T @ b / len(rows)
    return out


def monitor(settings: Settings, level: np.ndarray, local: bool = True, threshold=None):
    s = settings
    return Monitor(
        level,
        s.warmup,
        s.alpha,
        s.beta,
        s.calibration_floor,
        s.cusum_cap,
        s.cusum_drift,
        s.cusum_threshold if threshold is None else threshold,
        local,
    )


def monitor_reference(mean, scale, following, current, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """原语义在正常运行流上的逐变量残差水平 (J,) 与 CUSUM 峰值 (J,)；用训练与验证段标定。"""

    J = mean.shape[1]
    probe = monitor(settings, np.ones(J), local=False, threshold=math.inf)
    r2 = np.stack([probe.residual(*features(mean[u], scale[u], following[u], current[u]), scale[u])
                   for u in range(len(mean))])
    level = r2.mean(axis=0)
    track, peak = monitor(settings, level, local=False, threshold=math.inf), np.zeros(J)
    for u in range(len(mean)):
        phi, y = features(mean[u], scale[u], following[u], current[u])
        track.update(phi, y, scale[u])
        np.maximum(peak, track.value, out=peak)
    return level, peak


def cusum_threshold(peak: np.ndarray, settings: Settings, margin: float) -> np.ndarray:
    """逐变量阈值 h_j = max(margin * peak_j, cap - 1 - drift)。

    CUSUM 最大值的零分布近似指数尾，P(max_{t<=H} C_j > h) ~ H K_j exp(-theta_j h)；标定峰值 peak_j 约为
    标定长度上恰被越过一次的水平，margin = 2 把越过概率再压低一个同样的因子（约一个标定长度的倒数）。
    逐变量标定使重尾、强自相关的变量不再决定其余变量的灵敏度；下界使单步的截断增量不会单独越界。
    """

    return np.maximum(margin * peak, settings.cusum_cap - 1.0 - settings.cusum_drift)


def summary(
    names: tuple[str, ...],
    observed: np.ndarray,
    nll: np.ndarray,
    gain: np.ndarray,
    c2: np.ndarray,
    gradient: np.ndarray,
    correlation: np.ndarray,
    reference: Reference,
    minutes: float,
) -> str:
    """失配摘要：校准后的 NLL 升幅、字段敏感性排序与失配变量的时滞相关变化。"""

    excess = nll - reference.nll
    order = np.argsort(-excess)
    lines = [
        "Recalibrated observation mismatch per observed variable (mean log-loss per step above its training "
        "level in nats; recalibrated gain on the network forecast; recalibrated noise scale relative to the "
        "network's own spread), most mismatched first:"
    ]
    for j in order:
        lines.append(
            f"- {names[observed[j]]}: {excess[j]:+.3f} nats, gain {gain[j]:.2f}, scale {math.sqrt(c2[j]):.2f}"
        )
    flat = gradient / gradient.max()
    top = np.argsort(-flat, axis=None)[:8]
    ranked = ", ".join(
        f"{names[i]}.{FIELDS[f]} {flat[i, f]:.2f}"
        for i, f in zip(*np.unravel_index(top, flat.shape), strict=True)
    )
    lines += ["", f"Field sensitivity of the recalibrated observation loss (largest = 1): {ranked}.", ""]
    lines.append(
        "Largest changes of lagged correlation for the most mismatched variables, corr(variable now, other "
        "variable lag minutes earlier), training -> now:"
    )
    change = np.abs(correlation - reference.correlation)
    for j in order[:3]:
        i = observed[j]
        row = change[i].copy()
        row[i, 0] = np.nan
        row = np.nan_to_num(row, nan=-1.0)
        for k, lag in zip(*np.unravel_index(np.argsort(-row, axis=None)[:5], row.shape), strict=True):
            before, now = reference.correlation[i, k, lag], correlation[i, k, lag]
            lines.append(f"- {names[i]} with {names[k]} at {lag * minutes:g} min: {before:.2f} -> {now:.2f}")
    return "\n".join(lines)


Proposer = Callable[[dict[str, Any]], list[tuple[torch.Tensor, dict[str, dict[str, str]], dict[str, Any]]]]


class Runner:
    """在一个评测流上运行 SEER（给定候选来源）。"""

    def __init__(
        self, p: Prepared, cards, box, reference: Reference, settings: Settings, minutes: float
    ) -> None:
        self.p, self.cards, self.box, self.reference = p, cards, box, reference
        self.settings, self.minutes = settings, minutes
        self.model = p.model.eval()
        self.observed = self.model.observation.observed.cpu().numpy()
        self.level = np.ones(len(self.observed)) if settings.monitor_level is None else settings.monitor_level
        train = p.dataset.ends(p.split.train, has_next=True)
        stream = Stream.build(p.dataset, p.normalization, p.device)
        _, _, scale = outputs(self.model, stream, train, self.model.fields, settings.batch_size)
        self.floors = innovation_floors(  # 各记录相对每个来源重构的新息下界（Natural），块长即预热长度 W
            stream.values.cpu().numpy(), train, stream.starts.cpu().numpy(), self.model.observation.lags,
            self.observed, scale, settings.warmup,
        )
        every = supports(len(self.model.fields))
        self.own = np.array([[int(j) in sup for sup in every] for j in self.observed])

    def run(self, stream: EvalStream, propose: Proposer) -> dict[str, Any]:
        s, model = self.settings, self.model
        data = Stream.build(stream.dataset, self.p.normalization, self.p.device)
        ends = stream.dataset.ends(stream.segments)
        index = torch.as_tensor(ends, device=self.p.device)
        following, exists = data.following(index)
        following = following[:, model.observation.observed].cpu().numpy()
        exists = exists.cpu().numpy()
        values = data.values.cpu().numpy()
        starts, lags = data.starts.cpu().numpy(), model.observation.lags

        def levels(u: int) -> np.ndarray:
            """(d, L)：x_{s,t+1-l}，段首之前取段首。"""

            return values[np.maximum(ends[u] + 1 - np.arange(lags), starts[ends[u]])].T

        current = values[ends][:, self.observed]
        support = np.all((values[ends] >= self.box[0]) & (values[ends] <= self.box[1]), axis=1)
        labelled = np.isfinite(stream.dataset.targets[ends]).all(axis=1)
        base = Semantics(
            self.model.fields, self.cards, *outputs(model, data, ends, model.fields, s.batch_size)
        )
        watch, arm, event, started = monitor(s, self.level), 0, None, False
        events, predictions = [], np.full((len(ends), base.prediction.shape[1]), np.nan)
        for i in range(len(ends)):
            u = i - 1
            if u >= 0 and exists[u] and support[u]:
                phi, y = features(base.mean[u], base.scale[u], following[u], current[u])
                fired = watch.update(phi, y, base.scale[u])
                if event is not None and event.candidates:
                    scale = event.scale[u]  # 原语义的尺度，各假设共用
                    sphi, sy = features(event.mean[:, u], scale, following[u], current[u])
                    event.log_head += event.calibration.update(sphi[1:], sy[1:], scale)[0]
                    U = event.tested
                    event.computed.update(levels(u), event.visible[U], following[u][U], scale[U])
                    event.profile.update(sphi[0], sy[0], scale)
                    event.natural.update(levels(u), following[u][U], scale[U])
                    event.scored_step()
                    b = event.belief()
                    event.trace.append(
                        {"position": i, "w": b.w, "log_e": b.log_e, "shares": b.shares.round(6).tolist()}
                    )
                    if b.w >= s.threshold and b.shares.max() > 0.5:  # 审批：候选成为新的原语义，重新预热
                        base, event = self._close(event, base, events), None
                        watch, arm = monitor(s, self.level), i
                        fired = False
                trigger = None
                if not started and watch.count == s.warmup:
                    trigger, started = "start", True
                elif fired:
                    trigger = "cusum"
                if trigger:
                    if event is not None:
                        base = self._close(event, base, events)
                    window = np.arange(arm, i)
                    window = window[exists[window] & support[window]]
                    event = self._open(
                        i,
                        trigger,
                        window,
                        ends,
                        data,
                        following,
                        current,
                        base,
                        propose,
                        stream,
                        len(events) + 1,
                    )
                    watch.refract()
                    arm = i
            if labelled[i]:
                if event is None or not event.candidates:
                    predictions[i] = base.prediction[i]
                else:
                    q = event.belief().weights()
                    stacked = np.stack([base.prediction[i]] + [c.prediction[i] for c in event.candidates])
                    predictions[i] = q @ stacked
        if event is not None:
            self._close(event, base, events)
        return {"ends": ends, "prediction": predictions, "events": events}

    def _open(
        self, i, trigger, window, ends, data, following, current, base, propose, stream, index
    ) -> Event:
        s, model = self.settings, self.model
        epsilon, pi = event_mass(s.risk, s.threshold, index)
        theta, c2, nll = calibrate(base.mean[window], base.scale[window], following[window], current[window])
        gradient = semantic_gradient(model, data, ends[window], base.fields, theta, c2, s.batch_size)
        lags = model.reader.lags
        rows = ends[window]
        correlation = lagged_correlation(data.values.cpu().numpy(), rows, lags)
        context = {
            "position": i,
            "trigger": trigger,
            "stream": stream,
            "base": base,
            "gradient": gradient,
            "summary": summary(
                self.p.dataset.feature_names,
                self.observed,
                nll.mean(0),
                theta[:, 0],
                c2,
                gradient,
                correlation,
                self.reference,
                self.minutes,
            ),
        }
        candidates, untestable = [], []
        for fields, cards, note in propose(context):
            # 候选只在它修订的记录上受检；只修订未被观测的输入时，它对任何记录的生成都没有声称，不可检验
            if not (fields != base.fields).flatten(1).any(dim=1)[model.observation.observed].any():
                untestable.append(note)
                continue
            out = outputs(model, data, ends, fields, s.batch_size, start=i)
            candidates.append(Semantics(fields, cards, *out, note=note))
        k = len(candidates)
        names = [self.p.dataset.feature_names[j] for j in self.observed]
        revised = np.zeros((k, len(names)), dtype=bool)
        for r, c in zip(revised, candidates, strict=True):
            r[:] = (c.fields != base.fields).flatten(1).any(dim=1)[model.observation.observed].cpu().numpy()
        tested = revised.any(axis=0)
        if not tested.any():  # 无可检验候选：事件不会越界，被检验集仅为保持形状
            tested[:] = True
        scored = [
            outputs(model, data, ends, c.fields, s.batch_size, i, tested)[1:] for c in [base, *candidates]
        ]
        record = {
            "position": i,
            "row": int(ends[i]),
            "trigger": trigger,
            "pre_event_samples": len(window),
            "summary": context["summary"],
            "gradient": gradient.round(6).tolist(),
            "candidates": [c.note for c in candidates],
            "untestable": untestable,
            "tested": [n for n, m in zip(names, tested, strict=True) if m],
            "proposal": context.get("proposal"),
            "index": index,
            "risk": epsilon,
            "pi": pi,
        }
        with torch.no_grad():  # 各候选卡计算字段的文本引用：计算族支撑的先验
            route = [model.observation.references(c.fields[None], 0)[0].cpu().numpy() for c in candidates]
        observed = model.observation.observed
        visible = visibility(observed, len(model.fields), torch.as_tensor(tested, device=observed.device))
        J, d = len(self.observed), len(model.fields)
        return Event(
            i,
            trigger,
            candidates,
            record,
            np.full(k, 1.0 / max(k, 1)),
            Prequential((k, J), s.alpha, s.beta, s.calibration_floor),
            Computed(  # 预热后以参考后验预测，尺度下界取数值精度：精确算出的记录可被表达
                (np.stack(route) if route else np.zeros((0, J, d)))[:, tested],
                model.observation.lags,
                s.alpha,
                s.beta,
                s.exact_floor,
            ),
            Profile((J,), s.calibration_floor),
            Natural(
                self._floors(base),
                np.flatnonzero(tested),
                visible.cpu().numpy()[tested],
                model.observation.lags,
                s.exact_floor,
            ),
            np.stack([m for m, _ in scored]),
            scored[0][1],
            np.zeros((k, J)),
            tested,
            revised,
            visible.cpu().numpy(),
            pi,
            s.warmup,
        )

    def _floors(self, base: Semantics) -> np.ndarray:
        """当前语义的新息下界：经批准声明为算出的记录可由任何来源精确重构（下界 0，由 Natural 取 epsilon^2），
        其上的再次来源替换不在检测范围内；其余记录用训练中的最小新息。"""

        floors = self.floors.copy()
        if base.computed is not None:
            floors[base.computed] = np.where(self.own[base.computed], np.inf, 0.0)
        return floors

    def _close(self, event: Event, base: Semantics, events: list) -> Semantics:
        s = self.settings
        record = event.record
        if event.candidates:
            b = event.belief()
            top = int(np.argmax(b.shares))
            approved = b.w >= s.threshold and b.shares[top] > 0.5
            ws = [t["w"] for t in event.trace]
            head0, computed0 = event.burned_parts or (0.0, 0.0)  # 分项取预热之后，与零族同一样本
            record.update(
                w=b.w,
                log_e=b.log_e,
                log_e_each=b.log_e_each.tolist(),
                shares=b.shares.tolist(),
                max_w=max(ws, default=0.0),
                crossed=max(ws, default=0.0) >= s.threshold,
                approved=bool(approved),
                samples=event.scored,
                trace=event.trace,
                parts={  # 对数似然分项：候选头族、重构族 (K, |U|)；原族头族、训练重构 (|U|,)
                    "head": (event.log_head - head0)[:, event.tested].round(3).tolist(),
                    "computed": (event.computed.log_marginal() - computed0).round(3).tolist(),
                    "profile": event.profile.log_likelihood()[event.tested].round(3).tolist(),
                    "natural": event.natural.log_likelihood().round(3).tolist(),
                    "each": ((event.log_m() - (event.burned if event.burned is not None else 0.0))
                             - event.log_null())[:, event.tested].round(3).tolist(),
                },
            )
            if approved:
                chosen = event.candidates[top]
                filled = np.isnan(chosen.prediction[:, 0])
                for name in ("prediction", "mean", "scale"):
                    getattr(chosen, name)[filled] = getattr(base, name)[filled]
                before = np.zeros(len(self.observed), dtype=bool) if base.computed is None else base.computed
                chosen.computed = before | event.revised[top]
                base = chosen
        events.append(record)
        return base
