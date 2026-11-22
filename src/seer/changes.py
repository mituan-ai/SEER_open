"""未知变化测试集与数值漂移对照（数据集目录中的 changes.json）。

测试流为常规测试段按顺序相接的序列。变化只改写记录的生成方式：从 onset（测试流内的位置比例）起，
把变量的记录替换为 ``record`` 表达式在原始量纲上的值，之后一直保持；过程与目标不变。
一个变化改写多个变量时写 ``records``（变量 -> 表达式），各表达式都基于改写前的记录计算。
表达式只含列名、数字、算术与下列因果算子
（时间参数以分钟计，按数据集的采样间隔换算为步，段起点之前的历史按起点补齐）：

    lag(x, m)    m 分钟前的值
    mean(x, m)   最近 m 分钟（含当前）的平均
    ewm(x, m)    时间常数 m 分钟的一阶滞后
    hold(x, m)   每 m 分钟采样一次并保持
    sqrt, exp, log, abs, maximum, minimum, clip

带 ``control`` 的变化（字段写理由）归入对照组，与漂移一起检验修订不造成损害。
漂移对照作用在同一测试流上，以训练段统计量为尺度：
    bias_std   x + c * std            （族内：固定偏置）
    gain       m + c * (x - m)        （族内：固定增益，m 为训练均值）
    ramp_gain  m + g(t) * (x - m)     （族外：增益自 onset 起线性缓变，到测试流末端为 c）
    scenarios  直接取数据集中满足规则的段（族外：负荷或工况变化），不改写记录
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from seer.data import StreamDataset, select

FUNCTIONS = ("lag", "mean", "ewm", "hold", "sqrt", "exp", "log", "abs", "maximum", "minimum", "clip")


@dataclass(frozen=True)
class EvalStream:
    """一个评测流：数据（记录可能已被改写）、参与评测的段与各段的变化起点行号。"""

    name: str
    kind: str  # regular / change / control（不在检测范围内的变化）/ drift_in / drift_out
    dataset: StreamDataset
    segments: np.ndarray
    onsets: np.ndarray  # 各段变化起点的行号；起点之前的段为段终点，regular 为段起点
    variables: tuple[str, ...]
    cards: dict[str, dict[str, str]]  # 变化的语义卡（可选；随仓库的 changes.json 不提供）


def _steps(minutes: float, sampling: float) -> int:
    return max(int(round(float(minutes) / sampling)), 0)


def evaluate(expression: str, columns: dict[str, np.ndarray], sampling: float) -> np.ndarray:
    """在一段 (T,) 列上计算表达式。"""

    def lag(x, m):
        k = _steps(m, sampling)
        return np.concatenate([np.repeat(x[:1], k), x[: len(x) - k]])[: len(x)]

    def mean(x, m):
        k = max(_steps(m, sampling), 1)
        padded = np.concatenate([np.repeat(x[:1], k - 1), x])
        return np.convolve(padded, np.ones(k) / k, mode="valid")

    def ewm(x, m):
        alpha = 1.0 - np.exp(-sampling / float(m))
        y = np.empty_like(x)
        y[0] = x[0]
        for s in range(1, len(x)):
            y[s] = y[s - 1] + alpha * (x[s] - y[s - 1])
        return y

    def hold(x, m):
        k = max(_steps(m, sampling), 1)
        steps = np.arange(len(x))
        return x[steps - steps % k]

    names = {
        **columns,
        "lag": lag,
        "mean": mean,
        "ewm": ewm,
        "hold": hold,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "abs": np.abs,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "clip": np.clip,
    }
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load)
        if not isinstance(node, allowed + (ast.operator, ast.unaryop)):
            raise ValueError(f"unsupported syntax in {expression!r}")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"unknown name {node.id!r} in {expression!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS):
            raise ValueError(f"unsupported call in {expression!r}")
    value = eval(compile(tree, "<record>", "eval"), {"__builtins__": {}}, names)
    return np.broadcast_to(np.asarray(value, dtype=np.float64), (len(next(iter(columns.values()))),))


def _rewrite(
    dataset: StreamDataset, segments: np.ndarray, onsets: np.ndarray, variables: tuple[str, ...], record
) -> StreamDataset:
    """``record(block, position)`` 给出一段各变量的新记录 (T, len(variables))；
    ``position`` 为各行在测试流中的位置。"""

    values = dataset.values.copy()
    columns = [dataset.feature_names.index(v) for v in variables]
    offset = 0
    for index, onset in zip(segments, onsets, strict=True):
        segment = dataset.segments[index]
        length = segment.stop - segment.start
        new = record(values[segment.start : segment.stop].astype(np.float64), offset + np.arange(length))
        values[onset : segment.stop, columns] = new.reshape(length, -1)[onset - segment.start :]
        offset += length
    return replace(dataset, values=values)


def build_streams(
    dataset: StreamDataset,
    test: np.ndarray,
    changes: dict[str, Any],
    sampling: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[EvalStream]:
    """常规测试流、各变化 x 各起点的未知变化流，以及漂移对照流。

    ``test`` 为常规测试段；``mean``、``std`` 为训练段输入统计量（原始量纲）。
    """

    names = dataset.feature_names
    starts = np.array([dataset.segments[i].start for i in test])
    lengths = np.array([dataset.segments[i].stop - dataset.segments[i].start for i in test])
    before = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    total = int(lengths.sum())
    streams = [EvalStream("regular", "regular", dataset, test, starts, (), {})]
    for fraction in changes["onsets"]:
        position = int(np.floor(float(fraction) * total))
        onsets = starts + np.clip(position - before, 0, lengths)
        tag = f"onset{int(round(100 * float(fraction))):02d}"
        for change in changes["changes"]:
            records = change.get("records") or {change["variable"]: change["record"]}

            def record(block, _, records=records):
                columns = {name: block[:, i] for i, name in enumerate(names)}
                return np.stack([evaluate(e, columns, sampling) for e in records.values()], axis=1)

            changed = _rewrite(dataset, test, onsets, tuple(records), record)
            streams.append(
                EvalStream(
                    f"{change['id']}@{tag}",
                    "control" if change.get("control") else "change",
                    changed,
                    test,
                    onsets,
                    tuple(records),
                    change.get("cards", {}),
                )
            )
        for drift in changes.get("drifts", []):
            if "scenarios" in drift:
                continue
            j = names.index(drift["variable"])
            m, s = float(mean[j]), float(std[j])

            def record(block, at, drift=drift, j=j, m=m, s=s, position=position):
                x = block[:, j]
                if "bias_std" in drift:
                    return x + float(drift["bias_std"]) * s
                if "gain" in drift:
                    return m + float(drift["gain"]) * (x - m)
                ramp = np.clip((at - position) / max(total - 1 - position, 1), 0.0, 1.0)
                return m + (1.0 + (float(drift["ramp_gain"]) - 1.0) * ramp) * (x - m)

            changed = _rewrite(dataset, test, onsets, (drift["variable"],), record)
            kind = f"drift_{drift['family']}"
            streams.append(
                EvalStream(f"{drift['id']}@{tag}", kind, changed, test, onsets, (drift["variable"],), {})
            )
    for drift in changes.get("drifts", []):
        if "scenarios" in drift:
            segments = select(dataset, drift["scenarios"])
            onsets = np.array([dataset.segments[i].start for i in segments])
            streams.append(
                EvalStream(drift["id"], f"drift_{drift['family']}", dataset, segments, onsets, (), {})
            )
    return streams
