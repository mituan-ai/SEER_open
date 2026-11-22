"""离线构建合成语义先验的描述池。

1. 抽样：在训练段的标准化逐步流上随机抽取变换 x~_j = a * sum_l h_l x_{k,t-l} + b + eps，
   可叠加采样保持。j 只在 observed 中取，k 以 ``self_source_probability`` 取 j 自身，否则取其他任一非恒定变量
   （记录可改由任何被测量的量算出，含操作输入）。核为四族之一：同时刻估算（estimate，h = delta_0，
   只用于其他来源）、纯延迟（delay）、一阶平滑加死区（smooth）、滑动平均加死区（spread），
   截断到 lmax 并归一化。
   增益与偏置以矩匹配为中心（a* = sign(corr) std(x_j) / std(u)，u 为核作用后的来源），再加相对扰动；
   噪声标准差为 x_j 一步增量标准差的随机分数（观测头读取一步增量，按水平标准差取的噪声会淹没增量）。
   原语义的数值适应族是逐变量仿射校准，所以 x~_j 对 x_j 的
   仿射拟合残差不足 ``min_departure``（相对标准差）的变换与原语义等价或在数值上可忽略，予以排除。
   分布参数来自数据集目录的 seer.json。
2. 描述：变换参数换算为物理语言（来源变量名、以分钟计的时滞、平滑与保持、增益方向与大致倍数、噪声），
   连同原语义卡与工艺资料交给 LLM；LLM 只改写受影响变量中被该变化改写的字段，每个变换给出 3 种措辞。
   每次调用描述 ``BATCH`` 个变换；请求错误、截断或校验失败重试一次。
3. 编码：改写字段用默认编码器编码，未改写字段沿用原嵌入。

材料只有训练段数据、语义卡与工艺资料；不读取 changes.json、测试数据或测试统计量。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import numpy as np

from seer.io import sha256_json, sha256_text
from seer.llm import TRUNCATED, Caller, LLMConfig, call_llm, parse_json_response
from seer.methods.prior import Transforms, apply_stream
from seer.semantic_cards import _CJK, FIELDS, Materials, _materials_section

PROMPT_VERSION = "seer-prior-descriptions-v1"
PHRASINGS = 3
BATCH = 10
ATTEMPTS = 3
KERNELS = ("estimate", "delay", "smooth", "spread")  # estimate：由另一变量在同一时刻算出


def build_kernel(kind: str, lmax: int, rng: np.random.Generator, prior: dict[str, Any], self_source: bool):
    """返回 (h, 描述参数)。自身来源的纯延迟至少 1 步，且没有同时刻估算（恒等）。"""

    h = np.zeros(lmax + 1)
    if kind == "estimate":
        h[0] = 1.0
        return h, {"kind": kind, "dead_time": 0}
    if kind == "delay":
        lag = int(rng.integers(1 if self_source else 0, lmax + 1))
        h[lag] = 1.0
        return h, {"kind": kind, "dead_time": lag}
    dead = int(rng.integers(0, lmax // 2 + 1))
    lags = np.arange(lmax + 1 - dead)
    if kind == "smooth":
        tau = float(rng.choice(prior["smoothing_steps"]))
        h[dead:] = np.exp(-lags / tau)
        detail = {"kind": kind, "dead_time": dead, "time_constant": tau}
    else:
        width = int(min(rng.choice(prior["spread_steps"]), lmax + 1 - dead))
        h[dead : dead + width] = 1.0
        detail = {"kind": kind, "dead_time": dead, "width": width}
    return h / h.sum(), detail


def sample_transforms(
    values: np.ndarray,
    starts: list[tuple[int, int]],
    observed: tuple[int, ...],
    prior: dict[str, Any],
    count: int,
    seed: int,
) -> tuple[Transforms, list[dict[str, Any]], int]:
    """在标准化训练流 ``values`` (T, d) 上抽取 ``count`` 个有效变换；``starts`` 为各段 [start, stop)。
    返回变换、各变换的描述参数与被排除的候选数。"""

    rng = np.random.default_rng(seed)
    lmax = int(prior["max_lag"])
    train = np.concatenate([np.arange(a, b) for a, b in starts])
    # 来源：训练段上任一非恒定变量（记录可改由任何被测量的量算出，含操作输入），被改变量是观测变量
    sources = [v for v in range(values.shape[1]) if values[train, v].std() > 1e-6]
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    rejected = 0
    while len(rows) < count:
        j = int(rng.choice(observed))
        self_source = rng.random() < float(prior["self_source_probability"])
        k = j if self_source else int(rng.choice([v for v in sources if v != j]))
        kind = str(rng.choice(KERNELS[1:] if self_source else KERNELS))
        h, detail = build_kernel(kind, lmax, rng, prior, self_source)
        hold = int(rng.choice(prior["hold_steps"])) if rng.random() < float(prior["hold_probability"]) else 1
        u = np.concatenate([_filtered(values[a:b, k], h) for a, b in starts])
        xj = values[train, j]
        corr = float(np.corrcoef(u, xj)[0, 1]) if u.std() > 1e-9 else 0.0
        sign = -1.0 if corr < 0 else 1.0
        g = float(prior["gain_perturbation"])
        relative = 1.0 + rng.uniform(-g, g)
        gain = sign * xj.std() / max(u.std(), 1e-9) * relative
        bias = xj.mean() - gain * u.mean() + xj.std() * rng.uniform(-g, g)
        step = np.concatenate([np.diff(values[a:b, j]) for a, b in starts]).std()
        noise = step * rng.uniform(*prior["noise_fraction"])  # 与一步增量同一尺度：读取的是增量
        candidate = Transforms(
            *(np.asarray([v]) for v in (j, k)), h[None], *(np.asarray([v]) for v in (gain, bias, noise, hold))
        )
        changed = np.concatenate([apply_stream(values[a:b], candidate, 0)[:, j] for a, b in starts])
        departure = _affine_residual(changed, xj)
        if u.std() <= 1e-6 or departure < float(prior["min_departure"]):  # 滤波后恒定：范围匹配增益无界
            rejected += 1
            continue
        rows.append(
            {"target": j, "source": k, "kernel": h, "gain": gain, "bias": bias, "noise": noise, "hold": hold}
        )
        details.append(
            {
                **detail,
                "hold": hold,
                "relative_gain": relative,
                "sign": "negative" if sign < 0 else "positive",
                "noise_fraction": float(noise / max(step, 1e-12)),
                "correlation": corr,
                "departure": departure,
            }
        )
    transforms = Transforms(
        target=np.asarray([r["target"] for r in rows], dtype=np.int64),
        source=np.asarray([r["source"] for r in rows], dtype=np.int64),
        kernel=np.stack([r["kernel"] for r in rows]),
        gain=np.asarray([r["gain"] for r in rows]),
        bias=np.asarray([r["bias"] for r in rows]),
        noise=np.asarray([r["noise"] for r in rows]),
        hold=np.asarray([r["hold"] for r in rows], dtype=np.int64),
    )
    return transforms, details, rejected


def _filtered(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    steps = np.arange(len(x))
    return x[np.maximum(steps[:, None] - np.arange(h.size), 0)] @ h


def _affine_residual(changed: np.ndarray, original: np.ndarray) -> float:
    """changed 对 original 的仿射最小二乘残差标准差，相对 changed 的标准差。"""

    design = np.c_[original, np.ones_like(original)]
    coef, *_ = np.linalg.lstsq(design, changed, rcond=None)
    return float(np.std(changed - design @ coef) / max(np.std(changed), 1e-12))


def _minutes(steps: float, sampling: float) -> str:
    value = steps * sampling
    if value >= 120:
        return f"{value / 60:.3g} h"
    return f"{value:.3g} min"


def physical_description(
    names: tuple[str, ...], j: int, k: int, detail: dict[str, Any], sampling: float
) -> str:
    """把变换参数写成工程语言，不含公式。"""

    target, source = names[j], names[k]
    if k == j:
        origin = f"The record of {target} is now derived from its own measurement"
    else:
        origin = f"The record of {target} is no longer its own measurement; it is now computed from {source}"
    dead = int(detail["dead_time"])
    timing = []
    if detail["kind"] in ("estimate", "delay"):
        timing.append(
            f"taken {_minutes(dead, sampling)} earlier"
            if dead
            else "taken at the same time step, without delay"
        )
    elif detail["kind"] == "smooth":
        if dead:
            timing.append(f"with a transport delay of {_minutes(dead, sampling)}")
        timing.append(
            f"passed through a first-order smoothing filter with a time constant of "
            f"{_minutes(detail['time_constant'], sampling)}"
        )
    else:
        if dead:
            timing.append(f"with a transport delay of {_minutes(dead, sampling)}")
        timing.append(
            f"averaged over the last {_minutes(detail['width'], sampling)} "
            f"({detail['width']} samples), which spreads its response in time"
        )
    text = f"{origin}, {', '.join(timing)}."
    if detail["hold"] > 1:
        period = _minutes(detail["hold"], sampling)
        text += f" The value is refreshed only every {period} and held constant in between."
    if k != j:
        direction = (
            "rises when the source rises" if detail["sign"] == "positive" else "falls when the source rises"
        )
        text += (
            f" A static conversion maps the source onto the usual range of {target}; the converted value "
            f"{direction}, with a gain about {detail['relative_gain']:.2g} times the range-matched "
            "conversion."
        )
    elif abs(detail["relative_gain"] - 1.0) > 0.05:
        text += f" The recorded value is scaled by about {detail['relative_gain']:.2g} relative to before."
    if detail["noise_fraction"] > 0.005:
        text += (
            f" Random noise of about {100 * detail['noise_fraction']:.2g}% of its typical step-to-step "
            "variation is added."
        )
    return text


_TASK = (
    "You maintain the variable documents of an industrial prediction task. Each input variable has a card "
    "with four text fields:\n"
    "1. source: what the recorded value is and where it comes from, with its unit and sampling.\n"
    "2. computation: how the recorded value is obtained: a direct reading, a schedule, a control action, or "
    "a calculation from other quantities.\n"
    "3. role: what the variable means for the process and how it relates to the prediction target.\n"
    "4. dependency: which other input variables drive it, through which mechanism, and how quickly.\n\n"
    "Each change below alters how one recorded variable is produced. The process itself and the prediction "
    "target are unchanged; only the meaning of the record changed. For every change, rewrite only those "
    "fields of the affected variable's card that the change makes wrong, and keep the other fields. "
    "Write the rewritten fields as the new card would read after the change, in plain technical English, "
    "without formulas or code. Give "
    f"{PHRASINGS} alternative phrasings per change: each phrasing states the same facts in different words "
    "and covers the same fields."
)
_OUTPUT = (
    "Return strict JSON with this shape:\n"
    '{"changes": [{"id": "change id", "fields": ["source", "computation", ...], '
    '"phrasings": [{"source": "...", "computation": "..."}, ...]}]}\n'
    "Rules:\n"
    "1. One entry per change, with the given id.\n"
    "2. fields lists the rewritten fields, a non-empty subset of source, computation, role, dependency.\n"
    f"3. phrasings has exactly {PHRASINGS} objects, each with exactly the listed fields.\n"
    "4. When the change computes the record from another variable, name that variable exactly as given.\n"
    "5. State delays, filter time constants and refresh periods in minutes or hours as given.\n"
    "6. Return JSON only, without markdown."
)


def build_prompt(materials: Materials, cards: list[dict[str, Any]], items: list[dict[str, Any]]) -> str:
    card_block = json.dumps(
        [{k: card[k] for k in ("variable_name", *FIELDS)} for card in cards], ensure_ascii=False, indent=1
    )
    changes = "\n".join(
        f"- id {item['id']} (variable {item['target']}): {item['description']}" for item in items
    )
    return (
        "You are revising variable documents after plant changes.\n"
        f"Prompt version: {PROMPT_VERSION}\n\n{_TASK}\n\n{_materials_section(materials)}\n\n"
        f"<current_cards>\n{card_block}\n</current_cards>\n\n<changes>\n{changes}\n</changes>\n\n{_OUTPUT}"
    )


def validate(
    payload: dict[str, Any], items: list[dict[str, Any]], cards: dict[str, dict[str, Any]]
) -> list[dict]:
    entries = {str(entry.get("id")): entry for entry in payload.get("changes", []) if isinstance(entry, dict)}
    if set(entries) != {item["id"] for item in items}:
        raise ValueError(f"expected changes {[i['id'] for i in items]}, got {sorted(entries)}")
    names = sorted(cards, key=len, reverse=True)  # 变量名可为中文，按要求原样出现在描述中
    result = []
    for item in items:
        entry = entries[item["id"]]
        fields = entry.get("fields")
        if (
            not isinstance(fields, list)
            or not fields
            or not set(fields) <= set(FIELDS)
            or len(set(fields)) < len(fields)
        ):
            raise ValueError(f"change {item['id']} has invalid fields {fields!r}")
        fields = [f for f in FIELDS if f in fields]
        phrasings = entry.get("phrasings")
        if not isinstance(phrasings, list) or len(phrasings) != PHRASINGS:
            raise ValueError(f"change {item['id']} needs {PHRASINGS} phrasings")
        texts = []
        for phrasing in phrasings:
            if not isinstance(phrasing, dict) or set(phrasing) != set(fields):
                raise ValueError(f"change {item['id']} phrasing does not cover exactly {fields}")
            clean = {f: " ".join(str(phrasing[f]).split()) for f in fields}
            for f, text in clean.items():
                prose = text
                for name in names:
                    prose = prose.replace(name, "")
                if not text or _CJK.search(prose) or text == cards[item["target"]][f]:
                    raise ValueError(f"change {item['id']} field {f} is empty, not English, or unchanged")
            if item["source"] != item["target"] and not any(item["source"] in t for t in clean.values()):
                raise ValueError(f"change {item['id']} does not name its source variable {item['source']}")
            texts.append(clean)
        if len({json.dumps(t, sort_keys=True) for t in texts}) < PHRASINGS:
            raise ValueError(f"change {item['id']} repeats a phrasing")
        result.append({"id": item["id"], "fields": fields, "phrasings": texts})
    return result


def describe(
    materials: Materials,
    cards: list[dict[str, Any]],
    items: list[dict[str, Any]],
    config: LLMConfig,
    *,
    caller: Caller = call_llm,
    workers: int = 6,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """按批调用 LLM；返回与 ``items`` 对齐的描述（任一批失败时为 None）与全部调用记录。"""

    by_name = {card["variable_name"]: card for card in cards}
    batches = [items[s : s + BATCH] for s in range(0, len(items), BATCH)]

    def one(batch: list[dict[str, Any]]) -> tuple[list[dict] | None, dict[str, Any]]:
        prompt = build_prompt(materials, cards, batch)
        record: dict[str, Any] = {
            "ids": [i["id"] for i in batch],
            "prompt_sha256": sha256_text(prompt),
            "attempts": [],
        }
        for number in range(1, ATTEMPTS + 1):
            attempt: dict[str, Any] = {"attempt": number}
            try:
                reply = caller(prompt, config)
                attempt.update(
                    response_model=reply.model,
                    finish_reason=reply.finish_reason,
                    usage=reply.usage,
                    seconds=round(reply.seconds, 3),
                    raw_response=reply.text,
                )
                if reply.finish_reason in TRUNCATED:
                    raise RuntimeError(f"output ended with finish_reason={reply.finish_reason}")
                described = validate(parse_json_response(reply.text), batch, by_name)
            except Exception as error:  # 记录失败并重试
                record["attempts"].append(
                    {**attempt, "status": "failed", "error": f"{type(error).__name__}: {error}"}
                )
                continue
            record["attempts"].append({**attempt, "status": "accepted"})
            return described, record
        return None, record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(one, batches))
    records = [record for _, record in outcomes]
    if any(described is None for described, _ in outcomes):
        return None, records
    return [entry for described, _ in outcomes for entry in described], records


def usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [a for r in records for a in r["attempts"]]
    return {
        "calls": len(attempts),
        "seconds": round(sum(a.get("seconds", 0.0) for a in attempts), 3),
        **{
            key: sum(a.get("usage", {}).get(key, 0) for a in attempts)
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
        },
    }


def coverage(
    transforms: Transforms,
    details: list[dict[str, Any]],
    rejected: int,
    names: tuple[str, ...],
    sampling: float,
) -> dict:
    """变换族的覆盖：受影响变量、来源、核族、时滞、平滑与保持的分布。"""

    def counts(values) -> dict[str, int]:
        keys, number = np.unique(np.asarray(values, dtype=object).astype(str), return_counts=True)
        return {str(k): int(n) for k, n in zip(keys, number, strict=True)}

    centroid = (transforms.kernel * np.arange(transforms.kernel.shape[1])).sum(axis=1)
    spread = np.sqrt(
        (transforms.kernel * (np.arange(transforms.kernel.shape[1]) - centroid[:, None]) ** 2).sum(axis=1)
    )
    return {
        "transforms": len(transforms),
        "target": counts([names[j] for j in transforms.target]),
        "source": counts(
            ["self" if j == k else "other" for j, k in zip(transforms.target, transforms.source, strict=True)]
        ),
        "kernel": counts([d["kind"] for d in details]),
        "hold_steps": counts(transforms.hold),
        "mean_lag_minutes": _quantiles(centroid * sampling),
        "lag_spread_minutes": _quantiles(spread * sampling),
        "departure": _quantiles([d["departure"] for d in details]),
        "rejected": rejected,
    }


def _quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)}


def stamp(
    materials: Materials, cards: list[dict[str, Any]], prior: dict[str, Any], count: int, seed: int
) -> dict:
    return {
        "prompt_version": PROMPT_VERSION,
        "phrasings": PHRASINGS,
        "count": count,
        "seed": seed,
        "prior": prior,
        "cards_sha256": sha256_json([{k: c[k] for k in FIELDS} for c in cards]),
        **materials.digests(),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
