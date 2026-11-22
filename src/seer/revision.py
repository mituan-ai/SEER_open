"""修订事件中的 LLM 联合候选。

LLM 读取任务与工艺材料、当前语义卡与失配摘要，给出至多 K 条联合修订：每条改写一个或多个变量的若干字段，
附依据。LLM 只提供语义文本，不写公式或程序；修订的效果由冻结网络执行、由证据裁决。
提示词不含数据划分、评测场景、描述池或真实变更。回答按提示词与模型设置的哈希缓存在 ``directory``，
重跑时直接读取；失败不缓存。连接故障不计入尝试次数，退避等待网络恢复。
多个进程并行评测时，同一提示词由先到的进程调用（按哈希加文件锁），其余进程等待并读取其缓存，与单进程
顺序运行时相同；缓存先写临时文件再改名，不会留下写了一半的记录。
"""

from __future__ import annotations

import fcntl
import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from seer.io import sha256_json, write_json
from seer.llm import TRUNCATED, Caller, LLMConfig, call_llm, parse_json_response, patiently
from seer.semantic_cards import FIELDS, Materials, _materials_section

PROMPT_VERSION = "seer-revision-v1"
ATTEMPTS = 4
LOCKS = Path(tempfile.gettempdir()) / "seer_proposal_locks"  # 锁文件不放进需要提交的缓存目录


@contextmanager
def _exclusive(name: str) -> Iterator[None]:
    LOCKS.mkdir(parents=True, exist_ok=True)
    with (LOCKS / f"{name}.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

_TASK = """\
A deployed prediction network reads one semantic card per input variable. Each card has four fields:
- source: what the recorded value is and where it comes from (instrument, setpoint, controller output or
  calculation), with unit and sampling;
- computation: how the recorded value is obtained (direct reading, given profile, control action, or
  calculated from which inputs);
- role: what the variable does in the process and how it relates to the prediction target;
- dependency: which inputs drive the variable in operation, through what mechanism, and how fast.

The network was trained to change how it reads its inputs when a card describes a different recording,
for example a value now calculated from other variables, delayed, filtered, or sampled and held.
After deployment, the observation model of the network no longer explains the recorded inputs well, even after
a per-variable affine recalibration. The process itself and its relation to the target are assumed unchanged;
what may have changed is what some records mean: their source, their computation, or their timing.

Propose up to {budget} alternative explanations. Each is one physically coherent event that may change several
fields of one or more variables at once (for example, a failed meter replaced by an estimate changes both its
source and its computation, and its dependency on other variables). The mismatch summary points to where the
current cards may be wrong, but it is not a conclusion; you may revise other variables when the process
makes it plausible. Distinct candidates should be genuinely different explanations, ordered from most to
least plausible.

Write each revised field as a complete replacement in English, in the same style as the current cards.
State approximate quantities in words where the summary supports them (for example a delay or a sampling
period in minutes). Do not write equations, formulas, code or coefficients."""

_OUTPUT = """\
Return one JSON object, nothing else:
{{"candidates": [{{"revisions": {{"<variable name>": {{"<field>": "<replacement text>"}}}},
                  "rationale": "<why the mismatch supports this explanation>"}}]}}
Use variable names exactly as listed and fields among {fields}. Give 1 to {budget} candidates."""


def build_prompt(materials: Materials, cards: dict[str, dict[str, str]], summary: str, budget: int) -> str:
    current = [{"variable_name": name, **cards[name]} for name in materials.task.feature_names]
    return (
        "You are revising variable documents of a deployed industrial prediction model.\n"
        f"Prompt version: {PROMPT_VERSION}\n\n"
        f"{_TASK.format(budget=budget)}\n\n{_materials_section(materials)}\n\n"
        f"<current_cards>\n{json.dumps(current, ensure_ascii=False, indent=1)}\n</current_cards>\n\n"
        f"<mismatch_summary>\n{summary.strip()}\n</mismatch_summary>\n\n"
        f"{_OUTPUT.format(budget=budget, fields=', '.join(FIELDS))}"
    )


def validate(payload: dict[str, Any], names: tuple[str, ...], budget: int) -> list[dict[str, Any]]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= budget:
        raise ValueError(f"expected 1 to {budget} candidates")
    out = []
    for candidate in candidates:
        revisions = candidate.get("revisions")
        if not isinstance(revisions, dict) or not revisions:
            raise ValueError("each candidate needs revisions")
        for name, row in revisions.items():
            if name not in names or not isinstance(row, dict) or not row:
                raise ValueError(f"unknown variable or empty revision: {name!r}")
            for field, text in row.items():
                if field not in FIELDS or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"invalid field revision {name}.{field}")
        revisions = {name: {f: t.strip() for f, t in row.items()} for name, row in revisions.items()}
        if revisions not in [c["revisions"] for c in out]:
            out.append({"revisions": revisions, "rationale": str(candidate.get("rationale", "")).strip()})
    return out


def propose(
    materials: Materials,
    cards: dict[str, dict[str, str]],
    summary: str,
    budget: int,
    config: LLMConfig,
    directory,
    *,
    caller: Caller = call_llm,
) -> dict[str, Any]:
    """一个事件的候选记录：``candidates``、提示词、原始回答与用量；全部尝试失败时报错，不以空候选冒充修订关闭。"""

    prompt = build_prompt(materials, cards, summary, budget)
    path = directory / f"{sha256_json({'prompt': prompt, 'llm': config.public()})[:20]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with _exclusive(path.stem):  # 并行进程遇到同一提示词时只调用一次
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return _call(prompt, materials, budget, config, path, caller)


def _call(prompt: str, materials: Materials, budget: int, config: LLMConfig, path, caller: Caller):
    attempts = []
    for _ in range(ATTEMPTS):
        try:
            reply = patiently(lambda: caller(prompt, config))
            if reply.finish_reason in TRUNCATED:
                raise RuntimeError(f"finish_reason={reply.finish_reason}")
            candidates = validate(parse_json_response(reply.text), materials.task.feature_names, budget)
        except Exception as error:  # 回答失败记录后重试；连接故障已在 patiently 内等待
            attempts.append({"error": f"{type(error).__name__}: {error}"})
            continue
        record = {
            "prompt_version": PROMPT_VERSION,
            "llm": config.public(),
            "model": reply.model,
            "usage": reply.usage,
            "seconds": reply.seconds,
            "failed_attempts": attempts,
            "candidates": candidates,
            "prompt": prompt,
            "reply": reply.text,
        }
        partial = path.with_name(path.name + ".partial")
        write_json(partial, record)
        partial.replace(path)
        return record
    raise RuntimeError(f"revision proposal failed: {attempts}")
