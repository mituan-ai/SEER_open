from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seer.config import dataset_config
from seer.data import read_feature_names
from seer.llm import LLMConfig, LLMReply
from seer.semantic_artifact import build_random, load_artifact
from seer.semantic_cards import (
    FIELDS,
    PIPELINE_VERSION,
    build_prompt,
    dataset_materials,
    generate_cards,
    load_cards,
    load_materials,
    save_cards,
    validate_cards,
)

DATASETS = ("ladle_preheating", "thickener_dewatering", "indpensim")


def test_artifact_round_trip(tmp_path: Path) -> None:
    original = build_random(("a", "b", "c"), dim=256, seed=1)
    original.save(tmp_path / "artifact")
    loaded = load_artifact(tmp_path / "artifact", original.variable_names)
    np.testing.assert_array_equal(original.fields, loaded.fields)
    with pytest.raises(ValueError, match="feature order"):
        load_artifact(tmp_path / "artifact", ("b", "a", "c"))


def _card(name: str, suffix: str = "", parents: tuple[str, ...] = ()) -> dict:
    return {
        "variable_name": name,
        **{field: f"The {field} of {name}{suffix}." for field in FIELDS},
        "parents": list(parents),
        "computed_from": [],
        "rationale": {"basis": "process_knowledge", "evidence": "general process knowledge"},
    }


def test_card_validation() -> None:
    cards = validate_cards({"cards": [_card("a"), _card("b", parents=("a",))]}, ("a", "b"))
    assert [card["parents"] for card in cards] == [[], ["a"]]
    with pytest.raises(ValueError, match="order"):
        validate_cards({"cards": [_card("b"), _card("a")]}, ("a", "b"))
    with pytest.raises(ValueError, match="other input"):
        validate_cards({"cards": [_card("a", parents=("a",)), _card("b")]}, ("a", "b"))
    with pytest.raises(ValueError, match="other input"):
        validate_cards({"cards": [_card("a", parents=("z",)), _card("b")]}, ("a", "b"))
    bad = _card("b")
    bad["role"] = "压力"
    with pytest.raises(ValueError, match="English"):
        validate_cards({"cards": [_card("a"), bad]}, ("a", "b"))
    bad = _card("b")
    bad["source"] = "Unlike the previous column, it is a gauge."
    with pytest.raises(ValueError, match="column order"):
        validate_cards({"cards": [_card("a"), bad]}, ("a", "b"))


def test_card_quotes_must_come_from_the_materials() -> None:
    cited = _card("b")
    cited["rationale"] = {"basis": "task_document", "evidence": "“Pressure  rises during filling。”"}
    sources = "Background: pressure rises during filling. Other text."
    assert validate_cards({"cards": [_card("a"), cited]}, ("a", "b"), sources=sources)
    cited["rationale"]["evidence"] = "Pressure falls during filling."
    with pytest.raises(ValueError, match="not in the materials"):
        validate_cards({"cards": [_card("a"), cited]}, ("a", "b"), sources=sources)


def _materials(tmp_path: Path, document: str = "# Process\n\nPressure rises during filling.\n"):
    spec = {
        "dataset": "toy",
        "target": "y",
        "horizon": "now",
        "online_boundary": "inputs only",
        "process_background": "A toy process.",
        "split_description": "SPLIT",
        "variables": [{"name": "a"}, {"name": "b"}],
    }
    (tmp_path / "task_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "PROCESS.md").write_text(document, encoding="utf-8")
    return load_materials(tmp_path, protocol_terms=("process_13",))


@pytest.mark.parametrize(
    "leak", ["测试集包含高负荷工况。", "Batch far_fault_091 drifts.", "过程 process_13 较长。"]
)
def test_process_document_rejects_split_information(tmp_path: Path, leak: str) -> None:
    with pytest.raises(ValueError, match="split or evaluation"):
        _materials(tmp_path, f"# Process\n\n{leak}\n")


def _reply(text: str, finish: str | None = "stop") -> LLMReply:
    return LLMReply(text, "m-0101", finish, {"prompt_tokens": 10, "completion_tokens": 5}, 0.5)


def _cards_json(suffix: str) -> str:
    return json.dumps({"cards": [_card("a", suffix), _card("b", suffix)]})


def test_pipeline_drafts_three_times_reviews_once_and_records_failures(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    config = LLMConfig("m", "http://x", "KEY")
    replies = iter(
        [
            _reply(_cards_json(" one")),
            _reply("", "length"),  # 截断：记录并重试
            _reply(_cards_json(" two")),
            _reply("not json"),
            _reply("still not json"),  # 第三次归纳两次尝试均失败，不进入复核
            _reply(_cards_json(" final")),
        ]
    )
    prompts = []

    def caller(prompt, _config):
        prompts.append(prompt)
        return next(replies)

    cards, report, files = generate_cards(materials, config, caller=caller)
    assert [card["source"] for card in cards] == ["The source of a final.", "The source of b final."]
    assert report["pipeline_version"] == PIPELINE_VERSION and report["status"] == "accepted"
    assert [run["status"] for run in report["extraction_runs"]] == ["accepted", "accepted", "failed"]
    assert report["extraction_runs"][1]["attempts"][0]["error"].endswith("finish_reason=length")
    assert report["review"]["draft_runs"] == [1, 2] and report["calls"] == 6
    assert "of a one" in prompts[-1] and "of a two" in prompts[-1]
    assert set(files) == {"prompt.txt", "review_prompt.txt"}


def test_saved_cards_report_stale_materials(tmp_path: Path) -> None:
    materials = _materials(tmp_path)
    config = LLMConfig("m", "http://x", "KEY")
    cards, report, _ = generate_cards(materials, config, caller=lambda *_: _reply(_cards_json("")))
    path = tmp_path / "out" / "semantic_cards.json"
    save_cards(path, materials=materials, cards=cards, llm_config=config, report=report)
    payload = load_cards(path, materials)
    assert not payload["edited_after_generation"] and payload["llm"]["response_model"] == "m-0101"
    assert "api_key_env" not in payload["llm"] and "base_url" not in payload["llm"]
    (tmp_path / "PROCESS.md").write_text("# Process\n\nPressure falls during discharge.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        load_cards(path, load_materials(tmp_path))


@pytest.mark.parametrize("dataset", DATASETS)
def test_task_specs_and_materials_match_dataset_variables(dataset: str) -> None:
    config = dataset_config(dataset)
    names = read_feature_names(config)
    materials = dataset_materials(config)  # PROCESS.md 通过划分与场景信息检查
    task = materials.task
    assert task.feature_names == names
    prompt = build_prompt(materials)
    assert all(f"- name: {name}\n" in prompt for name in names)
    assert materials.document.strip() in prompt
    assert not task.split_description and not task.scenario_descriptions


def test_revision_waits_out_connection_failures(tmp_path: Path, monkeypatch) -> None:
    import httpx

    from seer import llm, revision

    monkeypatch.setattr(llm, "OUTAGE", (0.0, 0.0, 1.0))
    materials = _materials(tmp_path)
    cards = {name: dict.fromkeys(FIELDS, "x") for name in materials.task.feature_names}
    reply = json.dumps({"candidates": [{"revisions": {"a": {"source": "now estimated"}}, "rationale": "r"}]})
    outcomes = iter([httpx.RemoteProtocolError("peer closed")] * 6 + ["not json"] * 3 + [reply])

    def caller(_prompt, _config):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return _reply(outcome)

    config = LLMConfig("m", "http://x", "KEY")
    record = revision.propose(materials, cards, "summary", 2, config, tmp_path, caller=caller)
    assert [c["revisions"] for c in record["candidates"]] == [{"a": {"source": "now estimated"}}]
    assert len(record["failed_attempts"]) == 3


def test_parallel_proposals_for_one_prompt_call_the_llm_once(tmp_path: Path, monkeypatch) -> None:
    """并行进程（此处用线程代替）遇到同一提示词时只调用一次，其余读取其缓存；不留下临时文件。"""

    import threading
    import time

    from seer import revision

    monkeypatch.setattr(revision, "LOCKS", tmp_path / "locks")
    materials = _materials(tmp_path)
    cards = {name: dict.fromkeys(FIELDS, "x") for name in materials.task.feature_names}
    reply = json.dumps({"candidates": [{"revisions": {"a": {"source": "now estimated"}}, "rationale": "r"}]})
    calls = []

    def caller(_prompt, _config):
        calls.append(1)
        time.sleep(0.2)
        return _reply(reply)

    config = LLMConfig("m", "http://x", "KEY")
    out = tmp_path / "proposals"
    records = []

    def work() -> None:
        records.append(revision.propose(materials, cards, "s", 2, config, out, caller=caller))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1 and len(records) == 4 and all(r == records[0] for r in records)
    assert [p.suffix for p in out.iterdir()] == [".json"]
