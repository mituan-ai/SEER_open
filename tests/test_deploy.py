from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from seer.changes import EvalStream
from seer.config import resolve_run
from seer.training.online import Reference, Runner, Settings
from seer.training.runner import prepare

SETTINGS = Settings(
    budget=2,
    risk=0.05,
    threshold=0.5,
    alpha=2.0,
    beta=1.0,
    warmup=5,
    calibration_floor=0.1,
    exact_floor=1e-3,
    cusum_drift=0.5,
    cusum_cap=9.0,
    cusum_threshold=4.0,
    batch_size=16,
)


def _setup(toy_config_dir: Path, tmp_path: Path):
    spec = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    p = prepare(spec)
    p.model.eval()
    d, lags = len(p.dataset.feature_names), p.model.reader.lags
    reference = Reference(np.zeros(len(p.model.observation.observed)), np.zeros((d, d, lags)))
    box = np.array([np.full(d, -np.inf), np.full(d, np.inf)])
    runner = Runner(p, {n: {"source": n} for n in p.dataset.feature_names}, box, reference, SETTINGS, 1.0)
    test = p.split.test
    onsets = np.array([p.dataset.segments[i].start for i in test])
    stream = EvalStream("regular", "regular", p.dataset, test, onsets, (), {})
    return p, runner, stream


def _perturbed(fields: torch.Tensor):
    def propose(context):
        assert context["summary"] and context["gradient"].shape == fields.shape[:2]
        noisy = fields + 0.5 * torch.randn(fields.shape, generator=torch.Generator().manual_seed(0))
        return [
            (fields.clone(), context["base"].cards, {"id": "same"}),
            (noisy, context["base"].cards, {"id": "noisy"}),
        ]

    return propose


def test_online_revision_is_causal_and_records_events(toy_config_dir: Path, tmp_path: Path) -> None:
    p, runner, stream = _setup(toy_config_dir, tmp_path)
    result = runner.run(stream, _perturbed(p.model.fields))
    events = result["events"]
    assert events and events[0]["trigger"] == "start" and events[0]["pre_event_samples"] == SETTINGS.warmup
    assert all(0.0 <= t["w"] <= 1.0 for e in events for t in e["trace"])
    labelled = np.isfinite(result["prediction"]).all(axis=1)
    assert labelled.sum() == np.isfinite(p.dataset.targets[result["ends"]]).all(axis=1).sum()

    cut = int(np.flatnonzero(labelled)[len(np.flatnonzero(labelled)) // 2])
    values = p.dataset.values.copy()
    later = result["ends"][cut:]
    values[later] += 3.0  # 只改写 cut 及之后的记录
    changed = runner.run(
        replace(stream, dataset=replace(p.dataset, values=values)), _perturbed(p.model.fields)
    )
    np.testing.assert_allclose(changed["prediction"][:cut], result["prediction"][:cut], equal_nan=True)


def test_candidates_revising_only_unobserved_inputs_are_not_tested(
    toy_config_dir: Path, tmp_path: Path
) -> None:
    p, runner, stream = _setup(toy_config_dir, tmp_path)
    hidden = sorted(set(range(len(p.dataset.feature_names))) - set(p.model.observation.observed.tolist()))
    assert hidden, "toy 数据集需要一个未被观测的输入"

    def propose(context):
        fields = context["base"].fields.clone()
        fields[hidden] += 1.0
        return [(fields, context["base"].cards, {"id": "hidden"})]

    events = runner.run(stream, propose)["events"]
    assert events and all(not e["candidates"] and e["untestable"] == [{"id": "hidden"}] for e in events)
    assert not any(e.get("approved") for e in events)
