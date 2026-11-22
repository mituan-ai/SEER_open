from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seer.config import resolve_run
from seer.io import read_csv, read_json
from seer.training.runner import prepare, run, run_status, step_std


@pytest.mark.parametrize("method", ["base", "seer"])
def test_run_writes_outputs_and_resumes(method: str, toy_config_dir: Path, tmp_path: Path) -> None:
    spec = resolve_run("toy", "gru", method, 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    assert run_status(spec) == "not_started"
    run_dir = run(spec, log=lambda _: None)
    for name in (
        "config.resolved.yaml",
        "history.csv",
        "best.pt",
        "predictions.csv",
        "metrics.csv",
        "summary.json",
    ):
        assert (run_dir / name).exists(), name
    summary = read_json(run_dir / "summary.json")
    assert summary["config_sha256"] == spec.config_sha256 and summary["epochs_run"] == 2
    groups = {(row["group_type"], row["group"]) for row in read_csv(run_dir / "metrics.csv")}
    assert ("overall", "test") in groups and ("domain", "far") in groups
    assert run_status(spec) == "completed"

    messages: list[str] = []
    run(spec, log=messages.append)
    assert messages and messages[0].startswith("[skip]")

    spec.config["training"]["learning_rate"] = 0.01
    assert run_status(spec) == "stale"
    with pytest.raises(RuntimeError, match="another config"):
        run(spec, log=lambda _: None)


def test_reported_best_epoch_is_the_saved_checkpoint(toy_config_dir: Path, tmp_path: Path) -> None:
    import torch

    from seer.training.evaluation import regression_metrics
    from seer.training.runner import Stream, predict, prepare

    spec = resolve_run(
        "toy",
        "gru",
        "seer",
        2024,
        {"training.max_epochs": 6},
        config_dir=toy_config_dir,
        output_root=tmp_path / "runs",
    )
    run_dir = run(spec, log=lambda _: None)
    summary = read_json(run_dir / "summary.json")
    p = prepare(spec)
    p.model.load_state_dict(torch.load(run_dir / "best.pt"))
    stream = Stream.build(p.dataset, p.normalization, p.device)
    ends = p.dataset.ends(p.split.validation, labelled=True)
    target, prediction = predict(p.model, stream, ends, p.normalization, 16)
    assert abs(regression_metrics(target, prediction)["RMSE"] - summary["best_val_RMSE"]) < 1e-9


def test_prior_replaces_window_next_step_and_fields_consistently(
    toy_config_dir: Path, tmp_path: Path
) -> None:
    import numpy as np
    import torch

    from seer.methods.prior import apply_stream
    from seer.training.runner import Prior, Stream, prepare

    spec = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    p = prepare(spec)
    stream = Stream.build(p.dataset, p.normalization, p.device)
    prior = Prior(p.prior, 1.0, 0, p.device)
    prior.params["noise"].zero_()
    ends = torch.as_tensor(p.dataset.ends(p.split.train, has_next=True)[:6])
    x = stream.windows(ends).clone()
    nxt, _ = stream.following(ends)
    nxt = nxt.clone()
    fields = p.model.fields.expand(len(ends), *p.model.fields.shape).clone()
    draw_state = prior.generator.get_state()
    prior(stream, ends, x, nxt, fields)
    prior.generator.set_state(draw_state)
    torch.rand(len(ends), generator=prior.generator)
    draw = torch.randint(len(prior.params["target"]), (len(ends),), generator=prior.generator)
    phase = torch.randint(0, 1 << 20, (len(ends),), generator=prior.generator)
    values = stream.values.numpy().astype(np.float64)
    for row, end in enumerate(ends.tolist()):
        n = int(draw[row])
        start = int(stream.starts[end])
        segment = values[start : int(stream.stops[end])]
        transforms = p.prior.transforms.take(np.array([n]))
        transforms = type(transforms)(**{**transforms.__dict__, "noise": np.zeros(1)})
        expected = apply_stream(segment, transforms, 0, phase=int(phase[row]) - start)
        j = int(transforms.target[0])
        local = end - start
        np.testing.assert_allclose(x[row, :, j].numpy(), expected[local - 7 : local + 1, j], atol=1e-5)
        np.testing.assert_allclose(float(nxt[row, j]), expected[local + 1, j], atol=1e-5)
        assert not torch.equal(fields[row, j], p.model.fields[j])


def test_efficiency_is_measured_separately_and_tracks_the_run(toy_config_dir: Path, tmp_path: Path) -> None:
    from seer.training.runner import efficiency_status, measure

    spec = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    with pytest.raises(RuntimeError, match="not a completed run"):
        measure(spec, log=lambda _: None)
    run(spec, log=lambda _: None)
    assert efficiency_status(spec) == "pending"
    measure(spec, log=lambda _: None)
    record = read_json(spec.run_dir / "efficiency.json")
    for key in ("inference_ms_per_sample", "latency_ms_median", "train_ms_per_sample", "parameters"):
        assert record[key] > 0
    assert efficiency_status(spec) == "measured"
    changed = resolve_run(
        "toy",
        "gru",
        "seer",
        2024,
        {"efficiency.latency_samples": 5},
        config_dir=toy_config_dir,
        output_root=tmp_path / "runs",
    )
    assert changed.run_dir == spec.run_dir and changed.config_sha256 == spec.config_sha256
    assert efficiency_status(changed) == "pending"


def test_a_run_directory_is_locked(toy_config_dir: Path, tmp_path: Path) -> None:
    from seer.training.runner import _run_lock

    spec = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    with _run_lock(spec.run_dir), pytest.raises(RuntimeError, match="in use"):
        run(spec, log=lambda _: None)


def test_runs_from_other_code_are_outdated(toy_config_dir: Path, tmp_path: Path, monkeypatch) -> None:
    import seer.training.runner as runner

    spec = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    run(spec, log=lambda _: None)
    monkeypatch.setattr(runner, "implementation_sha256", lambda: "other")
    assert run_status(spec) == "outdated"
    with pytest.raises(RuntimeError, match="other code"):
        run(spec, log=lambda _: None)


def test_evaluation_random_state_is_fixed_and_isolated() -> None:
    import torch

    from seer.training.runner import evaluation_rng

    torch.manual_seed(1)
    expected_next = torch.rand(1)
    torch.manual_seed(1)
    with evaluation_rng(torch.device("cpu")):
        first = torch.rand(2)
    with evaluation_rng(torch.device("cpu")):
        second = torch.rand(2)
    assert torch.equal(first, second) and torch.equal(torch.rand(1), expected_next)


def test_runs_without_their_semantic_artifact_are_blocked(toy_config_dir: Path, tmp_path: Path) -> None:
    spec = resolve_run(
        "toy",
        "gru",
        "seer",
        2024,
        {"method.embedding": "unbuilt-encoder"},
        config_dir=toy_config_dir,
        output_root=tmp_path / "runs",
    )
    assert run_status(spec) == "blocked"
    with pytest.raises(RuntimeError, match="not built yet"):
        run(spec, log=lambda _: None)
    no_field = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path)
    assert run_status(no_field) == "not_started"


def test_epoch_log_matches_the_scheduler_progress_pattern(
    toy_config_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
    module_spec = importlib.util.spec_from_file_location("run_study", path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, "run_study", module)  # dataclass 需要按模块名查找
    module_spec.loader.exec_module(module)
    spec = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path / "runs")
    messages: list[str] = []
    run(spec, log=messages.append)
    matches = [module.PROGRESS.search(message) for message in messages]
    assert [match[1] for match in matches if match] == ["1", "2"]


def test_observation_scale_floor_comes_from_training_steps(toy_config_dir: Path, tmp_path: Path) -> None:
    spec = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir, output_root=tmp_path)
    p = prepare(spec)
    observed = list(p.model.observation.observed.cpu())
    std = step_std(p.dataset, p.split.train, p.normalization)[observed]
    expected = float(spec.config["method"]["scale_floor"]) * std
    np.testing.assert_allclose(p.model.observation.floor.cpu().numpy(), expected, rtol=1e-6)


def test_dataset_method_overrides_only_existing_fields(toy_config_dir: Path, tmp_path: Path) -> None:
    seer = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir, output_root=tmp_path)
    base = resolve_run("toy", "gru", "base", 2024, config_dir=toy_config_dir, output_root=tmp_path)
    assert seer.config["method"]["observation_weight"] == 0.5
    assert "observation_weight" not in base.config["method"] and "method" not in seer.config["dataset"]
