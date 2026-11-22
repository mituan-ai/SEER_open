from __future__ import annotations

import numpy as np
import pytest

from seer.changes import build_streams, evaluate
from seer.data import Segment, StreamDataset


def test_record_operators_are_causal_and_in_minutes() -> None:
    x = np.arange(10.0)
    columns = {"a": x, "b": 2 * x}
    np.testing.assert_array_equal(evaluate("lag(a, 4)", columns, 2.0), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(evaluate("mean(a, 6)", columns, 2.0)[3:], x[3:] - 1)
    np.testing.assert_array_equal(evaluate("hold(a, 6)", columns, 2.0), [0, 0, 0, 3, 3, 3, 6, 6, 6, 9])
    np.testing.assert_allclose(
        evaluate("0.5 * b + sqrt(maximum(a, 4))", columns, 1.0), x + np.sqrt(np.maximum(x, 4))
    )
    y = evaluate("ewm(a, 3)", columns, 1.0)
    assert y[0] == 0 and np.all(np.diff(y) > 0) and np.all(y <= x)
    for bad in ("__import__('os')", "a.real", "open(a)", "c + 1", "[a][0]"):
        with pytest.raises(ValueError):
            evaluate(bad, columns, 1.0)


def test_streams_rewrite_only_after_onset_and_keep_targets() -> None:
    values = np.random.default_rng(0).normal(size=(40, 2)).astype(np.float32)
    targets = values[:, :1].copy()
    segments = tuple(
        Segment(s, s + 10, {"name": f"s{s}", "bucket": "x" if s < 20 else "y"}) for s in range(0, 40, 10)
    )
    data = StreamDataset("toy", ("a", "b"), ("y",), 3, values, targets, segments)
    changes = {
        "onsets": [0.0, 0.5],
        "changes": [
            {"id": "c", "variable": "a", "record": "lag(b, 1)", "cards": {"a": {"source": "new"}}},
            {"id": "d", "variable": "b", "record": "lag(b, 1)", "control": "out of scope"},
        ],
        "drifts": [
            {"id": "bias", "family": "in", "variable": "b", "bias_std": 1.0},
            {"id": "ramp", "family": "out", "variable": "b", "ramp_gain": 2.0},
            {"id": "load", "family": "out", "scenarios": {"column": "bucket", "values": ["y"]}},
        ],
    }
    streams = {
        s.name: s for s in build_streams(data, np.array([0, 1]), changes, 1.0, np.zeros(2), np.full(2, 2.0))
    }
    assert set(streams) == {
        "regular",
        "c@onset00",
        "c@onset50",
        "d@onset00",
        "d@onset50",
        "bias@onset00",
        "bias@onset50",
        "ramp@onset00",
        "ramp@onset50",
        "load",
    }
    assert [streams[n].kind for n in ("regular", "c@onset00", "d@onset00", "bias@onset00", "load")] == [
        "regular",
        "change",
        "control",
        "drift_in",
        "drift_out",
    ]
    late = streams["c@onset50"]  # 测试流长 20，起点在第二段开头，之后一直保持
    np.testing.assert_array_equal(late.onsets, [10, 10])
    np.testing.assert_array_equal(late.dataset.values[:10], values[:10])
    np.testing.assert_allclose(late.dataset.values[11:20, 0], values[10:19, 1])
    np.testing.assert_array_equal(late.dataset.values[20:], values[20:])  # 非测试段不改写
    np.testing.assert_array_equal(late.dataset.targets, targets)
    np.testing.assert_allclose(
        streams["bias@onset00"].dataset.values[:20, 1], values[:20, 1] + 2.0, rtol=1e-6
    )
    ramp = streams["ramp@onset50"].dataset.values[10:20, 1] / values[10:20, 1]
    np.testing.assert_allclose(ramp, 1 + np.arange(10) / 9, rtol=1e-5)
    np.testing.assert_array_equal(streams["load"].segments, [2, 3])


def test_multi_variable_change_reads_records_before_rewriting() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    data = StreamDataset("toy", ("a", "b"), ("y",), 3, values, values[:, :1], (Segment(0, 10, {}),))
    changes = {"onsets": [0.0], "changes": [{"id": "swap", "records": {"a": "b", "b": "a"}}]}
    swapped = build_streams(data, np.array([0]), changes, 1.0, np.zeros(2), np.ones(2))[1]
    assert swapped.variables == ("a", "b")
    np.testing.assert_array_equal(swapped.dataset.values, values[:, ::-1])


def test_prior_sources_vary_on_training_rows() -> None:
    """来源只取训练段上非恒定的变量：只在其余数据上变化的变量会给出无界的范围匹配增益。"""

    from seer.priors import sample_transforms

    rng = np.random.default_rng(0)
    values = rng.normal(size=(90, 4))
    values[:60, 3] = 1.0  # 训练段恒定，之后变化
    prior = {
        "max_lag": 2,
        "smoothing_steps": [1, 2],
        "spread_steps": [2, 2],
        "hold_steps": [2],
        "hold_probability": 0.0,
        "self_source_probability": 0.0,
        "noise_fraction": [0.0, 0.0],
        "gain_perturbation": 0.1,
        "min_departure": 0.0,
    }
    transforms, _, _ = sample_transforms(values, [(0, 30), (30, 60)], (0, 1), prior, 40, 0)
    assert 3 not in transforms.source
    assert np.abs(transforms.gain).max() < 10
