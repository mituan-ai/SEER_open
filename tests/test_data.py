from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import FEATURES, WINDOW, write_synthetic_dataset

from seer.config import dataset_config
from seer.data import Normalization, build_split, dataset_fingerprint, load_dataset


def toy_config(dataset: Path) -> dict:
    return {
        "name": "toy",
        "path": str(dataset),
        "file": "stream.npz",
        "split": {
            "train": {"column": "domain_bucket", "values": ["train"]},
            "validation": {"column": "scenario_name", "values": ["train_1"]},
            "test": {"column": "domain_bucket", "values": ["same_family", "far"]},
        },
        "eval_groups": {"domain": "domain_bucket"},
    }


def test_stream_layout_windows_split_and_fingerprint(tmp_path: Path) -> None:
    config = toy_config(write_synthetic_dataset(tmp_path))
    dataset = load_dataset(config)
    split = build_split(dataset, config)
    assert set(split.groups["domain"]) == {"same_family", "far"}
    assert not set(split.train) & set(split.validation) and not set(split.test) & set(split.train)
    ends = dataset.ends(split.train, labelled=True)
    windows = dataset.windows(ends)
    assert windows.shape == (ends.size, WINDOW, len(FEATURES))
    assert np.allclose(
        dataset.targets[ends, 0], windows[:, :, 0].mean(axis=1) + 0.5 * windows[:, -1, 1], atol=1e-5
    )
    assert (dataset.segment_index(ends) == dataset.segment_index(ends - WINDOW + 1)).all()  # 窗口不跨段
    with_next = dataset.ends(split.train, has_next=True)
    assert dataset.has_next(with_next).all() and not dataset.has_next(dataset.ends(split.train)).all()
    assert dataset.ends(split.train).size > ends.size  # 无标签时刻也是窗口末端
    assert list(dataset_fingerprint(config).values())[0] is not None


def test_constant_training_variables_get_a_finite_scale() -> None:
    x = np.zeros((40, 3), dtype=np.float32)
    x[:, 0] = np.random.default_rng(0).normal(size=40)
    x[:, 1] = 100.0  # 全部训练批次中的搅拌转速
    norm = Normalization.fit(x, np.zeros((10, 1), dtype=np.float32))
    changed = x.copy()
    changed[:, 1] = 80.0
    assert np.allclose(norm.inputs(x)[:, 1:], 0.0)
    assert np.allclose(norm.inputs(changed)[:, 1], -0.2)
    assert np.isclose(norm.input_std[2], 1.0)


def test_normalization_uses_given_statistics() -> None:
    x = np.random.default_rng(0).normal(3.0, 2.0, size=(200, 2)).astype(np.float32)
    y = np.random.default_rng(1).normal(size=(50, 1)).astype(np.float32)
    norm = Normalization.fit(x, y)
    assert np.allclose(norm.inputs(x).mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(norm.inverse_targets(norm.targets(y)), y, atol=1e-5)


@pytest.mark.dataset
@pytest.mark.parametrize(
    "name, counts",
    [
        ("thickener_dewatering", (3915, 783, 2244)),
        ("indpensim", (451, 90, 339)),
        ("ladle_preheating", (11735, 2500, 5605)),
    ],
)
def test_released_streams_split_into_the_documented_windows(name: str, counts: tuple[int, ...]) -> None:
    """随仓库的流数据包按配置划分后的有标签窗口数；划分的段互不重叠，输入全部有限。"""

    config = dataset_config(name)
    dataset = load_dataset(config)
    split = build_split(dataset, config)
    assert (
        tuple(dataset.ends(part, labelled=True).size for part in (split.train, split.validation, split.test))
        == counts
    )
    assert not {*split.train} & {*split.validation} and not {*split.test} & {*split.train, *split.validation}
    assert np.isfinite(dataset.values).all()
