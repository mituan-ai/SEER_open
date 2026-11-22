from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

FEATURES = ("feed", "pressure", "phase")
WINDOW = 8


def write_synthetic_dataset(root: Path, *, seed: int = 0, prefix: str = "") -> Path:
    """包含 train / same_family / far 段的小型逐步流数据集；每隔一步有标签。"""

    rng = np.random.default_rng(seed)
    dataset = root / "toy"
    dataset.mkdir(parents=True)
    values, targets, buckets, scenarios = [], [], [], []
    for bucket, scenario in [
        ("train", "train_0"),
        ("train", "train_1"),
        ("train", "train_2"),
        ("same_family", "same_family_0"),
        ("far", "far_0"),
    ]:
        block = rng.normal(size=(30, len(FEATURES)))
        for step in range(30):
            window = block[max(step - WINDOW + 1, 0) : step + 1]
            targets.append(window[:, 0].mean() + 0.5 * window[-1, 1] if step % 2 == 0 else np.nan)
        values.append(block)
        buckets.append(bucket)
        scenarios.append(prefix + scenario)
    starts = np.arange(len(values)) * 30
    np.savez(
        dataset / "stream.npz",
        values=np.concatenate(values).astype(np.float32),
        targets=np.asarray(targets, dtype=np.float32)[:, None],
        feature_names=np.asarray(FEATURES),
        target_names=np.asarray(["y"]),
        window_size=np.asarray(WINDOW),
        segment_start=starts,
        segment_stop=starts + 30,
        meta_domain_bucket=np.asarray(buckets),
        meta_scenario_name=np.asarray(scenarios),
    )
    spec = {
        "sampling_minutes": 1,
        "observed": ["feed", "pressure"],
        "reader_lags": 3,
        "prior": {
            "max_lag": 3,
            "smoothing_steps": [1, 2],
            "spread_steps": [2, 3],
            "hold_steps": [2, 3],
            "hold_probability": 0.25,
            "self_source_probability": 0.4,
            "noise_fraction": [0.0, 0.05],
            "gain_perturbation": 0.2,
            "min_departure": 0.1,
        },
        "event": {"warmup": 5, "cusum_drift": 0.5, "cusum_threshold": 4.0},
    }
    (dataset / "seer.json").write_text(json.dumps(spec), encoding="utf-8")
    return dataset


def write_toy_pool(directory: Path, values: np.ndarray, count: int = 12) -> None:
    """toy 数据集的描述池：真实抽样的变换与随机字段嵌入。"""

    from seer.priors import sample_transforms
    from seer.semantic_prior import PriorPool

    spec = json.loads((directory.parents[1] / "toy" / "seer.json").read_text(encoding="utf-8"))
    transforms, _, _ = sample_transforms(values, [(0, 30), (30, 60)], (0, 1), spec["prior"], count, 0)
    rng = np.random.default_rng(1)
    affected = np.zeros((count, 4), dtype=bool)
    affected[:, :2] = True
    PriorPool(FEATURES, transforms, affected, rng.normal(size=(count, 3, 4, 256)), {}).save(directory)


@pytest.fixture
def toy_config_dir(tmp_path: Path) -> Path:
    """只包含一个合成数据集的配置目录。"""

    from seer.semantic_artifact import build_random

    dataset = write_synthetic_dataset(tmp_path)
    source = "deepseek-flash__qwen3.7-text-embedding"
    build_random(FEATURES, dim=256, seed=0).save(tmp_path / "artifacts" / source)
    values = np.random.default_rng(2).normal(size=(60, len(FEATURES)))
    write_toy_pool(tmp_path / "priors" / source, values)
    config_dir = tmp_path / "configs"
    (config_dir / "datasets").mkdir(parents=True)
    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("backbones.yaml", "methods.yaml", "llm.yaml", "embedding.yaml"):
        (config_dir / name).write_text((root / name).read_text(encoding="utf-8"), encoding="utf-8")
    with (config_dir / "embedding.yaml").open("a", encoding="utf-8") as handle:  # 未构建产物的编码器
        handle.write("  unbuilt-encoder: {base_url: http://x, api_key_env: SEER_UNBUILT_API_KEY}\n")
    training = yaml.safe_load((root / "training.yaml").read_text(encoding="utf-8"))
    training["training"].update({"device": "cpu", "max_epochs": 2, "batch_size": 16})
    training["efficiency"].update(
        {"timing_warmup_loops": 0, "timing_repeat_loops": 1, "latency_samples": 3, "train_timing_steps": 2}
    )
    (config_dir / "training.yaml").write_text(yaml.safe_dump(training), encoding="utf-8")
    (config_dir / "datasets" / "toy.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy",
                "path": str(dataset),
                "file": "stream.npz",
                "split": {
                    "train": {"column": "domain_bucket", "values": ["train"]},
                    "validation": {"column": "scenario_name", "values": ["train_1"]},
                    "test": {"column": "domain_bucket", "values": ["same_family", "far"]},
                },
                "eval_groups": {"domain": "domain_bucket", "scenario": "scenario_name"},
                "artifacts": str(tmp_path / "artifacts"),
                "priors": str(tmp_path / "priors"),
                "method": {"observation_weight": 0.5},
            }
        ),
        encoding="utf-8",
    )
    return config_dir
