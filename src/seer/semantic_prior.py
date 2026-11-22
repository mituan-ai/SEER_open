"""合成语义先验的描述池：变换参数与各措辞的字段嵌入（训练读取）。

``pool.npz``：
    target, source (N,)       受影响变量与来源变量的列号
    kernel (N, lmax + 1)      时滞核 h
    gain, bias, noise (N,)    标准化尺度上的 a、b 与噪声标准差
    hold (N,)                 采样保持周期（步，1 为不保持）
    affected (N, F)           LLM 改写的字段
    fields (N, P, F, k)       受影响变量在各措辞下的四个字段嵌入（未改写字段为原嵌入）
``metadata.json`` 记录变量名、描述文本哈希与编码器设置。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seer.io import read_json, sha256_json, write_json
from seer.methods.prior import Transforms

POOL_TYPE = "seer_semantic_prior"


@dataclass(frozen=True)
class PriorPool:
    variable_names: tuple[str, ...]
    transforms: Transforms
    affected: np.ndarray
    fields: np.ndarray
    metadata: dict[str, Any]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self.transforms, name) for name in Transforms.__dataclass_fields__}
        np.savez(directory / "pool.npz", affected=self.affected, fields=self.fields, **arrays)
        write_json(
            directory / "metadata.json",
            {
                **self.metadata,
                "artifact_type": POOL_TYPE,
                "variable_names": list(self.variable_names),
                "arrays_sha256": sha256_json(self.fields),
            },
        )


def load_pool(directory: Path, feature_names: tuple[str, ...]) -> PriorPool:
    metadata = read_json(directory / "metadata.json")
    if metadata.get("artifact_type") != POOL_TYPE or tuple(metadata["variable_names"]) != feature_names:
        raise ValueError(f"{directory} is not a prior pool for these variables")
    with np.load(directory / "pool.npz") as arrays:
        transforms = Transforms(**{name: arrays[name] for name in Transforms.__dataclass_fields__})
        pool = PriorPool(feature_names, transforms, arrays["affected"], arrays["fields"], metadata)
    if sha256_json(pool.fields) != metadata["arrays_sha256"] or not np.isfinite(pool.fields).all():
        raise ValueError(f"prior pool hash mismatch in {directory}")
    return pool
