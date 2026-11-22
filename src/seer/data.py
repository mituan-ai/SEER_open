"""逐段流数据集、声明式划分、评测组与归一化。

每个数据集目录提供一个流数据包 ``stream.npz``：

    values         (T, d) float32，全部段按顺序拼接，每行一个时刻
    targets        (T, p) float32，无标签处为 NaN
    feature_names  (d,) 变量名；target_names (p,) 标签名；window_size 历史窗口长度
    segment_start, segment_stop  (S,) 各段的全局行号 [start, stop)
    meta_<列名>    (S,) 逐段元数据（划分与评测组按这些列声明）

窗口在读取时现场切出：以全局行号 t 为窗口末端，窗口为同段内的 [t - L + 1, t]；
下一步观测 x_{t+1} 只在 t + 1 仍属同段时存在。流数据包的 SHA-256 纳入运行身份。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seer.io import resolve_path, sha256_file


@dataclass(frozen=True)
class Segment:
    start: int  # 全局行号 [start, stop)
    stop: int
    meta: dict[str, Any]


@dataclass(frozen=True)
class StreamDataset:
    name: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    window_size: int
    values: np.ndarray  # (T, d)，全部段按顺序拼接
    targets: np.ndarray  # (T, p)，无标签处为 NaN
    segments: tuple[Segment, ...]

    @property
    def labelled(self) -> np.ndarray:
        return np.isfinite(self.targets).all(axis=1)

    def segment_column(self, name: str) -> np.ndarray:
        if name not in self.segments[0].meta:
            raise KeyError(f"dataset {self.name} has no segment column {name!r}")
        return np.asarray([segment.meta[name] for segment in self.segments], dtype=object)

    def ends(self, segment_ids: np.ndarray, *, labelled: bool = False, has_next: bool = False) -> np.ndarray:
        """所选段内可作窗口末端的全局行号。"""

        parts = []
        for index in np.asarray(segment_ids, dtype=np.int64):
            segment = self.segments[int(index)]
            parts.append(np.arange(segment.start + self.window_size - 1, segment.stop - int(has_next)))
        ends = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return ends[self.labelled[ends]] if labelled else ends

    def has_next(self, ends: np.ndarray) -> np.ndarray:
        stops = np.asarray([segment.stop for segment in self.segments])
        return ends + 1 < stops[self.segment_index(ends)]

    def segment_index(self, rows: np.ndarray) -> np.ndarray:
        starts = np.asarray([segment.start for segment in self.segments])
        return np.searchsorted(starts, rows, side="right") - 1

    def windows(self, ends: np.ndarray) -> np.ndarray:
        offsets = np.arange(-self.window_size + 1, 1)
        return self.values[np.asarray(ends)[:, None] + offsets]

    def meta(self, rows: np.ndarray) -> list[dict[str, Any]]:
        return [self.segments[int(index)].meta for index in self.segment_index(rows)]


@dataclass(frozen=True)
class Split:
    """各划分为段号；样本为段内窗口末端。"""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    groups: dict[str, dict[str, np.ndarray]]  # 分组类型 -> 分组名称 -> 段号

    def summary(self, dataset: StreamDataset) -> dict[str, Any]:
        def count(segments: np.ndarray) -> dict[str, int]:
            return {
                "segments": int(segments.size),
                "labelled_windows": int(dataset.ends(segments, labelled=True).size),
                "windows": int(dataset.ends(segments).size),
            }

        return {
            "train": count(self.train),
            "validation": count(self.validation),
            "test": count(self.test),
            "groups": {
                kind: {name: count(ids) for name, ids in groups.items()}
                for kind, groups in self.groups.items()
            },
        }


@dataclass(frozen=True)
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, targets: np.ndarray) -> Normalization:
        """``values`` 为训练段的逐步记录，``targets`` 为训练标签（不含 NaN）。"""

        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        mean, std = x.mean(axis=0), x.std(axis=0)
        # 训练时恒定的变量用 |mean| 缩放，均值为零时用 1；
        # 因此测试时的变化表示为相对偏差，避免数值溢出。
        constant = std <= 1e-8 * np.maximum(np.abs(mean), 1.0)
        std = np.where(constant, np.where(np.abs(mean) > 0, np.abs(mean), 1.0), std)
        return cls(
            input_mean=mean.astype(np.float32),
            input_std=std.astype(np.float32),
            target_mean=y.mean(axis=0).astype(np.float32),
            target_std=np.maximum(y.std(axis=0), 1e-12).astype(np.float32),
        )

    def inputs(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.input_mean) / self.input_std).astype(np.float32)

    def targets(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.target_std + self.target_mean

    def as_dict(self, feature_names: tuple[str, ...], target_names: tuple[str, ...]) -> dict[str, Any]:
        return {
            "input_mean": dict(zip(feature_names, self.input_mean.tolist(), strict=True)),
            "input_std": dict(zip(feature_names, self.input_std.tolist(), strict=True)),
            "target_mean": dict(zip(target_names, self.target_mean.tolist(), strict=True)),
            "target_std": dict(zip(target_names, self.target_std.tolist(), strict=True)),
        }


def bundle_path(config: dict[str, Any]) -> Path:
    return resolve_path(config["path"]) / config.get("file", "stream.npz")


_FINGERPRINTS: dict[tuple, str] = {}


def dataset_fingerprint(config: dict[str, Any]) -> dict[str, str | None]:
    """流数据包的 SHA-256；按文件的 (size, mtime) 在进程内缓存。"""

    path = bundle_path(config)
    if not path.exists():
        return {path.name: None}
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _FINGERPRINTS:
        _FINGERPRINTS[key] = sha256_file(path)
    return {path.name: _FINGERPRINTS[key]}


def read_feature_names(config: dict[str, Any]) -> tuple[str, ...]:
    with np.load(bundle_path(config)) as bundle:
        return tuple(str(name) for name in bundle["feature_names"])


def load_dataset(config: dict[str, Any]) -> StreamDataset:
    with np.load(bundle_path(config)) as bundle:
        columns = [key for key in bundle.files if key.startswith("meta_")]
        metas = {key[len("meta_") :]: bundle[key].tolist() for key in columns}
        starts, stops = bundle["segment_start"].tolist(), bundle["segment_stop"].tolist()
        segments = tuple(
            Segment(int(start), int(stop), {name: values[i] for name, values in metas.items()})
            for i, (start, stop) in enumerate(zip(starts, stops, strict=True))
        )
        dataset = StreamDataset(
            str(config["name"]),
            tuple(str(name) for name in bundle["feature_names"]),
            tuple(str(name) for name in bundle["target_names"]),
            int(bundle["window_size"]),
            bundle["values"].astype(np.float32),
            bundle["targets"].astype(np.float32),
            segments,
        )
    if not np.isfinite(dataset.values).all():
        raise ValueError(f"dataset {config['name']} contains non-finite inputs")
    return dataset


def select(dataset: StreamDataset, rule: dict[str, Any], pool: np.ndarray | None = None) -> np.ndarray:
    values = dataset.segment_column(str(rule["column"]))
    indices = np.flatnonzero(np.isin(values, np.asarray(rule["values"], dtype=object)))
    if pool is not None:
        indices = np.intersect1d(indices, pool)
    if indices.size == 0:
        raise ValueError(f"no segments match {rule} in dataset {dataset.name}")
    return indices.astype(np.int64)


def build_split(dataset: StreamDataset, config: dict[str, Any]) -> Split:
    """按声明的段元数据规则划分；验证集从训练数据池中留出。"""

    split = config["split"]
    pool = select(dataset, split["train"])
    validation = select(dataset, split["validation"], pool)
    train = np.setdiff1d(pool, validation)
    test = select(dataset, split["test"])
    if np.intersect1d(test, pool).size:
        raise ValueError(f"test segments overlap the training pool in dataset {dataset.name}")
    groups: dict[str, dict[str, np.ndarray]] = {}
    for kind, column in config.get("eval_groups", {}).items():
        values = dataset.segment_column(str(column))[test]
        groups[kind] = {str(value): test[values == value] for value in sorted(set(values), key=str)}
    return Split(train=train, validation=validation, test=test, groups=groups)


def training_normalization(dataset: StreamDataset, split: Split) -> Normalization:
    """训练段全部逐步记录的输入统计与有标签训练窗口的目标统计。"""

    rows = np.concatenate(
        [np.arange(dataset.segments[i].start, dataset.segments[i].stop) for i in split.train]
    )
    return Normalization.fit(dataset.values[rows], dataset.targets[dataset.ends(split.train, labelled=True)])
