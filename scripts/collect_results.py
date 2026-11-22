"""把训练 study 各运行的计算效率整理为 outputs/<study>/efficiency.csv。

    uv run python scripts/collect_results.py configs/studies/smoke.yaml [--backbones ...]

每行一个 (dataset, backbone, 指标, method)，同一指标的各方法相邻；种子为列（``seed_<seed>``），数值保留五位
小数（计数为整数）。耗时、显存与轮数取自各运行的 efficiency.json 与 summary.json，只换算单位（每样本耗时 us、
训练时间 s），不取平均、不筛选；参数量按模块统计模型的可训练参数（不含语义嵌入等缓冲区）：
主干与目标头、语义读取层、观测头（只用于部署时的证据，不参与目标预测），三者之和为总数。目标精度见在线评测
（scripts/run_online.py）的 metrics.csv。

全部声明的运行必须以当前配置和代码完成并已测量效率；同一 (dataset, backbone, seed) 的各方法须使用相同的
数据内容、主干与训练配置（``pairing_sha256``）。``--allow-missing`` 改为写入 efficiency.partial.csv。
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from seer.config import load_study
from seer.io import ROOT, read_json, resolve_path, write_csv
from seer.training.runner import efficiency_status, prepare, run_status

# 参数分块：模型顶层模块 -> 报告的块；主干与目标头合为一块，Base 只有这一块。
BLOCKS = {
    "backbone": "backbone_head",
    "head": "backbone_head",
    "reader": "reader",
    "observation": "observation",
}

# 报告指标 -> (记录中的字段, 单位, 换算倍数)；计数类按整数写出。
# 训练墙钟时间受同卡并发影响，不作计时指标。
METRICS = {
    "parameters": ("parameters", "count", 1.0),
    "parameters_backbone_head": ("backbone_head", "count", 1.0),
    "parameters_reader": ("reader", "count", 1.0),
    "parameters_observation": ("observation", "count", 1.0),
    "inference_per_sample": ("inference_ms_per_sample", "us", 1e3),
    "latency_mean": ("latency_ms_mean", "ms", 1.0),
    "latency_median": ("latency_ms_median", "ms", 1.0),
    "latency_p95": ("latency_ms_p95", "ms", 1.0),
    "train_step_per_sample": ("train_ms_per_sample", "us", 1e3),
    "train_time": ("train_time_h", "s", 3600.0),
    "epochs_run": ("epochs_run", "epoch", 1.0),
    "best_epoch": ("best_epoch", "epoch", 1.0),
    "inference_peak_gpu_memory": ("inference_peak_gpu_memory_mb", "MiB", 1.0),
    "train_peak_gpu_memory": ("train_peak_gpu_memory_mb", "MiB", 1.0),
    "peak_rss": ("peak_rss_mb", "MiB", 1.0),
    "peak_rss_increase": ("peak_rss_increase_mb", "MiB", 1.0),
}
COUNTS = {"count", "epoch"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("study")
    parser.add_argument(
        "--allow-missing", action="store_true", help="write raw_results.partial.csv for finished runs only"
    )
    parser.add_argument("--backbones", nargs="+", help="只整理这些主干（默认 study 的全部主干）")
    args = parser.parse_args()

    study = load_study(resolve_path(args.study))
    backbones = args.backbones or study.raw["backbones"]
    runs = [spec for spec in study.runs if spec.backbone in backbones]
    missing = [
        spec.name
        for spec in runs
        if run_status(spec) != "completed" or efficiency_status(spec) != "measured"
    ]
    if missing and not args.allow_missing:
        raise SystemExit(
            f"{len(missing)} of {len(runs)} runs are not completed with the current config and code "
            f"or not measured, e.g. {missing[:3]}"
        )

    values: dict[tuple, dict[int, object]] = defaultdict(dict)
    pairing: dict[tuple, set[str]] = defaultdict(set)
    for spec in runs:
        if spec.name in missing:
            continue
        record = {
            **read_json(spec.run_dir / "summary.json"),
            **read_json(spec.run_dir / "efficiency.json"),
            **parameter_blocks(spec),
        }
        pairing[(spec.dataset, spec.backbone, spec.seed)].add(record["pairing_sha256"])
        for metric, (field, _, scale) in METRICS.items():
            cell = values[(spec.dataset, spec.backbone, metric, spec.method)]
            if spec.seed in cell:
                raise SystemExit(f"{spec.name}: one run per (dataset, backbone, method, seed) expected")
            cell[spec.seed] = None if record.get(field) is None else float(record[field]) * scale
    unpaired = [key for key, value in pairing.items() if len(value) > 1]
    if unpaired:
        raise SystemExit(f"paired runs differ outside the method config: {unpaired[:3]}")

    raw = study.raw
    rows = []
    for dataset, backbone, metric, method in sorted(
        values,
        key=lambda k: (
            raw["datasets"].index(k[0]),
            backbones.index(k[1]),
            list(METRICS).index(k[2]),
            raw["methods"].index(k[3]),
        ),
    ):
        unit = METRICS[metric][1]
        row = {"dataset": dataset, "backbone": backbone, "metric": metric, "unit": unit, "method": method}
        for seed in raw["seeds"]:
            value = values[(dataset, backbone, metric, method)].get(seed)
            digits = 0 if unit in COUNTS else 5
            row[f"seed_{seed}"] = "" if value is None else f"{value:.{digits}f}"
        rows.append(row)
    name = "efficiency.partial.csv" if missing else "efficiency.csv"
    output = ROOT / "outputs" / study.name / name
    write_csv(output, rows)
    print(f"{output}: {len(rows)} rows from {len(runs) - len(missing)} runs")


def parameter_blocks(spec) -> dict[str, int]:
    """各块的可训练参数量；块与总数 ``parameters`` 不一致时报错。"""

    model = prepare(spec).model
    blocks = dict.fromkeys(BLOCKS.values(), 0)
    for name, parameter in model.named_parameters():
        blocks[BLOCKS[name.split(".")[0]]] += parameter.numel()
    total = sum(p.numel() for p in model.parameters())
    if sum(blocks.values()) != total:
        raise SystemExit(f"{spec.name}: parameter blocks do not sum to the total")
    return {**blocks, "parameters": total}


if __name__ == "__main__":
    main()
