# SEER

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-1D6B73?style=flat-square&logo=python&logoColor=white&labelColor=16191E" alt="Python 3.14">
  <img src="https://img.shields.io/badge/pytorch-2.11-C2410C?style=flat-square&logo=pytorch&logoColor=white&labelColor=16191E" alt="PyTorch 2.11">
  <img src="https://img.shields.io/badge/env-uv-6A6F77?style=flat-square&logo=uv&logoColor=white&labelColor=16191E" alt="uv">
  <img src="https://img.shields.io/badge/license-MIT-1D6B73?style=flat-square&labelColor=16191E" alt="MIT License">
  <br>
  <a href="./README.md">English</a> · <b>简体中文</b>
</p>

**When Data Overrule Documents: Revising Industrial Process Knowledge with LLMs** 的开源实现。

工业预测模型越来越多地读取过程文档：每个传感器测什么、如何计算、受哪些变量驱动。装置会变，文档却往往滞后。
SEER 研究的是：在没有新目标标签、只有输入流的情况下，证据何时足以修订文档，以及如何把修订后的文档交给冻结的预测网络。

文档是逐变量的语义卡，由 LLM 根据 `resources/datasets/<dataset>/` 中部署前的资料（`task_spec.json` 与
`PROCESS.md`）离线写成一次。语义读取层把语义卡嵌入转成逐变量、跨来源与时滞的因果核，因此修订改变的是冻结网络
读取输入的方式，而不改动任何参数。部署时，观测头用到来的记录检验 LLM 提出的修订，只有证据足够强时才采纳。

## 快速开始

```bash
uv sync --extra train
uv run python scripts/train.py --dataset thickener_dewatering --backbone gru --method seer \
  --seed 2024 --set training.max_epochs=2
```

仓库自带三个数据集的处理后数据（`stream.npz`），不附带任何语义内容。`--method base` 可以直接训练；
`--method seer` 需要先按下文“生成语义”自行生成语义卡、嵌入与合成语义先验。没有 GPU 时加 `--device cpu`。

| 数据集 | 任务 |
| --- | --- |
| `ladle_preheating` | 60 步历史 → 未来 5 步温度（现场数据，已缩放到 [0, 1]） |
| `thickener_dewatering` | 30 分钟历史 → 10 分钟后的底流浓度 |
| `indpensim` | 120 步历史 → 离线青霉素浓度（IndPenSim） |

## 用法

- **选项：** `--backbone` 取 `gru lstm transformer informer mamba itransformer patchtst moderntcn` 之一；
  `--method` 取 `base` 或 `seer`。任意配置字段可用 `--set section.field=value` 覆盖。
- **批量运行：** `uv run python scripts/run_study.py [configs/studies/smoke.yaml]` 在全部可见 GPU 上运行各组合
  （默认读取 `configs/run.yaml`）。`scripts/collect_results.py` 把计算开销整理为 CSV。
- **在线评测：** `uv run python scripts/run_online.py configs/studies/smoke.yaml` 在常规、变化与漂移流
  （`resources/datasets/<dataset>/changes.json`）上评测已训练的检查点。`--modes base off` 不需要密钥；
  `seer`（在线修订）在证据触发后调用 LLM 与编码器，需要 `uv sync --extra llm` 和下面的 API 配置。
- **API 配置：** 把 `configs/env/api.env` 复制为 `configs/env/api.local.env` 并填写接口地址与密钥；
  `scripts/check_api.py` 检查连通性。
- **生成语义：** `scripts/build_semantic_cards.py --dataset all --call-api`、
  `scripts/build_semantic_artifact.py --dataset all` 与 `scripts/build_semantic_prior.py --dataset all --call-api`。
  LLM 只读取 `task_spec.json` 与 `PROCESS.md`。

## 目录

```text
src/seer/              语义读取层、观测头、证据、合成先验、主干网络、训练
configs/               数据集、共享层与实验清单；run.yaml 选择要执行的运行
scripts/               训练、调度、在线评测、语义构建、API 检查
resources/datasets/    各数据集：task_spec.json、PROCESS.md、seer.json、changes.json、stream.npz
outputs/               生成的语义、运行、日志与缓存（均不提交）
```

## 许可

MIT。