# SEER

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-1D6B73?style=flat-square&logo=python&logoColor=white&labelColor=16191E" alt="Python 3.14">
  <img src="https://img.shields.io/badge/pytorch-2.11-C2410C?style=flat-square&logo=pytorch&logoColor=white&labelColor=16191E" alt="PyTorch 2.11">
  <img src="https://img.shields.io/badge/env-uv-6A6F77?style=flat-square&logo=uv&logoColor=white&labelColor=16191E" alt="uv">
  <img src="https://img.shields.io/badge/license-MIT-1D6B73?style=flat-square&labelColor=16191E" alt="MIT License">
  <br>
  <b>English</b> · <a href="./README.zh-CN.md">简体中文</a>
</p>

Open-source implementation of **When Data Overrule Documents: Revising Industrial Process Knowledge with LLMs**.

Industrial predictors increasingly read process documents: what each sensor measures, how it is computed,
and which variables drive it. Plants change, and the documents lag behind. SEER asks when the input stream
alone, without new target labels, is strong enough evidence to revise a document, and how to feed the revised
document to a frozen predictor.

The documents are per-variable semantic cards. An LLM writes them once, offline, from the deployment-time
materials in `resources/datasets/<dataset>/` (`task_spec.json` and `PROCESS.md`). A semantic reader turns
the card embeddings into a per-variable causal kernel over sources and lags, so a revision changes how the
frozen network reads its inputs without changing any parameter. At deployment, an observation head scores
LLM-proposed revisions against the incoming records, and a revision is adopted only when the evidence is
strong enough.

## Quick start

```bash
uv sync --extra train
uv run python scripts/train.py --dataset thickener_dewatering --backbone gru --method seer \
  --seed 2024 --set training.max_epochs=2
```

Processed data (`stream.npz`) for all three datasets is included; no semantic content is shipped.
`--method base` trains directly; `--method seer` first needs the semantic cards, embeddings and synthetic
semantic prior, generated as described under *Generating semantics*. Add `--device cpu` if no GPU is available.

| Dataset | Task |
| --- | --- |
| `ladle_preheating` | 60-step history → next 5 temperatures (field data, scaled to [0, 1]) |
| `thickener_dewatering` | 30-min history → underflow concentration 10 min ahead |
| `indpensim` | 120-step history → offline penicillin concentration (IndPenSim) |

## Usage

- **Options:** `--backbone` is one of `gru lstm transformer informer mamba itransformer patchtst moderntcn`;
  `--method` is `base` or `seer`. Override any config field with `--set section.field=value`.
- **Batch runs:** `uv run python scripts/run_study.py [configs/studies/smoke.yaml]` runs every
  combination on all visible GPUs (default: `configs/run.yaml`).
  `scripts/collect_results.py` gathers the computational cost into a CSV.
- **Online evaluation:** `uv run python scripts/run_online.py configs/studies/smoke.yaml` evaluates
  the trained checkpoints on regular, changed and drift streams (`resources/datasets/<dataset>/changes.json`).
  `--modes base off` needs no API key; `seer` (online revision) calls the LLM and the embedding model after
  the evidence triggers, so it needs `uv sync --extra llm` and the API settings below.
- **API settings:** copy `configs/env/api.env` to `configs/env/api.local.env` and fill in the
  endpoints; `scripts/check_api.py` checks them.
- **Generating semantics:** `scripts/build_semantic_cards.py --dataset all --call-api`,
  `scripts/build_semantic_artifact.py --dataset all` and
  `scripts/build_semantic_prior.py --dataset all --call-api`. The LLM reads only
  `task_spec.json` and `PROCESS.md`.

## Layout

```text
src/seer/              semantic reader, observation head, evidence, synthetic prior, backbones, training
configs/               datasets, shared layers, studies; run.yaml selects the runs to execute
scripts/               training, scheduling, online evaluation, semantic builds, API check
resources/datasets/    per dataset: task_spec.json, PROCESS.md, seer.json, changes.json, stream.npz
outputs/               generated semantics, runs, logs and caches (all ignored)
```

## License

MIT.
