"""把语义卡的四个字段编码为冻结字段向量。

    # 用默认编码器编码默认 LLM 的语义卡
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all
    # 语义来源实验：各 LLM 配默认编码器，各编码器配默认 LLM
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all --llm all
    uv run --extra llm python scripts/build_semantic_artifact.py --dataset all --embedding all

读取 outputs/semantics/<dataset>/<llm>/ 中的语义卡，产物写入 <artifacts>/<llm>__<embedding>/。
除非指定 --force，否则保留现有有效产物；语义卡、材料或编码器配置变化后，旧产物报告 stale 并失败。
缺少语义卡或未配置编码器的模型对会跳过；有任一失败时退出码为 1。
"""

from __future__ import annotations

import argparse
import shutil

from seer.config import CONFIG_DIR, artifact_dir, dataset_config
from seer.data import read_feature_names
from seer.io import ROOT, load_api_env_files, resolve_path, sha256_json
from seer.semantic_artifact import EmbeddingConfig, build_from_cards, load_artifact
from seer.semantic_cards import card_texts, cards_path, dataset_materials, load_cards
from seer.semantic_models import load_registry

DATASETS = ("ladle_preheating", "thickener_dewatering", "indpensim")


def build(dataset: str, llm: str, embedding: str, encoders, *, force: bool) -> str:
    config = dataset_config(dataset)
    names = read_feature_names(config)
    cards = cards_path(dataset, llm)
    if not cards.exists():
        return "no cards"
    if not encoders.configured(embedding):
        return "not configured"
    encoder = EmbeddingConfig.from_mapping(encoders.entry(embedding))
    payload = load_cards(cards, dataset_materials(config))  # 语义卡与当前材料不一致时在此报告 stale
    output = resolve_path(artifact_dir(config, llm, embedding))
    if output.exists() and any(output.iterdir()):
        metadata = load_artifact(output, names).metadata
        current = metadata.get("cards_sha256") == sha256_json(card_texts(payload["cards"]))
        if current and metadata.get("embedding") == encoder.public() and not force:
            return "exists"
        if not force:
            raise ValueError(
                f"stale artifact {output.relative_to(ROOT)}: cards or encoder changed; use --force"
            )
        shutil.rmtree(output)
    provenance = {"cards_path": str(cards.relative_to(ROOT)), "llm": payload["llm"]}
    artifact = build_from_cards(payload["cards"], embedding=encoder, provenance=provenance)
    artifact.validate(names)
    artifact.save(output)
    return f"built {output.relative_to(ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", nargs="+", required=True, help="dataset names or 'all'")
    parser.add_argument("--llm", default="default", help="a model of configs/llm.yaml, 'default', or 'all'")
    parser.add_argument("--embedding", default="default", help="an embedding model, 'default', or 'all'")
    parser.add_argument("--force", action="store_true", help="rebuild existing artifacts")
    args = parser.parse_args()
    load_api_env_files()
    llms = load_registry(CONFIG_DIR / "llm.yaml")
    encoders = load_registry(CONFIG_DIR / "embedding.yaml")
    outcomes = []
    for dataset in DATASETS if args.dataset == ["all"] else args.dataset:
        for llm in llms.select(args.llm):
            for embedding in encoders.select(args.embedding):
                try:
                    outcome = build(dataset, llm, embedding, encoders, force=args.force)
                except Exception as error:  # 继续处理其余模型对，错误在汇总中报告
                    outcome = f"FAILED {type(error).__name__}: {str(error)[:300]}"
                print(f"{dataset:<22} {f'{llm} + {embedding}':<48} {outcome}", flush=True)
                outcomes.append(outcome)
    if any(outcome.startswith("FAILED") for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
