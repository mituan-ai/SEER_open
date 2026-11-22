"""构建合成语义先验的描述池（seer.priors）。

    # 只抽样并写出物理描述与覆盖统计供审阅：outputs/semantic_priors/<dataset>/sample_review/
    uv run python scripts/build_semantic_prior.py --dataset all
    # 调用默认 LLM 写描述、默认编码器编码
    uv run --extra llm python scripts/build_semantic_prior.py --dataset all --call-api

描述写入 outputs/semantic_priors/<dataset>/<llm>/descriptions.json（变换参数、物理描述、各措辞、生成记录），
描述池写入 <priors>/<llm>__<embedding>/（pool.npz、metadata.json）。已有描述与当前抽样设置、语义卡、
材料一致时直接编码；不一致时报告 stale 并失败，需用 --force 重建。
"""

from __future__ import annotations

import argparse

import numpy as np

from seer.config import CONFIG_DIR, artifact_dir, dataset_config, dataset_spec, prior_dir
from seer.data import build_split, load_dataset, training_normalization
from seer.io import ROOT, load_api_env_files, read_json, resolve_path, sha256_json, write_json
from seer.llm import LLMConfig
from seer.priors import coverage, describe, physical_description, sample_transforms, stamp, usage_summary
from seer.semantic_artifact import EmbeddingConfig, embed_texts, load_artifact
from seer.semantic_cards import FIELDS, cards_path, dataset_materials, load_cards
from seer.semantic_models import load_registry, model_slug
from seer.semantic_prior import PriorPool

DATASETS = ("ladle_preheating", "thickener_dewatering", "indpensim")
STAMP_KEYS = ("prompt_version", "phrasings", "count", "seed", "prior", "cards_sha256", "task_spec_sha256")


def sample(dataset: str, count: int, seed: int):
    config = dataset_config(dataset)
    spec = dataset_spec(config)
    data = load_dataset(config)
    split = build_split(data, config)
    values = training_normalization(data, split).inputs(data.values)
    ranges = [(data.segments[i].start, data.segments[i].stop) for i in split.train]
    observed = tuple(data.feature_names.index(name) for name in spec["observed"])
    transforms, details, rejected = sample_transforms(values, ranges, observed, spec["prior"], count, seed)
    names = data.feature_names
    items = [
        {
            "id": f"t{index:03d}",
            "target": names[int(transforms.target[index])],
            "source": names[int(transforms.source[index])],
            "description": physical_description(
                names,
                int(transforms.target[index]),
                int(transforms.source[index]),
                detail,
                spec["sampling_minutes"],
            ),
            "detail": detail,
        }
        for index, detail in enumerate(details)
    ]
    report = coverage(transforms, details, rejected, names, spec["sampling_minutes"])
    return config, spec, data, transforms, items, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", nargs="+", required=True, help="dataset names or 'all'")
    parser.add_argument("--count", type=int, default=300, help="transforms per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", default="default")
    parser.add_argument("--embedding", default="default")
    parser.add_argument("--call-api", action="store_true")
    parser.add_argument("--force", action="store_true", help="regenerate existing descriptions")
    args = parser.parse_args()
    for dataset in DATASETS if args.dataset == ["all"] else args.dataset:
        seed = args.seed + DATASETS.index(dataset)  # 各数据集的随机序列互不相同
        config, spec, data, transforms, items, report = sample(dataset, args.count, seed)
        if not args.call_api:
            review = ROOT / "outputs" / "semantic_priors" / dataset / "sample_review"
            write_json(review / "transforms.json", {"coverage": report, "items": items})
            print(dataset, review / "transforms.json", report["rejected"], "rejected")
            continue
        load_api_env_files()
        llms = load_registry(CONFIG_DIR / "llm.yaml")
        encoders = load_registry(CONFIG_DIR / "embedding.yaml")
        llm, embedding = llms.resolve(args.llm), encoders.resolve(args.embedding)
        materials = dataset_materials(config)
        cards = load_cards(cards_path(dataset, llm), materials)["cards"]
        current = stamp(materials, cards, spec["prior"], args.count, seed)
        path = ROOT / "outputs" / "semantic_priors" / dataset / model_slug(llm) / "descriptions.json"
        if path.exists() and not args.force:
            saved = read_json(path)
            # 变换在训练段上抽样：训练数据或划分改变后，已保存的描述不再对应当前变换
            if any(saved["stamp"][key] != current[key] for key in STAMP_KEYS) or sha256_json(
                saved["items"]
            ) != sha256_json(items):
                raise SystemExit(f"stale descriptions {path}; rebuild with --force")
            described = saved["descriptions"]
        else:
            llm_config = LLMConfig.from_mapping(llms.entry(llm))
            described, records = describe(materials, cards, items, llm_config)
            payload = {
                "stamp": current,
                "llm": llm_config.public(),
                "coverage": report,
                "items": items,
                "descriptions": described,
                "generation": {"usage": usage_summary(records), "records": records},
            }
            write_json(path if described else path.with_suffix(".failed.json"), payload)
            if described is None:
                raise SystemExit(f"description failed for {dataset}; see {path.with_suffix('.failed.json')}")
            print(dataset, "described", usage_summary(records))
        encoder = EmbeddingConfig.from_mapping(encoders.entry(embedding))
        original = load_artifact(
            resolve_path(artifact_dir(config, llm, embedding)), data.feature_names
        ).fields
        texts = [p[f] for entry in described for p in entry["phrasings"] for f in entry["fields"]]
        vectors, tokens = embed_texts(texts, encoder)
        fields = np.repeat(original[transforms.target][:, None], len(described[0]["phrasings"]), axis=1)
        affected = np.zeros((len(described), len(FIELDS)), dtype=bool)
        cursor = 0
        for row, entry in enumerate(described):
            for phrasing in range(len(entry["phrasings"])):
                for field in entry["fields"]:
                    fields[row, phrasing, FIELDS.index(field)] = vectors[cursor]
                    cursor += 1
            affected[row, [FIELDS.index(f) for f in entry["fields"]]] = True
        PriorPool(
            data.feature_names,
            transforms,
            affected,
            fields,
            {
                "descriptions_path": str(path.relative_to(ROOT)),
                "descriptions_sha256": sha256_json(described),
                "embedding": encoder.public(),
                "embedding_input_tokens": tokens,
                "dataset_fingerprint": sha256_json(data.values),
            },
        ).save(resolve_path(prior_dir(config, llm, embedding)))
        print(dataset, "pool", prior_dir(config, llm, embedding), fields.shape)


if __name__ == "__main__":
    main()
