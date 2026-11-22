"""冻结字段嵌入 (d, 4, k)；单字段更新不需要重新编码整卡。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from seer.io import sha256_json, write_json
from seer.semantic_cards import FIELDS as FIELD_NAMES
from seer.semantic_cards import card_texts

ARTIFACT_TYPE = "seer_field_semantics"


@dataclass(frozen=True)
class SemanticArtifact:
    variable_names: tuple[str, ...]
    fields: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.fields.shape[-1])

    def validate(self, feature_names: tuple[str, ...] | None = None) -> None:
        d = len(self.variable_names)
        if self.fields.shape != (d, len(FIELD_NAMES), 256) or not np.isfinite(self.fields).all():
            raise ValueError("field embeddings must be finite (d, 4, 256)")
        if len(set(self.variable_names)) != d:
            raise ValueError("duplicate variable names")
        if feature_names is not None and tuple(feature_names) != self.variable_names:
            raise ValueError("feature order does not match semantics")

    def save(self, output_dir: Path) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"artifact directory is not empty: {output_dir}")
        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / "semantics.npz", fields=self.fields)
        write_json(
            output_dir / "metadata.json",
            {
                **self.metadata,
                "artifact_type": ARTIFACT_TYPE,
                "variable_names": list(self.variable_names),
                "field_names": list(FIELD_NAMES),
                "semantic_dim": self.dim,
                "arrays_sha256": sha256_json(self.fields),
            },
        )


def load_artifact(artifact_dir: Path, feature_names: tuple[str, ...] | None = None) -> SemanticArtifact:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_type") != ARTIFACT_TYPE or metadata.get("field_names") != list(FIELD_NAMES):
        raise ValueError("not a current field semantic artifact")
    with np.load(artifact_dir / "semantics.npz") as arrays:
        artifact = SemanticArtifact(tuple(metadata["variable_names"]), arrays["fields"], metadata)
        artifact.validate(feature_names)
    if sha256_json(artifact.fields) != metadata["arrays_sha256"]:
        raise ValueError("semantic artifact hash mismatch")
    return artifact


EMBEDDING_PROVIDERS = {  # 接口类型 -> 设置输出维度的请求字段
    "openai_compatible": ("dimensions", "output_dimension"),
    "google_generative_language": ("outputDimensionality",),
    "cohere_native": ("output_dimension",),
}


@dataclass(frozen=True)
class EmbeddingConfig:
    """configs/embedding.yaml 中的一个模型条目（``Registry.entry``）。

    ``dimensions`` 为语义维度 k；``request_dimensions`` 为 API 请求维度，默认等于 k。
    请求维度较大时只保留前 k 维，该操作仅适用于经过 Matryoshka 训练的模型；当前登记的编码器
    均原生输出 k 维。字段向量取均值后按变量作 L2 归一化。
    """

    model: str
    base_url: str
    api_key_env: str
    dimensions: int
    provider: str = "openai_compatible"
    dimension_parameter: str = "dimensions"
    request_dimensions: int | None = None
    batch_size: int = 10
    timeout_seconds: float = 120.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> EmbeddingConfig:
        if not str(payload.get("model", "")).strip() or not str(payload.get("base_url", "")).strip():
            raise ValueError("embedding model and base_url must be set; fill configs/env/api.local.env")
        provider = str(payload.get("provider", "openai_compatible"))
        parameter = str(payload.get("dimension_parameter", "dimensions"))
        if provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        if parameter not in EMBEDDING_PROVIDERS[provider]:
            raise ValueError(
                f"{provider} sets the output dimension with one of {EMBEDDING_PROVIDERS[provider]}"
            )
        dimensions = int(payload["dimensions"])
        request = payload.get("request_dimensions")
        request = None if request in (None, "", "none") else int(request)
        if request is not None and request < dimensions:
            raise ValueError("request_dimensions must not be smaller than dimensions")
        return cls(
            model=str(payload["model"]).strip(),
            base_url=str(payload["base_url"]).strip(),
            api_key_env=str(payload["api_key_env"]),
            dimensions=dimensions,
            provider=provider,
            dimension_parameter=parameter,
            request_dimensions=request,
            batch_size=int(payload.get("batch_size", 10)),
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
        )

    @property
    def requested(self) -> int:
        return self.request_dimensions or self.dimensions

    def public(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension_parameter": self.dimension_parameter,
            "dimensions": self.dimensions,
            "request_dimensions": self.requested,
        }


def embed_texts(texts: list[str], config: EmbeddingConfig) -> tuple[np.ndarray, int | None]:
    """返回 (len(texts), k) 的编码与接口报告的输入 token 数（接口不报告时为 None）。"""

    vectors, tokens, _ = request_embeddings(texts, config)
    embeddings = np.asarray(vectors, dtype=np.float64)
    if embeddings.shape != (len(texts), config.requested):
        raise ValueError(
            f"embedding API returned shape {embeddings.shape}, not {(len(texts), config.requested)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding API returned non-finite values")
    return embeddings[:, : config.dimensions], tokens


def request_embeddings(
    texts: list[str], config: EmbeddingConfig
) -> tuple[list[list[float]], int | None, str | None]:
    """按接口类型发送编码请求，返回未经检查的向量、输入 token 数与接口报告的模型名。

    正式编码（``embed_texts``）与 ``scripts/check_api.py`` 共用此函数，二者的请求体因此一致。
    """

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
    if config.provider == "openai_compatible":
        return _embed_openai_compatible(texts, config, api_key)
    if config.provider == "cohere_native":
        return _embed_cohere(texts, config, api_key)
    rows = [_embed_google(text, config, api_key) for text in texts]
    model = rows[0][2] if rows else None
    return [row for row, _, _ in rows], _total([count for _, count, _ in rows]), model


def _total(counts: list[int | None]) -> int | None:
    return None if not counts or any(count is None for count in counts) else int(sum(counts))


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"embedding API returned HTTP {error.code}: {error.read()[:300]!r}") from error


def _embed_openai_compatible(
    texts: list[str], config: EmbeddingConfig, api_key: str
) -> tuple[list[list[float]], int | None, str | None]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds, max_retries=0)
    vectors: list[list[float]] = []
    tokens: list[int | None] = []
    model = None
    try:
        for start in range(0, len(texts), config.batch_size):
            batch = texts[start : start + config.batch_size]
            request: dict[str, Any] = {"model": config.model, "input": batch}
            if config.dimension_parameter == "dimensions":
                request["dimensions"] = config.requested
            else:
                request["extra_body"] = {"output_dimension": config.requested}
            response = client.embeddings.create(**request)
            usage = getattr(response, "usage", None)
            tokens.append(getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None))
            model = getattr(response, "model", None) or model
            data = sorted(response.data, key=lambda item: int(item.index))
            if [int(item.index) for item in data] != list(range(len(batch))):
                raise ValueError("embedding API returned missing or reordered rows")
            vectors.extend(item.embedding for item in data)
    finally:
        client.close()
    return vectors, _total(tokens), model


def _embed_cohere(
    texts: list[str], config: EmbeddingConfig, api_key: str
) -> tuple[list[list[float]], int | None, str | None]:
    """使用原生 v2 接口（OpenAI 兼容端点不支持 dimensions）；该接口不返回模型名。"""

    vectors: list[list[float]] = []
    tokens: list[int | None] = []
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        payload = _post_json(
            f"{config.base_url.rstrip('/')}/embed",
            {
                "model": config.model,
                "texts": batch,
                "input_type": "clustering",
                "output_dimension": config.requested,
                "embedding_types": ["float"],
                "truncate": "NONE",
            },
            {"Authorization": f"Bearer {api_key}"},
            config.timeout_seconds,
        )
        rows = (payload.get("embeddings") or {}).get("float")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise ValueError("Cohere embedding API returned missing embedding rows")
        vectors.extend(rows)
        tokens.append(((payload.get("meta") or {}).get("billed_units") or {}).get("input_tokens"))
    return vectors, _total(tokens), None


def _embed_google(
    text: str, config: EmbeddingConfig, api_key: str
) -> tuple[list[float], int | None, str | None]:
    model = config.model if config.model.startswith("models/") else f"models/{config.model}"
    payload = _post_json(
        f"{config.base_url.rstrip('/')}/{model}:embedContent",
        {"content": {"parts": [{"text": text}]}, "outputDimensionality": config.requested},
        {"x-goog-api-key": api_key},
        config.timeout_seconds,
    )
    values = (payload.get("embedding") or {}).get("values")
    if not isinstance(values, list):
        raise ValueError("Google embedding API returned no embedding values")
    return values, (payload.get("usageMetadata") or {}).get("promptTokenCount"), payload.get("modelVersion")


def build_from_cards(
    cards: list[dict[str, Any]], *, embedding: EmbeddingConfig, provenance: dict[str, Any]
) -> SemanticArtifact:
    """``provenance`` 记录语义卡路径与生成 LLM，只用于追溯。"""

    texts = card_texts(cards)
    names = tuple(card["variable_name"] for card in cards)
    start = time.perf_counter()
    vectors, tokens = embed_texts(texts, embedding)
    return SemanticArtifact(
        names,
        vectors.reshape(len(names), len(FIELD_NAMES), embedding.dimensions),
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            **provenance,
            "cards_sha256": sha256_json(texts),
            "embedding": embedding.public(),
            "embedding_generation": {
                "texts": len(texts),
                "seconds": round(time.perf_counter() - start, 3),
                "input_tokens": tokens,
            },
        },
    )


class EncodingCache:
    """按文本哈希缓存的字段编码（JSONL，每行 {"sha256", "vector"}），一个编码器配置一个文件。"""

    def __init__(self, path: Path, config: EmbeddingConfig) -> None:
        self.path, self.config = path, config
        self.vectors: dict[str, np.ndarray] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.vectors[row["sha256"]] = np.asarray(row["vector"], dtype=np.float64)

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [sha256_json(text) for text in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self.vectors))
        if missing:
            todo = [texts[keys.index(k)] for k in missing]
            vectors, _ = embed_texts(todo, self.config)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                for key, vector in zip(missing, vectors, strict=True):
                    self.vectors[key] = vector
                    handle.write(json.dumps({"sha256": key, "vector": vector.tolist()}) + "\n")
        return np.stack([self.vectors[k] for k in keys])


def revise_fields(
    fields: np.ndarray,
    names: tuple[str, ...],
    revisions: dict[str, dict[str, str]],
    cache: EncodingCache,
) -> np.ndarray:
    """把 ``revisions``（变量 -> 字段 -> 新文本）编码后替换进 (d, F, k) 字段嵌入，其余字段不变。"""

    items = [
        (names.index(v), FIELD_NAMES.index(f), text)
        for v, row in revisions.items()
        for f, text in row.items()
    ]
    out = np.array(fields, dtype=np.float64, copy=True)
    if items:
        vectors = cache.encode([text for _, _, text in items])
        for (i, f, _), vector in zip(items, vectors, strict=True):
            out[i, f] = vector
    return out


def build_random(names: tuple[str, ...], *, dim: int, seed: int) -> SemanticArtifact:
    """仅供离线单元测试，不作为实验方法。"""
    return SemanticArtifact(names, np.random.default_rng(seed).normal(size=(len(names), 4, dim)))
