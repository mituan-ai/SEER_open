from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seer.config import load_study, resolve_run

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "configs" / "studies"


def test_resolved_run_layers_and_artifacts() -> None:
    spec = resolve_run("ladle_preheating", "gru", "seer", 2024)
    assert spec.config["training"]["batch_size"] == 128 and spec.config["training"]["seed"] == 2024
    expected = "outputs/semantic_artifacts/ladle_preheating/deepseek-flash__qwen3.7-text-embedding"
    assert spec.config["method"]["artifact"] == expected
    assert spec.config["method"]["dataset"]["observed"][0] == "煤气阀门开度"
    assert spec.config["method"]["prior"].endswith("ladle_preheating/deepseek-flash__qwen3.7-text-embedding")
    assert spec.run_dir.parts[-4:] == ("ladle_preheating", "gru", "seer", "seed_2024")
    assert "artifact" not in resolve_run("ladle_preheating", "gru", "base", 2024).config["method"]


def test_default_valued_override_keeps_the_run_identity() -> None:
    default = resolve_run("indpensim", "gru", "seer", 2024)
    same = resolve_run("indpensim", "gru", "seer", 2024, {"method.embedding": "qwen3.7-text-embedding"})
    changed = resolve_run("indpensim", "gru", "seer", 2024, {"method.reader_heads": 2})
    assert same.run_dir == default.run_dir and same.config_sha256 == default.config_sha256
    assert changed.variant == "seer__method.reader_heads=2"
    assert (
        changed.pairing_sha256
        == default.pairing_sha256
        == resolve_run("indpensim", "gru", "base", 2024).pairing_sha256
    )


def test_runtime_fields_do_not_change_the_config_hash() -> None:
    spec = resolve_run("indpensim", "gru", "base", 2024)
    before = spec.config_sha256
    spec.config["training"]["device"] = "cpu"
    assert spec.config_sha256 == before


def test_invalid_overrides_are_rejected() -> None:
    with pytest.raises(KeyError):
        resolve_run("indpensim", "gru", "base", 2024, {"method.embedding": "qwen3.7-text-embedding"})
    with pytest.raises(KeyError):
        resolve_run("indpensim", "gru", "seer", 2024, {"method.embedding": "no-such-model"})
    with pytest.raises(ValueError):
        resolve_run("indpensim", "gru", "base", 2024, {"dataset.file": "x.csv"})


def test_smoke_study_and_run_config() -> None:
    smoke = load_study(STUDIES / "smoke.yaml")
    assert {(spec.dataset, spec.backbone, spec.method) for spec in smoke.runs} == {
        ("thickener_dewatering", "gru", "base"),
        ("thickener_dewatering", "gru", "seer"),
    }
    assert load_study(ROOT / "configs" / "run.yaml").runs


def test_data_and_artifact_contents_enter_the_run_identity(toy_config_dir: Path, tmp_path: Path) -> None:
    import os

    from seer.config import load_yaml

    before = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir)
    assert before.config["method"]["artifact_sha256"]
    bundle = Path(load_yaml(toy_config_dir / "datasets" / "toy.yaml")["path"]) / "stream.npz"
    with np.load(bundle) as arrays:
        content = dict(arrays)
    content["values"] = content["values"] + 1.0
    np.savez(bundle, **content)
    os.utime(bundle, ns=(1, 1))
    after = resolve_run("toy", "gru", "seer", 2024, config_dir=toy_config_dir)
    assert after.config_sha256 != before.config_sha256
    assert after.pairing_sha256 != before.pairing_sha256


def test_implementation_fingerprint_ignores_docstrings_and_comments(tmp_path: Path) -> None:
    from seer.config import implementation_sha256

    source = tmp_path / "src"
    source.mkdir()
    (source / "a.py").write_text('"""doc"""\n\ndef f():\n    return 1\n', encoding="utf-8")
    first = implementation_sha256(source)
    (source / "a.py").write_text('"""other doc"""\n# note\n\ndef f():\n    """x"""\n    return 1\n')
    implementation_sha256.cache_clear()
    assert implementation_sha256(source) == first
    (source / "a.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    implementation_sha256.cache_clear()
    assert implementation_sha256(source) != first
