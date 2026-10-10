from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.confirmatory_metrics import compute_confirmatory_descriptive_metrics
from scripts.research.analyze_confirmatory_closure import EXPLORATORY_CLASSIFICATION, analyze


SEEDS = [42, 1337, 2025, 3407, 9001]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    base_rows = [
        ("secret-source-a", "a0", "a/authentic.png", 0),
        ("secret-source-a", "a1", "a/edited.png", 1),
        ("secret-source-b", "b0", "b/authentic.png", 0),
        ("secret-source-b", "b1", "b/edited.png", 1),
    ]
    per_checkpoint = {}
    macro_values = []
    for offset, seed in enumerate(SEEDS):
        logits = np.asarray(
            [
                [0.2 - offset * 0.01, 0.0],
                [0.0, 0.2 + offset * 0.01],
                [0.0, 0.1 + offset * 0.01],
                [0.1, 0.0],
            ],
            dtype=np.float64,
        )
        records = [
            {
                "source_id": source,
                "sample_id": sample,
                "relative_path": relative,
                "true_label": label,
                "sample_idx": index,
                "logits": logits[index].tolist(),
            }
            for index, (source, sample, relative, label) in enumerate(base_rows)
        ]
        prediction_name = f"predictions_seed_{seed}.json"
        _write_json(
            outputs / prediction_name,
            {
                "seed": seed,
                "protocol": "stage1_frozen_backbone_linear_probe",
                "sample_size": 250,
                "predictions": records,
                "calibration_fitted": False,
                "threshold_tuned": False,
                "probability_ensemble": False,
            },
        )
        labels = np.asarray([row[3] for row in base_rows], dtype=np.int64)
        metrics = compute_confirmatory_descriptive_metrics(labels, logits)
        macro_values.append(metrics["macro_f1"])
        per_checkpoint[str(seed)] = {"metrics": metrics, "prediction_file": prediction_name}
    _write_json(outputs / "per_checkpoint_metrics.json", per_checkpoint)
    aggregate = {
        "aggregate_macro_f1": float(np.mean(macro_values)),
        "bootstrap_replicates": 10000,
        "bootstrap_rng_seed": 20261002,
        "ci_lower_95": 0.4,
        "ci_upper_95": 0.6,
        "bootstrap_distribution_sha256_float64_le": "a" * 64,
        "verdict": "INSUFFICIENT_CONFIRMATORY_EVIDENCE",
    }
    _write_json(outputs / "aggregate_confirmatory_metrics.json", aggregate)
    _write_json(
        outputs / "bootstrap_distribution_summary.json",
        {key: aggregate[key] for key in (
            "bootstrap_replicates",
            "bootstrap_rng_seed",
            "ci_lower_95",
            "ci_upper_95",
            "bootstrap_distribution_sha256_float64_le",
        )},
    )
    _write_json(
        outputs / "confirmatory_decision.json",
        {"success": False, "verdict": "INSUFFICIENT_CONFIRMATORY_EVIDENCE"},
    )
    _write_json(outputs / "execution_receipt.json", {"status": "COMPLETED_VALID"})
    required = [path for path in outputs.iterdir() if path.is_file()]
    checksum_payload = {
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)} for path in required
        }
    }
    _write_json(outputs / "checksums.json", checksum_payload)
    session_files = [path for path in outputs.iterdir() if path.is_file()]
    _write_json(
        outputs / "session_checksums.json",
        {
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
                for path in session_files
            }
        },
    )
    wrapper = tmp_path / "execution_wrapper_receipt.json"
    _write_json(
        wrapper,
        {
            "controller_exit_code": 0,
            "controller_completed": True,
            "outputs_exist": True,
            "confirmatory_result_exists": False,
        },
    )
    return outputs, wrapper


def test_closure_analysis_is_read_only_exploratory_and_redacted(tmp_path: Path) -> None:
    outputs, wrapper = _fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in outputs.iterdir() if path.is_file()}
    result_dir = tmp_path / "result"
    reconciliation, exploratory = analyze(
        outputs_dir=outputs,
        wrapper_receipt_path=wrapper,
        output_dir=result_dir,
        expected_sources=2,
        expected_samples=4,
    )
    after = {path.name: path.read_bytes() for path in outputs.iterdir() if path.is_file()}
    assert before == after
    assert reconciliation["confirmatory_verdict"] == "INSUFFICIENT_CONFIRMATORY_EVIDENCE"
    assert reconciliation["wrapper_adjudication"]["outer_wrapper_false_negative"] is True
    assert exploratory["analysis_classification"] == EXPLORATORY_CLASSIFICATION
    assert exploratory["scientific_guardrails"]["inference_calls"] == 0
    assert exploratory["scientific_guardrails"]["training_runs"] == 0
    serialized = json.dumps(exploratory)
    assert "secret-source-a" not in serialized
    assert "secret-source-b" not in serialized
    assert "a/authentic.png" not in serialized


def test_closure_analysis_rejects_checksum_mutation(tmp_path: Path) -> None:
    outputs, wrapper = _fixture(tmp_path)
    prediction = outputs / "predictions_seed_42.json"
    prediction.write_bytes(prediction.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Checksum verification failed"):
        analyze(
            outputs_dir=outputs,
            wrapper_receipt_path=wrapper,
            output_dir=tmp_path / "result",
            expected_sources=2,
            expected_samples=4,
        )
