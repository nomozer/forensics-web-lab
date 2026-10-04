"""Synthetic and fault-injection coverage for Phase 4C.2G confirmatory execution.

This suite must never read the real locked-test partition, run scientific
inference, create an authorization artifact, or invoke the UAC/readiness path.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import io
import re
import tarfile
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_synthetic_locked_manifest(tmp_path: Path) -> tuple[Path, Path, str, str]:
    root = tmp_path / "locked-fixture"
    root.mkdir()
    samples = []
    source_ids = [f"synthetic-{idx:03d}" for idx in range(343)]
    for source_id in source_ids:
        for label, label_id in (("authentic", 0), ("ai_edited", 1)):
            relative_path = Path("samples") / source_id / f"{label}.png"
            sample_path = root / relative_path
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_bytes(f"{source_id}:{label}".encode("utf-8"))
            samples.append(
                {
                    "sample_id": f"{source_id}-{label}",
                    "unique_source_id": source_id,
                    "relative_path": relative_path.as_posix(),
                    "sha256": _sha256(sample_path),
                    "label": label,
                    "label_id": label_id,
                    "partition": "locked_test",
                }
            )

    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "synthetic-locked-fixture",
        "partition": "locked_test",
        "samples": samples,
    }
    manifest_path = tmp_path / "locked_test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    split_payload = json.dumps(sorted(source_ids))
    split_seal = hashlib.sha256(split_payload.encode("utf-8")).hexdigest()
    return root, manifest_path, _sha256(manifest_path), split_seal


def _build_production_custodian_evaluator_fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "custodian-evaluator-root"
    output = tmp_path / "custodian-output"
    root.mkdir()
    samples = []
    source_ids = [f"integration-{idx:03d}" for idx in range(343)]
    for source_id in source_ids:
        for label, label_id in (("authentic", 0), ("ai_edited", 1)):
            relative_path = f"samples/{source_id}/{label}.bin"
            sample_path = root / relative_path
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_bytes(f"{source_id}:{label}".encode("utf-8"))
            samples.append(
                {
                    "sample_id": f"{source_id}-{label}",
                    "unique_source_id": source_id,
                    "relative_path": relative_path,
                    "expected_sha256": _sha256(sample_path),
                    "label": label,
                    "label_id": label_id,
                    "partition": "locked_test",
                }
            )

    inventory_path = root / "custodian_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "dataset_id": "synthetic-custodian-evaluator-integration",
                "partition": "locked_test",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    split_seal = hashlib.sha256(
        json.dumps(sorted(source_ids)).encode("utf-8")
    ).hexdigest()
    session_id = "synthetic-custodian-evaluator-integration"
    read_only_receipt = tmp_path / "read-only-receipt.json"
    read_only_receipt.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "locked_test_root": str(root.resolve()),
                "status": "READ_ONLY_VOLUME_VERIFIED",
                "os_enforced_read_only": True,
                "evidence_kind": "synthetic_read_only_fixture",
                "backing_volume_unique_id": "synthetic-integration-volume",
                "canary_writes_performed": 0,
            }
        ),
        encoding="utf-8",
    )
    isolation_receipt = tmp_path / "isolation-receipt.json"
    isolation_receipt.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "isolation_verified": True,
                "proxy_enabled": False,
                "remaining_active_default_routes": 0,
                "active_egress_adapters": [],
                "active_vpn_route_owners": [],
                "unidentified_route_owners": [],
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "output": output,
        "inventory_path": inventory_path,
        "session_id": session_id,
        "read_only_receipt": read_only_receipt,
        "isolation_receipt": isolation_receipt,
        "split_seal": split_seal,
    }


def _seal_production_custodian_fixture(fixture: dict[str, object]) -> tuple[Path, dict]:
    from scripts.research.seal_locked_test_manifest import seal_manifest_session

    receipt = seal_manifest_session(
        locked_test_root=fixture["root"],
        output_root=fixture["output"],
        session_id=fixture["session_id"],
        read_only_receipt_path=fixture["read_only_receipt"],
        isolation_receipt_path=fixture["isolation_receipt"],
        manifest_schema_path=REPO_ROOT
        / "docs/schemas/locked-test-manifest.v1.schema.json",
        expected_split_seal=fixture["split_seal"],
        allow_synthetic_read_only=True,
    )
    return Path(fixture["output"]) / "locked_test_manifest.json", receipt


def test_phase_4c2g04_records_no_human_authorization() -> None:
    environment_path = (
        REPO_ROOT / "research/evidence/phase-4c.2g.0.4/environment.json"
    )
    report_path = REPO_ROOT / "research/evidence/phase-4c.2g.0.4/PHASE_REPORT.md"

    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    authorization = environment["authorization"]

    assert authorization["human_approval_statement_received"] is False
    assert authorization["authorization_artifact_created"] is False
    assert authorization["authorization_consumed"] is False

    report = report_path.read_text(encoding="utf-8")
    assert "The human user explicitly approved" not in report
    assert (
        "No explicit user-authored authorization statement has been received. "
        "The assistant previously supplied example authorization wording, which "
        "does not constitute human approval."
    ) in report


def test_locked_test_manifest_contract_accepts_exact_paired_fixture(tmp_path: Path) -> None:
    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    root, manifest_path, manifest_sha256, split_seal = (
        _build_synthetic_locked_manifest(tmp_path)
    )
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"

    loaded = load_locked_test_manifest(
        locked_test_root=root,
        manifest_path=manifest_path,
        manifest_schema_path=schema_path,
        expected_manifest_sha256=manifest_sha256,
        expected_schema_sha256=_sha256(schema_path),
        expected_split_seal=split_seal,
    )

    assert len(loaded.samples) == 686
    assert len(loaded.source_ids) == 343
    assert loaded.label_map == {"authentic": 0, "ai_edited": 1}
    assert loaded.manifest_sha256 == manifest_sha256
    assert loaded.split_seal == split_seal


def test_production_custodian_manifest_round_trips_into_production_evaluator_loader(
    tmp_path: Path,
) -> None:
    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    fixture = _build_production_custodian_evaluator_fixture(tmp_path)
    manifest_path, receipt = _seal_production_custodian_fixture(fixture)
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"

    loaded = load_locked_test_manifest(
        locked_test_root=fixture["root"],
        manifest_path=manifest_path,
        manifest_schema_path=schema_path,
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_schema_sha256=_sha256(schema_path),
        expected_split_seal=fixture["split_seal"],
    )

    assert receipt["sample_count"] == len(loaded.samples) == 686
    assert receipt["source_count"] == len(loaded.source_ids) == 343
    assert Path(fixture["inventory_path"]).is_file()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("missing_sample", "Missing or non-regular sample"),
        ("extra_file", "file inventory mismatch"),
        ("nested_metadata_name", "file inventory mismatch"),
        ("tampered_sample", "Sample SHA-256 mismatch"),
    ),
)
def test_production_custodian_evaluator_integration_rejects_sample_mutations(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    fixture = _build_production_custodian_evaluator_fixture(tmp_path)
    manifest_path, receipt = _seal_production_custodian_fixture(fixture)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(fixture["root"])
    first_sample = root / manifest["samples"][0]["relative_path"]
    if mutation == "missing_sample":
        first_sample.unlink()
    elif mutation == "extra_file":
        (root / "unexpected.bin").write_bytes(b"unexpected")
    elif mutation == "nested_metadata_name":
        nested = root / "samples" / "custodian_inventory.json"
        nested.write_text("{}", encoding="utf-8")
    elif mutation == "tampered_sample":
        first_sample.write_bytes(b"tampered-sample")
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(f"Unknown mutation: {mutation}")

    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"
    with pytest.raises((FileNotFoundError, ValueError), match=expected_error):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_schema_sha256=_sha256(schema_path),
            expected_split_seal=fixture["split_seal"],
        )


def test_exact_root_custodian_metadata_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    fixture = _build_production_custodian_evaluator_fixture(tmp_path)
    manifest_path, receipt = _seal_production_custodian_fixture(fixture)
    inventory_path = Path(fixture["inventory_path"]).resolve()
    original_is_symlink = Path.is_symlink

    def fixture_is_symlink(path: Path) -> bool:
        if path.resolve(strict=False) == inventory_path:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", fixture_is_symlink)
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"
    with pytest.raises(ValueError, match="Symlink/reparse point forbidden"):
        load_locked_test_manifest(
            locked_test_root=fixture["root"],
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_schema_sha256=_sha256(schema_path),
            expected_split_seal=fixture["split_seal"],
        )


def test_locked_test_manifest_contract_rejects_checksum_pair_duplicate_missing_and_extra(
    tmp_path: Path,
) -> None:
    import pytest

    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    root, manifest_path, original_sha, split_seal = _build_synthetic_locked_manifest(
        tmp_path
    )
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"
    schema_sha = _sha256(schema_path)
    original_text = manifest_path.read_text(encoding="utf-8")

    manifest_path.write_text(original_text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=original_sha,
            expected_schema_sha256=schema_sha,
            expected_split_seal=split_seal,
        )

    manifest = json.loads(original_text)
    manifest["samples"][1]["label_id"] = 0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Label mapping mismatch"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=_sha256(manifest_path),
            expected_schema_sha256=schema_sha,
            expected_split_seal=split_seal,
        )

    manifest = json.loads(original_text)
    manifest["samples"][1]["sample_id"] = manifest["samples"][0]["sample_id"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=_sha256(manifest_path),
            expected_schema_sha256=schema_sha,
            expected_split_seal=split_seal,
        )

    manifest_path.write_text(original_text, encoding="utf-8")
    missing_path = root / manifest["samples"][2]["relative_path"]
    missing_path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing or non-regular sample"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=original_sha,
            expected_schema_sha256=schema_sha,
            expected_split_seal=split_seal,
        )

    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.write_bytes(
        f"{manifest['samples'][2]['unique_source_id']}:{manifest['samples'][2]['label']}".encode(
            "utf-8"
        )
    )
    extra_path = root / "unexpected.bin"
    extra_path.write_bytes(b"extra")
    with pytest.raises(ValueError, match="file inventory mismatch"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=original_sha,
            expected_schema_sha256=schema_sha,
            expected_split_seal=split_seal,
        )


def test_locked_manifest_rejects_duplicate_source_label_pair_explicitly(
    tmp_path: Path,
) -> None:
    import pytest

    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    root, manifest_path, _, split_seal = _build_synthetic_locked_manifest(tmp_path)
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][2]["unique_source_id"] = manifest["samples"][0][
        "unique_source_id"
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate source/label pair"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=_sha256(manifest_path),
            expected_schema_sha256=_sha256(schema_path),
            expected_split_seal=split_seal,
        )


def test_manifest_relative_path_contract_rejects_absolute_drive_and_traversal() -> None:
    import pytest

    from ml.evaluation.phase_4c2g_dataset import validate_manifest_relative_path

    assert validate_manifest_relative_path("samples/source/authentic.png").as_posix() == (
        "samples/source/authentic.png"
    )
    for unsafe in (
        "/absolute.png",
        "../escape.png",
        "samples/../../escape.png",
        "C:/locked/escape.png",
        r"C:\locked\escape.png",
        r"\\server\share\escape.png",
    ):
        with pytest.raises(ValueError, match="Unsafe manifest relative_path"):
            validate_manifest_relative_path(unsafe)


def test_locked_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    import pytest

    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest

    root, manifest_path, _, split_seal = _build_synthetic_locked_manifest(tmp_path)
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_sample = outside / "authentic.png"
    escaped_sample.write_bytes(b"outside locked fixture")
    escape_link = root / "escape-link"
    try:
        escape_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(escape_link), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["relative_path"] = "escape-link/authentic.png"
    manifest["samples"][0]["sha256"] = _sha256(escaped_sample)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes locked-test root"):
        load_locked_test_manifest(
            locked_test_root=root,
            manifest_path=manifest_path,
            manifest_schema_path=schema_path,
            expected_manifest_sha256=_sha256(manifest_path),
            expected_schema_sha256=_sha256(schema_path),
            expected_split_seal=split_seal,
        )


def test_stage1_model_loader_recreates_frozen_binary_architecture(tmp_path: Path) -> None:
    import torch

    from ml.evaluation.phase_4c2g_model import (
        build_locked_validation_transform,
        load_stage1_checkpoint,
    )
    from ml.training.mobilenetv3_forensics import MobileNetV3Forensics

    source_model = MobileNetV3Forensics(
        num_classes=2,
        pretrained=False,
        freeze_backbone=True,
    )
    checkpoint_path = tmp_path / "best_checkpoint.pt"
    torch.save(
        {
            "epoch": 6,
            "model_state_dict": source_model.state_dict(),
            "metadata": {"stage": "frozen", "sample_size": 250, "seed": 42},
        },
        checkpoint_path,
    )

    loaded = load_stage1_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_sha256=_sha256(checkpoint_path),
        expected_seed=42,
        device="cpu",
    )

    assert isinstance(loaded.model, MobileNetV3Forensics)
    assert loaded.model.classifier[0].in_features == 576
    assert loaded.model.classifier[-1].out_features == 2
    assert all(not parameter.requires_grad for parameter in loaded.model.features.parameters())
    assert loaded.model.training is False
    with torch.no_grad():
        assert tuple(loaded.model(torch.zeros(1, 3, 224, 224)).shape) == (1, 2)

    transform = build_locked_validation_transform()
    assert "Resize(size=(224, 224), interpolation=bicubic" in repr(transform)
    assert "mean=[0.485, 0.456, 0.406]" in repr(transform)
    assert "std=[0.229, 0.224, 0.225]" in repr(transform)


def test_checkpoint_hash_failure_happens_before_torch_load(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest
    import torch

    from ml.evaluation.phase_4c2g_model import load_stage1_checkpoint

    checkpoint_path = tmp_path / "best_checkpoint.pt"
    checkpoint_path.write_bytes(b"tampered-checkpoint")
    torch_load_called = False

    def forbidden_torch_load(*args, **kwargs):
        nonlocal torch_load_called
        torch_load_called = True
        raise AssertionError("torch.load must not run before SHA-256 verification")

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        load_stage1_checkpoint(
            checkpoint_path=checkpoint_path,
            expected_sha256="0" * 64,
            expected_seed=42,
            device="cpu",
        )

    assert torch_load_called is False


def test_five_checkpoint_orchestration_uses_exact_registered_order(
    tmp_path: Path,
) -> None:
    from ml.evaluation.locked_test_evaluator import (
        AUTHORIZED_SEEDS,
        CHECKPOINT_SHA256_REGISTRY,
        EXPECTED_CHECKPOINT_BYTES,
        LockedTestEvaluator,
    )
    from ml.evaluation.run_phase_4c2g_confirmatory import (
        run_five_checkpoint_orchestration,
    )

    records = []
    for source_index in range(343):
        source_id = f"synthetic-{source_index:03d}"
        records.extend(
            [
                {
                    "source_id": source_id,
                    "sample_idx": source_index * 2,
                    "true_label": 0,
                    "logits": [2.0, -2.0],
                },
                {
                    "source_id": source_id,
                    "sample_idx": source_index * 2 + 1,
                    "true_label": 1,
                    "logits": [-2.0, 2.0],
                },
            ]
        )

    observed_order = []

    def synthetic_inference(seed, checkpoint_info, data_loader):
        observed_order.append(seed)
        return {
            "seed": seed,
            "predictions": records,
            "calibration_fitted": False,
            "threshold_tuned": False,
            "probability_ensemble": False,
        }

    evaluator = LockedTestEvaluator(
        output_dir=tmp_path / "outputs",
        effective_evaluator_commit="a" * 40,
    )
    checkpoint_infos = {
        seed: {
            "seed": seed,
            "sha256": CHECKPOINT_SHA256_REGISTRY[seed],
            "byte_count": EXPECTED_CHECKPOINT_BYTES,
        }
        for seed in reversed(AUTHORIZED_SEEDS)
    }

    receipts = run_five_checkpoint_orchestration(
        evaluator=evaluator,
        checkpoint_infos=checkpoint_infos,
        data_loader=object(),
        inference_fn=synthetic_inference,
        session_id="synthetic-session",
    )

    assert observed_order == AUTHORIZED_SEEDS
    assert [receipt["checkpoint_seed"] for receipt in receipts] == AUTHORIZED_SEEDS
    assert evaluator.ledger.evaluation_attempts == 5
    assert evaluator.ledger.completed_model_evaluations == 5
    assert sorted((tmp_path / "outputs").glob("predictions_seed_*.json"))
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["probability_ensemble"]
        is False
        for path in (tmp_path / "outputs").glob("predictions_seed_*.json")
    )


def test_confirmatory_finalization_writes_complete_atomic_output_set(
    tmp_path: Path,
) -> None:
    from ml.evaluation.locked_test_evaluator import (
        AUTHORIZED_SEEDS,
        CHECKPOINT_SHA256_REGISTRY,
        EXPECTED_CHECKPOINT_BYTES,
        LockedTestEvaluator,
    )
    from ml.evaluation.run_phase_4c2g_confirmatory import (
        finalize_confirmatory_outputs,
        run_five_checkpoint_orchestration,
    )

    records = []
    for source_index in range(343):
        source_id = f"synthetic-{source_index:03d}"
        for label in (0, 1):
            records.append(
                {
                    "source_id": source_id,
                    "sample_id": f"{source_id}-{label}",
                    "sample_idx": source_index * 2 + label,
                    "true_label": label,
                    "logits": [3.0, -3.0] if label == 0 else [-3.0, 3.0],
                }
            )

    output_dir = tmp_path / "outputs"
    evaluator = LockedTestEvaluator(
        output_dir=output_dir,
        effective_evaluator_commit="b" * 40,
    )
    checkpoint_infos = {
        seed: {
            "seed": seed,
            "sha256": CHECKPOINT_SHA256_REGISTRY[seed],
            "byte_count": EXPECTED_CHECKPOINT_BYTES,
        }
        for seed in AUTHORIZED_SEEDS
    }

    def synthetic_inference(seed, checkpoint_info, data_loader):
        return {
            "seed": seed,
            "predictions": records,
            "calibration_fitted": False,
            "threshold_tuned": False,
            "probability_ensemble": False,
        }

    receipts = run_five_checkpoint_orchestration(
        evaluator=evaluator,
        checkpoint_infos=checkpoint_infos,
        data_loader=object(),
        inference_fn=synthetic_inference,
        session_id="synthetic-finalization",
    )
    result = finalize_confirmatory_outputs(
        evaluator=evaluator,
        receipts=receipts,
        session_id="synthetic-finalization",
        manifest_binding={
            "manifest_sha256": "c" * 64,
            "schema_sha256": "d" * 64,
            "split_seal": "e" * 64,
        },
        environment={"fixture_only": True, "scientific_result": False},
        n_bootstrap_replicates=50,
    )

    required = {
        "per_checkpoint_metrics.json",
        "aggregate_confirmatory_metrics.json",
        "bootstrap_distribution_summary.json",
        "confirmatory_decision.json",
        "execution_receipt.json",
        "checksums.json",
        "environment.json",
    }
    assert required <= {path.name for path in output_dir.iterdir()}
    assert not list(output_dir.glob("*.part"))
    assert result["confirmatory_decision"]["verdict"] == "CONFIRMATORY_SUCCESS"
    assert result["aggregate"]["aggregation"] == "arithmetic_mean_of_5_macro_f1"
    assert result["aggregate"]["probability_ensemble"] is False
    assert result["aggregate"]["bootstrap_replicates"] == 50
    assert result["execution_receipt"]["completed_model_evaluations"] == 5
    assert result["execution_receipt"]["evaluation_attempts"] == 5

    checksums = json.loads((output_dir / "checksums.json").read_text(encoding="utf-8"))
    assert "checksums.json" not in checksums["files"]
    for name, metadata in checksums["files"].items():
        artifact = output_dir / name
        assert metadata["sha256"] == _sha256(artifact)
        assert metadata["bytes"] == artifact.stat().st_size


def test_synthetic_custodian_to_atomic_confirmatory_publication_end_to_end(
    tmp_path: Path,
) -> None:
    from ml.evaluation.confirmatory_metrics import (
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_RNG_SEED,
    )
    from ml.evaluation.locked_test_evaluator import (
        AUTHORIZED_SEEDS,
        CHECKPOINT_SHA256_REGISTRY,
        EXPECTED_CHECKPOINT_BYTES,
        LockedTestEvaluator,
    )
    from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest
    from ml.evaluation.run_phase_4c2g_confirmatory import (
        finalize_confirmatory_outputs,
        run_five_checkpoint_orchestration,
    )

    fixture = _build_production_custodian_evaluator_fixture(tmp_path)
    manifest_path, custodian_receipt = _seal_production_custodian_fixture(fixture)
    schema_path = REPO_ROOT / "docs/schemas/locked-test-manifest.v1.schema.json"
    loaded = load_locked_test_manifest(
        locked_test_root=fixture["root"],
        manifest_path=manifest_path,
        manifest_schema_path=schema_path,
        expected_manifest_sha256=custodian_receipt["manifest_sha256"],
        expected_schema_sha256=_sha256(schema_path),
        expected_split_seal=fixture["split_seal"],
    )

    output_dir = tmp_path / "synthetic-evaluator-output"
    evaluator = LockedTestEvaluator(
        output_dir=output_dir,
        effective_evaluator_commit="f" * 40,
    )
    session_id = "synthetic-cross-component-end-to-end"
    evaluator.ledger.record_event(
        event_type="PRE_READ_UNSEAL",
        session_id=session_id,
        metadata={"mode": "synthetic_fixture_only", "scientific_result": False},
        inc_unsealing_session=True,
    )
    checkpoint_infos = {
        seed: {
            "seed": seed,
            "sha256": CHECKPOINT_SHA256_REGISTRY[seed],
            "byte_count": EXPECTED_CHECKPOINT_BYTES,
        }
        for seed in AUTHORIZED_SEEDS
    }

    def fixture_inference(seed, checkpoint_info, data_loader):
        assert data_loader is loaded
        return {
            "seed": seed,
            "protocol": "stage1_frozen_backbone_linear_probe",
            "sample_size": 250,
            "predictions": [
                {
                    "source_id": sample.unique_source_id,
                    "sample_id": sample.sample_id,
                    "sample_idx": index,
                    "relative_path": sample.relative_path,
                    "true_label": sample.label_id,
                    "logits": (
                        [3.0, -3.0] if sample.label_id == 0 else [-3.0, 3.0]
                    ),
                }
                for index, sample in enumerate(loaded.samples)
            ],
            "calibration_fitted": False,
            "threshold_tuned": False,
            "probability_ensemble": False,
        }

    receipts = run_five_checkpoint_orchestration(
        evaluator=evaluator,
        checkpoint_infos=checkpoint_infos,
        data_loader=loaded,
        inference_fn=fixture_inference,
        session_id=session_id,
    )
    result = finalize_confirmatory_outputs(
        evaluator=evaluator,
        receipts=receipts,
        session_id=session_id,
        manifest_binding={
            "manifest_sha256": custodian_receipt["manifest_sha256"],
            "schema_sha256": _sha256(schema_path),
            "split_seal": fixture["split_seal"],
        },
        environment={
            "fixture_only": True,
            "scientific_result": False,
            "locked_test_real_accesses": 0,
            "completed_real_model_evaluations": 0,
        },
        n_bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )

    assert result["aggregate"]["bootstrap_replicates"] == 10000
    assert result["aggregate"]["bootstrap_rng_seed"] == BOOTSTRAP_RNG_SEED
    assert result["aggregate"]["bootstrap_generator"] == (
        "numpy.random.Generator(numpy.random.PCG64)"
    )
    assert result["execution_receipt"]["evaluation_attempts"] == 5
    assert result["execution_receipt"]["completed_model_evaluations"] == 5
    assert result["environment"]["scientific_result"] is False
    assert result["environment"]["locked_test_real_accesses"] == 0
    assert not list(output_dir.glob("*.part"))
    assert json.loads(
        (output_dir / "checksums.json").read_text(encoding="utf-8")
    )["files"]


def test_real_inference_function_preserves_sample_identity_and_logits(
    monkeypatch,
) -> None:
    import torch

    import ml.evaluation.run_phase_4c2g_confirmatory as driver

    class DeterministicModel(torch.nn.Module):
        def forward(self, inputs):
            positive = inputs[:, 0, 0, 0]
            return torch.column_stack((-positive, positive))

    class Loaded:
        model = DeterministicModel().eval()

    observed_loader_args = {}

    def fake_load_stage1_checkpoint(**kwargs):
        observed_loader_args.update(kwargs)
        return Loaded()

    monkeypatch.setattr(driver, "load_stage1_checkpoint", fake_load_stage1_checkpoint)
    inputs = torch.zeros(686, 3, 1, 1)
    inputs[:, 0, 0, 0] = torch.tensor(
        [-1.0 if index % 2 == 0 else 1.0 for index in range(686)]
    )
    labels = torch.tensor([index % 2 for index in range(686)])
    source_ids = [f"synthetic-{index // 2:03d}" for index in range(686)]
    sample_ids = [f"sample-{index:03d}" for index in range(686)]
    relative_paths = [f"samples/{sample_id}.png" for sample_id in sample_ids]
    data_loader = [(inputs, labels, source_ids, sample_ids, relative_paths)]

    inference_fn = driver.make_stage1_inference_function(device="cpu")
    payload = inference_fn(
        42,
        {
            "path": "fixture-checkpoint.pt",
            "sha256": "a" * 64,
            "byte_count": 123,
        },
        data_loader,
    )

    assert observed_loader_args["expected_seed"] == 42
    assert observed_loader_args["expected_sha256"] == "a" * 64
    assert observed_loader_args["expected_bytes"] == 123
    assert len(payload["predictions"]) == 686
    assert payload["predictions"][0] == {
        "source_id": "synthetic-000",
        "sample_id": "sample-000",
        "sample_idx": 0,
        "relative_path": "samples/sample-000.png",
        "true_label": 0,
        "logits": [1.0, -1.0],
    }
    assert payload["predictions"][1]["logits"] == [-1.0, 1.0]
    assert payload["probability_ensemble"] is False
    assert payload["threshold_tuned"] is False
    assert payload["calibration_fitted"] is False


def test_windows_runtime_receipts_require_fresh_session_and_os_read_only_proof(
    tmp_path: Path,
) -> None:
    import pytest

    from ml.evaluation.phase_4c2g_windows import (
        BLOCKED_READ_ONLY_REASON,
        verify_isolation_receipt,
        verify_windows_read_only_receipt,
    )

    now = datetime.now(timezone.utc)
    session_id = "future-authorized-session"
    locked_root = tmp_path / "locked-root"
    locked_root.mkdir()

    isolation_path = tmp_path / "isolation.json"
    isolation_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "verified_at_utc": now.isoformat(),
                "isolation_verified": True,
                "active_egress_adapters": [],
                "active_vpn_route_owners": [],
                "unidentified_route_owners": [],
                "proxy_enabled": False,
                "remaining_active_default_routes": 0,
            }
        ),
        encoding="utf-8",
    )
    assert verify_isolation_receipt(
        isolation_path,
        expected_session_id=session_id,
        now_utc=now,
    )["status"] == "FRESH_SESSION_ISOLATION_VERIFIED"

    stale = json.loads(isolation_path.read_text(encoding="utf-8"))
    stale["verified_at_utc"] = (now - timedelta(minutes=10)).isoformat()
    isolation_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale isolation receipt"):
        verify_isolation_receipt(
            isolation_path,
            expected_session_id=session_id,
            now_utc=now,
        )

    read_only_path = tmp_path / "read-only.json"
    read_only_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "verified_at_utc": now.isoformat(),
                "locked_test_root": str(locked_root.resolve()),
                "status": "READ_ONLY_VOLUME_VERIFIED",
                "os_enforced_read_only": True,
                "evidence_kind": "windows_disk_is_read_only",
                "backing_volume_unique_id": "fixture-volume-id",
                "backing_disk_number": 7,
                "canary_writes_performed": 0,
            }
        ),
        encoding="utf-8",
    )
    assert verify_windows_read_only_receipt(
        read_only_path,
        locked_test_root=locked_root,
        expected_session_id=session_id,
        now_utc=now,
    )["status"] == "READ_ONLY_VOLUME_VERIFIED"

    writable = json.loads(read_only_path.read_text(encoding="utf-8"))
    writable["os_enforced_read_only"] = False
    writable["evidence_kind"] = "folder_readonly_attribute"
    read_only_path.write_text(json.dumps(writable), encoding="utf-8")
    with pytest.raises(RuntimeError, match=BLOCKED_READ_ONLY_REASON):
        verify_windows_read_only_receipt(
            read_only_path,
            locked_test_root=locked_root,
            expected_session_id=session_id,
            now_utc=now,
        )


def test_complete_cli_dispatches_real_confirmatory_session(monkeypatch, tmp_path: Path) -> None:
    import ml.evaluation.run_phase_4c2g_confirmatory as driver

    observed = []

    def fake_execute(args):
        observed.append(args)
        return {"verdict": "fixture-complete"}

    monkeypatch.setattr(driver, "execute_confirmatory_session", fake_execute)
    required_paths = {
        "authorization-file": tmp_path / "authorization.json",
        "package-archive": tmp_path / "package.tar.gz",
        "execution-authorization-binding": tmp_path
        / "execution-authorization-binding.json",
        "source-binding": tmp_path / "source-binding.json",
        "authorization-schema": tmp_path / "authorization.schema.json",
        "checkpoint-dir": tmp_path / "checkpoints",
        "locked-test-root": tmp_path / "locked-test",
        "manifest": tmp_path / "manifest.json",
        "manifest-schema": tmp_path / "manifest.schema.json",
        "isolation-receipt": tmp_path / "isolation.json",
        "read-only-receipt": tmp_path / "read-only.json",
        "output-dir": tmp_path / "outputs",
    }
    argv = []
    for flag, path in required_paths.items():
        argv.extend([f"--{flag}", str(path)])
    argv.extend(["--session-id", "authorized-session-001", "--device", "cpu"])

    assert driver.main(argv) == 0
    assert len(observed) == 1
    assert observed[0].session_id == "authorized-session-001"
    assert observed[0].locked_test_root == required_paths["locked-test-root"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell controller required")
def test_windows_authorized_session_orchestrator_contract_validation_only() -> None:
    script = REPO_ROOT / "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ContractValidationOnly",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(completed.stdout)
    assert contract["verdict"] == "AUTHORIZED_SESSION_CONTRACT_VALID"
    assert contract["mutations_performed"] == 0
    assert contract["uac_or_readiness_invoked"] is False
    assert (
        contract["exact_archive_binding"]
        == "external_execution_authorization_binding"
    )
    assert contract["ordered_steps"] == [
        "verify_authorization_and_final_package",
        "create_network_recovery_watchdog",
        "isolate_active_egress_adapters",
        "verify_fresh_same_session_isolation_receipt",
        "verify_os_backed_read_only_storage",
        "run_exact_five_checkpoint_driver",
        "write_atomic_outputs",
        "restore_network",
        "delete_and_verify_watchdog",
        "seal_receipts_and_access_ledger",
    ]
    assert contract["read_only_policy"]["canary_writes_performed"] == 0
    assert contract["read_only_policy"]["folder_readonly_attribute_accepted"] is False
    assert (
        contract["read_only_policy"]["failure_code"]
        == "BLOCKED_LOCKED_TEST_STORAGE_NOT_PROVABLY_READ_ONLY"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell controller required")
def test_authorized_controller_enforces_object_pipeline_adapter_contract() -> None:
    script = REPO_ROOT / "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1"
    source = script.read_text(encoding="utf-8")

    assert not re.search(
        r"(?:Disable|Enable)-NetAdapter\s+-InterfaceIndex", source
    )
    assert "Get-NetAdapter -IncludeHidden" in source
    assert re.search(r"\$resolvedAdapter\s*\|\s*Disable-NetAdapter", source)
    assert re.search(r"\$resolvedAdapter\s*\|\s*Enable-NetAdapter", source)
    assert "Expected exactly 1 adapter" in source
    assert "Adapter identity mismatch" in source

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ContractValidationOnly",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(completed.stdout)
    adapter = contract["adapter_cmdlet_contract"]
    assert adapter["disable_input_object_value_from_pipeline"] is True
    assert adapter["enable_input_object_value_from_pipeline"] is True
    assert adapter["direct_interface_index_calls"] == 0
    assert adapter["include_hidden_resolution"] is True
    assert adapter["pipeline_whatif_binding_probe"] == "PASS"
    assert adapter["adapter_mutations"] == 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell controller required")
def test_authorized_controller_contract_exercises_recovery_identity_and_watchdog() -> None:
    script = REPO_ROOT / "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1"
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-ContractValidationOnly",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(completed.stdout)

    recovery = contract["generated_recovery_script_contract"]
    assert recovery["windows_powershell_5_1_ast_parse_errors"] == 0
    assert recovery["direct_interface_index_calls"] == 0
    assert recovery["pipeline_enable_calls"] >= 1
    assert recovery["fixture_execution"] == "PASS"
    assert recovery["missing_adapter_rejected"] is True
    assert recovery["duplicate_adapter_rejected"] is True
    assert recovery["identity_mismatch_rejected"] is True

    identity = contract["adapter_identity_fixture_contract"]
    assert identity == {
        "exact_match": "PASS",
        "missing_adapter_rejected": True,
        "duplicate_adapter_rejected": True,
        "name_mismatch_rejected": True,
        "description_mismatch_rejected": True,
        "mac_mismatch_rejected": True,
        "isolation_pipeline_exercised": True,
        "recovery_pipeline_exercised": True,
        "restoration_pipeline_exercised": True,
    }

    watchdog = contract["watchdog_readback_contract"]
    assert watchdog["bounded_polling"] is True
    assert watchdog["eventual_absence_fixture"] == "PASS"
    assert watchdog["timeout_while_present_blocks_pass"] is True
    assert watchdog["restoration_failure_retains_watchdog"] is True


def test_pre_and_post_reservation_failures_are_distinguished(tmp_path: Path) -> None:
    import pytest

    from ml.evaluation.locked_test_evaluator import (
        AUTHORIZED_SEEDS,
        CHECKPOINT_SHA256_REGISTRY,
        EXPECTED_CHECKPOINT_BYTES,
        LockedTestEvaluator,
    )
    from ml.evaluation.run_phase_4c2g_confirmatory import (
        run_five_checkpoint_orchestration,
    )

    evaluator = LockedTestEvaluator(
        output_dir=tmp_path / "outputs",
        effective_evaluator_commit="f" * 40,
    )
    incomplete = {
        seed: {
            "seed": seed,
            "sha256": CHECKPOINT_SHA256_REGISTRY[seed],
            "byte_count": EXPECTED_CHECKPOINT_BYTES,
        }
        for seed in AUTHORIZED_SEEDS[:-1]
    }
    with pytest.raises(ValueError, match="Checkpoint seed set mismatch"):
        run_five_checkpoint_orchestration(
            evaluator=evaluator,
            checkpoint_infos=incomplete,
            data_loader=object(),
            inference_fn=lambda *args: {},
            session_id="pre-reservation",
        )
    assert evaluator.ledger.evaluation_attempts == 0

    checkpoint = {
        "seed": 42,
        "sha256": CHECKPOINT_SHA256_REGISTRY[42],
        "byte_count": EXPECTED_CHECKPOINT_BYTES,
    }

    def crash_after_reservation(*args):
        raise RuntimeError("synthetic forward crash")

    with pytest.raises(RuntimeError, match="Automatic retry forbidden"):
        evaluator.execute_checkpoint_evaluation(
            seed=42,
            checkpoint_info=checkpoint,
            inference_fn=crash_after_reservation,
            data_loader=object(),
            session_id="post-reservation",
        )
    assert evaluator.ledger.evaluation_attempts == 1
    assert evaluator.ledger.completed_model_evaluations == 0
    assert evaluator.ledger.entries[-1]["event_type"] == "EVALUATION_ATTEMPT_INTERRUPTED"
    with pytest.raises(RuntimeError, match="Automatic retry forbidden"):
        evaluator.execute_checkpoint_evaluation(
            seed=42,
            checkpoint_info=checkpoint,
            inference_fn=lambda *args: {},
            data_loader=object(),
            session_id="silent-retry",
        )


def test_bootstrap_distribution_digest_is_reproducible() -> None:
    import numpy as np

    from ml.evaluation.confirmatory_metrics import compute_source_cluster_bootstrap_ci
    from ml.evaluation.locked_test_evaluator import AUTHORIZED_SEEDS

    source_ids = []
    y_true = []
    for source_index in range(343):
        source_ids.extend([f"source-{source_index:03d}"] * 2)
        y_true.extend([0, 1])
    labels = np.asarray(y_true, dtype=np.int64)
    logits = np.column_stack((1 - labels, labels)).astype(np.float64)
    predictions = {seed: logits.copy() for seed in AUTHORIZED_SEEDS}

    first = compute_source_cluster_bootstrap_ci(
        source_ids=source_ids,
        y_true=labels,
        predictions_per_checkpoint=predictions,
        n_replicates=100,
        seed=20261002,
    )
    second = compute_source_cluster_bootstrap_ci(
        source_ids=source_ids,
        y_true=labels,
        predictions_per_checkpoint=predictions,
        n_replicates=100,
        seed=20261002,
    )
    assert first["bootstrap_distribution_sha256_float64_le"] == second[
        "bootstrap_distribution_sha256_float64_le"
    ]
    assert first["bootstrap_rng_seed"] == 20261002
    assert first["verdict"] == "CONFIRMATORY_SUCCESS"


def test_package_auditor_rejects_links_and_binary_weights(tmp_path: Path) -> None:
    import pytest

    from scripts.research.build_phase4c2g_execution_package import audit_package

    linked_archive = tmp_path / "linked.tar.gz"
    with tarfile.open(linked_archive, "w:gz") as tar:
        member = tarfile.TarInfo("unsafe-link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        tar.addfile(member)
    with pytest.raises(ValueError, match="Non-regular TAR member"):
        audit_package(linked_archive)

    weights_archive = tmp_path / "weights.tar.gz"
    with tarfile.open(weights_archive, "w:gz") as tar:
        payload = b"not-real-weights"
        member = tarfile.TarInfo("checkpoints/best_checkpoint.pt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="binary forbidden"):
        audit_package(weights_archive)


def test_complete_package_is_deterministic_self_contained_and_data_free(
    tmp_path: Path,
) -> None:
    from scripts.research.build_phase4c2g_execution_package import (
        PACKAGE_SOURCE_MEMBERS,
        build_package,
    )

    source_binding = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.5/evaluator_source_binding.json"
    )
    binding_payload = json.loads(source_binding.read_text(encoding="utf-8"))
    effective_commit = binding_payload["effective_evaluator_commit"]
    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"
    first = build_package(
        repo_root=REPO_ROOT,
        output_path=first_path,
        effective_execution_commit=effective_commit,
        execution_package_commit=effective_commit,
        source_binding_path=source_binding,
    )
    second = build_package(
        repo_root=REPO_ROOT,
        output_path=second_path,
        effective_execution_commit=effective_commit,
        execution_package_commit=effective_commit,
        source_binding_path=source_binding,
    )
    assert first.sha256 == second.sha256
    assert first.bytes == second.bytes
    assert set(PACKAGE_SOURCE_MEMBERS) <= set(first.members)
    assert {"PACKAGE_MANIFEST.json", "runtime_source_binding.json"} <= set(
        first.members
    )
    assert all(
        not name.lower().endswith(
            (".pt", ".pth", ".ckpt", ".onnx", ".png", ".jpg", ".webp")
        )
        for name in first.members
    )

    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(first_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import ml.evaluation.run_phase_4c2g_confirmatory as d; "
                "import ml.evaluation.phase_4c2g_dataset; "
                "import ml.evaluation.phase_4c2g_model; "
                "assert callable(d.run_five_checkpoint_orchestration)"
            ),
        ],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_mixed_commit_package_contains_exact_preflightable_controller(
    tmp_path: Path,
) -> None:
    from scripts.research.build_phase4c2g_execution_package import build_package

    binding_path = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.10/evaluator_source_binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["effective_evaluator_commit"] == (
        "2bbb1109c8ab18c9ff7120ef004acd0ba7074716"
    )
    assert binding["orchestrator_hotfix_commit"] == (
        "597a79af3cc76707edeefcb6dfc1d93f5f0e5ae1"
    )
    assert binding["execution_package_commit"] == (
        "76fbfec84f8fce9b7afd92a266c0d3a7c2f6ca48"
    )

    archive_path = tmp_path / "phase_4c2g_complete_executor_76fbfec.tar.gz"
    audit = build_package(
        repo_root=REPO_ROOT,
        output_path=archive_path,
        effective_execution_commit=binding["execution_package_commit"],
        execution_package_commit=binding["execution_package_commit"],
        source_binding_path=binding_path,
    )
    assert audit.bytes == 52089
    assert audit.sha256 == (
        "2301238a1148ff0dd237132c1274c962148d601016fd59220cc876748883ea71"
    )

    extracted = tmp_path / "mixed-commit-package"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    controller = (
        extracted / "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1"
    )
    assert controller.stat().st_size == 37740
    assert _sha256(controller) == (
        "878456e7f71c7f8755ad6760d163afe199bfaff5fc793b0ffd0ff2a7359f3047"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(controller),
            "-ContractValidationOnly",
        ],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(completed.stdout)
    assert contract["verdict"] == "AUTHORIZED_SESSION_CONTRACT_VALID"
    assert contract["adapter_cmdlet_contract"]["direct_interface_index_calls"] == 0
    assert contract["adapter_cmdlet_contract"]["adapter_mutations"] == 0


def test_inventory_hotfix_package_contains_cross_component_compatible_loader(
    tmp_path: Path,
) -> None:
    from scripts.research.build_phase4c2g_execution_package import build_package

    binding_path = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.12/evaluator_source_binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_commit = "0658dce11a7790877ae1b0820645d2e5a7e9a710"
    assert binding["effective_evaluator_commit"] == expected_commit
    assert binding["execution_package_commit"] == expected_commit

    archive_path = tmp_path / "phase_4c2g_complete_executor_0658dce.tar.gz"
    audit = build_package(
        repo_root=REPO_ROOT,
        output_path=archive_path,
        effective_execution_commit=expected_commit,
        execution_package_commit=expected_commit,
        source_binding_path=binding_path,
    )
    assert audit.bytes == 52896
    assert audit.sha256 == (
        "9ea55d331bff1e0d4e5ef111621733a04926d913b48f0144e238b9aea3dad83b"
    )

    extracted = tmp_path / "inventory-hotfix-package"
    extracted.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extracted, filter="data")

    fixture_workspace = tmp_path / "packaged-loader-fixture"
    fixture_workspace.mkdir()
    fixture = _build_production_custodian_evaluator_fixture(fixture_workspace)
    manifest_path, receipt = _seal_production_custodian_fixture(fixture)
    schema_path = extracted / "docs/schemas/locked-test-manifest.v1.schema.json"
    probe = (
        "import hashlib,json,sys; "
        "from pathlib import Path; "
        "from ml.evaluation.phase_4c2g_dataset import load_locked_test_manifest; "
        "root,manifest,schema,manifest_sha,split=sys.argv[1:]; "
        "loaded=load_locked_test_manifest(locked_test_root=Path(root), "
        "manifest_path=Path(manifest), manifest_schema_path=Path(schema), "
        "expected_manifest_sha256=manifest_sha, "
        "expected_schema_sha256=hashlib.sha256(Path(schema).read_bytes()).hexdigest(), "
        "expected_split_seal=split); "
        "assert len(loaded.samples)==686; assert len(loaded.source_ids)==343"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(fixture["root"]),
            str(manifest_path),
            str(schema_path),
            receipt["manifest_sha256"],
            fixture["split_seal"],
        ],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    controller = (
        extracted / "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1"
    )
    contract_validation = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(controller),
            "-ContractValidationOnly",
        ],
        cwd=extracted,
        capture_output=True,
        text=True,
        check=False,
    )
    assert contract_validation.returncode == 0, contract_validation.stderr
    assert json.loads(contract_validation.stdout)["verdict"] == (
        "AUTHORIZED_SESSION_CONTRACT_VALID"
    )


def test_real_source_binding_matches_every_effective_commit_git_object() -> None:
    from scripts.research.build_phase4c2g_execution_package import (
        PACKAGE_SOURCE_MEMBERS,
        audit_git_commit_components,
    )

    binding_path = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.5/evaluator_source_binding.json"
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    components = dict(binding["components"])
    components["authorization_schema"] = binding["authorization_schema"]
    audit = audit_git_commit_components(
        repo_root=REPO_ROOT,
        commit=binding["effective_evaluator_commit"],
        components=components,
        require_worktree_match=False,
    )

    assert len(audit) == len(PACKAGE_SOURCE_MEMBERS)
    assert {item["relative_path"] for item in audit.values()} == set(
        PACKAGE_SOURCE_MEMBERS
    )
    assert all(item["bytes_match"] for item in audit.values())
    assert all(item["sha256_match"] for item in audit.values())


def test_authorization_v2_structure_defers_exact_values_to_source_binding(
    tmp_path: Path,
) -> None:
    from ml.evaluation.locked_test_evaluator import LockedTestEvaluator

    schema_path = (
        REPO_ROOT / "docs/schemas/human-unsealing-authorization.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = schema["properties"]
    effective_commit = "a" * 40
    component_names = properties["evaluator_component_hashes"]["required"]
    component_hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in component_names
    }
    source_binding = tmp_path / "synthetic_source_binding.json"
    source_binding.write_text(
        json.dumps(
            {
                "effective_evaluator_commit": effective_commit,
                "execution_package_commit": effective_commit,
                "authorization_schema": {
                    "relative_path": str(schema_path),
                    "bytes": schema_path.stat().st_size,
                    "sha256": _sha256(schema_path),
                },
                "authorization_component_names": component_names,
                "components": {
                    name: {"sha256": digest}
                    for name, digest in component_hashes.items()
                },
            }
        ),
        encoding="utf-8",
    )
    authorization = {
        "status": "AUTHORIZED",
        "authorization_id": "fixture-auth-v2-001",
        "authorized_by": "Synthetic Test Fixture",
        "authorized_at_utc": "2026-10-02T20:00:00Z",
        "candidate_protocol": "stage1_frozen_backbone_linear_probe",
        "sample_size": 250,
        "exact_seeds": properties["exact_seeds"]["const"],
        "checkpoint_sha256s": properties["checkpoint_sha256s"]["const"],
        "evaluator_effective_commit": effective_commit,
        "evaluator_component_hashes": component_hashes,
        "maximum_unsealing_sessions": 1,
        "maximum_model_evaluation_attempts": 5,
        "expiry_policy": {"policy": "single_session_only"},
        "authorization_purpose": "Synthetic schema validation only; no data access.",
        "no_tuning_acknowledgment": True,
        "execution_package_commit": effective_commit,
        "execution_package_tree_clean": True,
        "sealed_package_sha256": "a" * 64,
        "sealed_package_bytes": 1,
        "locked_test_manifest_sha256": "b" * 64,
        "locked_test_manifest_schema_sha256": properties[
            "locked_test_manifest_schema_sha256"
        ]["const"],
        "locked_test_split_seal": properties["locked_test_split_seal"]["const"],
    }
    fixture_path = tmp_path / "synthetic_authorization_fixture.json"
    fixture_path.write_text(json.dumps(authorization), encoding="utf-8")

    verified = LockedTestEvaluator.verify_human_authorization(
        fixture_path,
        schema_path=schema_path,
        source_binding_path=source_binding,
        current_time_utc=datetime(2026, 10, 2, 20, 1, tzinfo=timezone.utc),
        bypass_git_checks=True,
    )
    assert verified["status"] == "AUTHORIZED"
    assert verified["evaluator_effective_commit"] == effective_commit
    assert verified["evaluator_component_hashes"] == component_hashes


def test_effective_commit_audit_reads_exact_git_object_bytes() -> None:
    from scripts.research.build_phase4c2g_execution_package import (
        audit_git_commit_components,
    )

    binding = json.loads(
        (
            REPO_ROOT
            / "research/evidence/phase-4c.2g.0.5/evaluator_source_binding.json"
        ).read_text(encoding="utf-8")
    )
    component = binding["components"]["confirmatory_metrics"]

    audit = audit_git_commit_components(
        repo_root=REPO_ROOT,
        commit=binding["effective_evaluator_commit"],
        components={"confirmatory_metrics": component},
    )

    assert audit["confirmatory_metrics"]["exists_at_commit"] is True
    assert audit["confirmatory_metrics"]["bytes_match"] is True
    assert audit["confirmatory_metrics"]["sha256_match"] is True


def test_effective_commit_audit_detects_runtime_file_changed_after_commit() -> None:
    import pytest

    from scripts.research.build_phase4c2g_execution_package import (
        audit_git_commit_components,
    )

    superseded_commit = "5cf84a33641b7bc7a232fcd602b89671c63bb2ad"
    relative_path = "ml/evaluation/run_phase_4c2g_confirmatory.py"
    committed = subprocess.run(
        ["git", "show", f"{superseded_commit}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout

    with pytest.raises(ValueError, match="changed after effective commit"):
        audit_git_commit_components(
            repo_root=REPO_ROOT,
            commit=superseded_commit,
            components={
                "run_confirmatory_cli": {
                    "relative_path": relative_path,
                    "bytes": len(committed),
                    "sha256": hashlib.sha256(committed).hexdigest(),
                }
            },
            require_worktree_match=True,
        )


def test_external_binding_rejects_archive_bytes_commit_and_manifest_mutations(
    tmp_path: Path,
) -> None:
    import copy
    import pytest

    from ml.evaluation.run_phase_4c2g_confirmatory import (
        verify_execution_authorization_binding,
    )

    archive = tmp_path / "sealed-executor.tar.gz"
    archive.write_bytes(b"synthetic sealed executor")
    archive_sha = _sha256(archive)
    commit = "a" * 40
    manifest_sha = "b" * 64
    components = {
        "runtime": {
            "relative_path": "runtime.py",
            "bytes": 7,
            "sha256": "c" * 64,
        }
    }
    source_binding = {
        "effective_evaluator_commit": commit,
        "execution_package_commit": commit,
        "authorization_schema": {"bytes": 123, "sha256": "d" * 64},
        "authorization_component_names": ["runtime"],
        "components": components,
    }
    binding = {
        "schema_version": "1.0.0",
        "status": "ACTIVE",
        "effective_execution_commit": commit,
        "execution_package_commit": commit,
        "archive": {
            "filename": archive.name,
            "sha256": archive_sha,
            "bytes": archive.stat().st_size,
            "member_count": 25,
        },
        "authorization_schema": {"bytes": 123, "sha256": "d" * 64},
        "runtime_components": components,
        "locked_test_manifest_commitment": {
            "status": "COMMITTED_BEFORE_UNSEALING",
            "sha256": manifest_sha,
            "provenance": "synthetic data-custodian fixture",
        },
    }
    binding_path = tmp_path / "execution_authorization_binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    authorization = {
        "sealed_package_sha256": archive_sha,
        "sealed_package_bytes": archive.stat().st_size,
        "evaluator_effective_commit": commit,
        "execution_package_commit": commit,
        "locked_test_manifest_sha256": manifest_sha,
        "evaluator_component_hashes": {"runtime": "c" * 64},
    }

    verified = verify_execution_authorization_binding(
        authorization=authorization,
        binding_path=binding_path,
        package_archive=archive,
        source_binding=source_binding,
    )
    assert verified["status"] == "EXECUTION_AUTHORIZATION_BINDING_VERIFIED"

    mutations = (
        ("sealed_package_sha256", "e" * 64),
        ("sealed_package_bytes", archive.stat().st_size + 1),
        ("evaluator_effective_commit", "f" * 40),
        ("locked_test_manifest_sha256", "0" * 64),
    )
    for field, value in mutations:
        mutated = copy.deepcopy(authorization)
        mutated[field] = value
        with pytest.raises(PermissionError, match=field):
            verify_execution_authorization_binding(
                authorization=mutated,
                binding_path=binding_path,
                package_archive=archive,
                source_binding=source_binding,
            )


def test_external_binding_blocks_when_manifest_commitment_is_absent(
    tmp_path: Path,
) -> None:
    import pytest

    from ml.evaluation.run_phase_4c2g_confirmatory import (
        verify_execution_authorization_binding,
    )

    archive = tmp_path / "sealed-executor.tar.gz"
    archive.write_bytes(b"synthetic sealed executor")
    binding_path = tmp_path / "execution_authorization_binding.json"
    binding_path.write_text(
        json.dumps(
            {
                "status": "BLOCKED_MANIFEST_COMMITMENT_ABSENT",
                "locked_test_manifest_commitment": {
                    "status": "ABSENT",
                    "sha256": None,
                    "required_action": "Independent data-custodian commitment required.",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PermissionError, match="BLOCKED_MANIFEST_COMMITMENT_ABSENT"):
        verify_execution_authorization_binding(
            authorization={},
            binding_path=binding_path,
            package_archive=archive,
            source_binding={},
        )
