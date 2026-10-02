"""Synthetic and fault-injection coverage for Phase 4C.2G confirmatory execution.

This suite must never read the real locked-test partition, run scientific
inference, create an authorization artifact, or invoke the UAC/readiness path.
"""

from __future__ import annotations

import json
import hashlib
import subprocess
import io
import tarfile
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


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

    effective_commit = "5cf84a33641b7bc7a232fcd602b89671c63bb2ad"
    source_binding = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.5/evaluator_source_binding.json"
    )
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


def test_authorization_v2_locks_effective_commit_and_exact_components(
    tmp_path: Path,
) -> None:
    from ml.evaluation.locked_test_evaluator import LockedTestEvaluator

    schema_path = (
        REPO_ROOT / "docs/schemas/human-unsealing-authorization.v2.schema.json"
    )
    source_binding = (
        REPO_ROOT
        / "research/evidence/phase-4c.2g.0.5/evaluator_source_binding.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = schema["properties"]
    authorization = {
        "status": "AUTHORIZED",
        "authorization_id": "fixture-auth-v2-001",
        "authorized_by": "Synthetic Test Fixture",
        "authorized_at_utc": "2026-10-02T20:00:00Z",
        "candidate_protocol": "stage1_frozen_backbone_linear_probe",
        "sample_size": 250,
        "exact_seeds": properties["exact_seeds"]["const"],
        "checkpoint_sha256s": properties["checkpoint_sha256s"]["const"],
        "evaluator_effective_commit": properties["evaluator_effective_commit"][
            "const"
        ],
        "evaluator_component_hashes": properties["evaluator_component_hashes"][
            "const"
        ],
        "maximum_unsealing_sessions": 1,
        "maximum_model_evaluation_attempts": 5,
        "expiry_policy": {"policy": "single_session_only"},
        "authorization_purpose": "Synthetic schema validation only; no data access.",
        "no_tuning_acknowledgment": True,
        "execution_package_commit": properties["execution_package_commit"]["const"],
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
    assert (
        verified["evaluator_effective_commit"]
        == "5cf84a33641b7bc7a232fcd602b89671c63bb2ad"
    )
    assert set(verified["evaluator_component_hashes"]) == set(
        json.loads(source_binding.read_text(encoding="utf-8"))[
            "authorization_component_names"
        ]
    )
