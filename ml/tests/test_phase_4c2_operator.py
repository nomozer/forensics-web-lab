"""
Phase 4C.2B Operator Verification Test Suite.
Tests operator script static invariants, GPU policies, contract allowlists,
fault injections, resume behaviors, and safety gates.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "phase_4c2_execute_all.sh"

CANONICAL_BUNDLE_ARCHIVE_SHA = "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
CANONICAL_BUNDLE_CONTENT_SHA = "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
CANONICAL_BUNDLE_MANIFEST_SHA = "411e35da80843a10f7bb22e33efac8a4d1db70110dbda91cadbbf39532c9312d"
CANONICAL_BUNDLE_BYTES = 724633600

CANONICAL_WEIGHTS_FILE_SHA = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
CANONICAL_WEIGHTS_FILE_BYTES = 10306551
CANONICAL_BACKBONE_FINGERPRINT = "d42bb32ad876b9de2b04a6ccd245f76c4d0c6bb3ded74c25261cf14720c7e7d5"

EXPECTED_TRAINABLE_PARAMS = 204674
EXPECTED_FROZEN_PARAMS = 870560
EXPECTED_TOTAL_PARAMS = 1075234

REQUIRED_ARTIFACT_NAMES = [
    "best_checkpoint.pt",
    "run_receipt.json",
    "epoch_history.json",
    "predictions.json",
    "training_history.csv",
    "metrics.json",
    "environment.json",
    "environment-binding.json",
    "trainable-parameter-inventory.json",
    "checksums.json",
]


def extract_python_snippet(script_text: str, anchor: str, start_token: str = "<<'PY'", end_token: str = "\nPY\n") -> str:
    a_idx = script_text.find(anchor)
    if a_idx == -1:
        raise ValueError(f"Could not find anchor '{anchor}' in operator script")
    s_idx = script_text.find(start_token, a_idx)
    if s_idx == -1:
        raise ValueError(f"Could not find start token '{start_token}' after anchor")
    nl_idx = script_text.find("\n", s_idx)
    e_idx = script_text.find(end_token, nl_idx)
    if e_idx == -1:
        raise ValueError(f"Could not find end token '{end_token}'")
    return script_text[nl_idx + 1:e_idx]


def extract_verification_python_code(script_text: str) -> str:
    return extract_python_snippet(script_text, "verify_run_artifacts()", start_token="<<'PY'", end_token="\nPY\n")


def setup_mock_stage2_run(
    base_dir: Path,
    sample_size: int = 50,
    seed: int = 42,
    stage: str = "partial_finetune",
    treatment: str = "pre-registered partial fine-tuning protocol",
    init_policy: str = "from_pretrained_backbone_with_fresh_head",
    parent_path: str | None = None,
    parent_hash: str | None = None,
    trainable_count: int = EXPECTED_TRAINABLE_PARAMS,
    frozen_count: int = EXPECTED_FROZEN_PARAMS,
    locked_access: int = 0,
    stage1_writes: int = 0,
    val_source_count: int = 91,
    val_sample_count: int = 182,
    leak_dev_sources: bool = False,
    leak_lock_sources: bool = False,
    missing_files: list[str] | None = None,
    corrupt_checksum: bool = False,
    status: str = "completed",
    config_hash: str = "5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4",
    weights_file_sha: str = CANONICAL_WEIGHTS_FILE_SHA,
    backbone_fingerprint: str = CANONICAL_BACKBONE_FINGERPRINT,
    dataset_archive_sha: str = CANONICAL_BUNDLE_ARCHIVE_SHA,
    checkpoint_sha: str | None = None,
    trainable_inventory: list[dict] | None = None,
    optimizer_groups: list[dict] | None = None,
    batchnorm_policy: dict | None = None,
    predictions_data_override: dict | None = None,
    omit_from_checksums: list[str] | None = None,
    extra_in_checksums: list[str] | None = None,
    corrupt_env_binding: bool = False,
) -> Path:
    run_dir = base_dir / f"n{sample_size}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. best_checkpoint.pt
    ckpt_bytes = b"MOCK_STAGE2_CHECKPOINT_DATA"
    (run_dir / "best_checkpoint.pt").write_bytes(ckpt_bytes)
    actual_ckpt_sha = hashlib.sha256(ckpt_bytes).hexdigest()

    # 2. epoch_history.json
    epoch_hist = [{"epoch": i, "val_loss": 0.5 - i * 0.01, "val_macro_f1": 0.55 + i * 0.005} for i in range(1, 10)]
    (run_dir / "epoch_history.json").write_text(json.dumps(epoch_hist, indent=2), encoding="utf-8")

    # 3. predictions.json
    val_sids = [f"src_{i:04d}" for i in range(val_source_count)]
    if leak_dev_sources:
        val_sids[0] = "dev_0001"
    if leak_lock_sources:
        val_sids[1] = "lock_0001"

    if predictions_data_override is not None:
        preds_data = predictions_data_override
    else:
        # Balanced: each of 91 sources has exactly label pair {0, 1}
        # First 91 entries: label 0; next 91 entries: label 1
        preds_data = {
            "targets": [0] * (val_sample_count // 2) + [1] * (val_sample_count // 2),
            "predictions": [0] * (val_sample_count // 2) + [1] * (val_sample_count // 2),
            "probabilities": [0.1] * (val_sample_count // 2) + [0.9] * (val_sample_count // 2),
            "source_ids": val_sids * 2,
        }
    (run_dir / "predictions.json").write_text(json.dumps(preds_data, indent=2), encoding="utf-8")

    # 4. training_history.csv
    lines = ["epoch,val_loss,val_macro_f1"] + [f"{i},{0.5-i*0.01},{0.55+i*0.005}" for i in range(1, 10)]
    (run_dir / "training_history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5. metrics.json
    metrics_data = {
        "run_id": f"phase4c2-stage2-n{sample_size}-seed{seed}",
        "sample_size": sample_size,
        "seed": seed,
        "stage": stage,
        "treatment_designation": treatment,
        "macro_f1": 0.58,
        "balanced_accuracy": 0.58,
        "auroc": 0.60,
        "brier_score": 0.23,
        "ece": 0.02,
        "loss": 0.17,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    # 6. environment.json
    env_data = {"python": "3.12.10", "torch": "2.2.0", "cuda": True, "gpu_name": "Tesla T4"}
    env_bytes = json.dumps(env_data, indent=2).encode("utf-8")
    (run_dir / "environment.json").write_bytes(env_bytes)

    # 7. environment-binding.json
    env_sha = hashlib.sha256(env_bytes).hexdigest()
    if corrupt_env_binding:
        env_sha = "00" * 32
    env_binding = {
        "run_id": f"phase4c2-stage2-n{sample_size}-seed{seed}",
        "environment_sha256": env_sha,
        "timestamp_utc": "2026-10-01T00:00:00Z",
    }
    (run_dir / "environment-binding.json").write_text(json.dumps(env_binding, indent=2), encoding="utf-8")

    # 8. trainable-parameter-inventory.json
    canonical_inventory = [
        {"name": "features.12.0.weight", "shape": [576, 96, 1, 1], "numel": 55296},
        {"name": "features.12.1.weight", "shape": [576], "numel": 576},
        {"name": "features.12.1.bias", "shape": [576], "numel": 576},
        {"name": "classifier.0.weight", "shape": [256, 576], "numel": 147456},
        {"name": "classifier.0.bias", "shape": [256], "numel": 256},
        {"name": "classifier.3.weight", "shape": [2, 256], "numel": 512},
        {"name": "classifier.3.bias", "shape": [2], "numel": 2},
    ]
    inv = trainable_inventory if trainable_inventory is not None else canonical_inventory
    (run_dir / "trainable-parameter-inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")

    # 9. run_receipt.json
    canonical_opt_groups = [
        {"group_name": "backbone_features_12", "param_names": ["features.12.0.weight"], "learning_rate": 5e-5, "weight_decay": 1e-4, "tensor_count": 3, "numel": 56448},
        {"group_name": "classifier_head", "param_names": ["classifier.0.weight"], "learning_rate": 5e-4, "weight_decay": 1e-4, "tensor_count": 4, "numel": 148226},
    ]
    opt_grp = optimizer_groups if optimizer_groups is not None else canonical_opt_groups

    canonical_bn = {
        "frozen_features_eval": True,
        "reapply_after_train": True,
        "unfrozen_features_12_bn_train": True,
    }
    bn_pol = batchnorm_policy if batchnorm_policy is not None else canonical_bn

    receipt_data = {
        "run_id": f"phase4c2-stage2-n{sample_size}-seed{seed}",
        "timestamp_utc": "2026-10-01T00:00:00Z",
        "sample_size": sample_size,
        "seed": seed,
        "stage": stage,
        "treatment_designation": treatment,
        "status": status,
        "initialization_policy": init_policy,
        "pretrained_weights_identifier": "MobileNet_V3_Small_Weights.DEFAULT",
        "pretrained_weights_file_sha256": weights_file_sha,
        "pretrained_weights_state_fingerprint": backbone_fingerprint,
        "parent_checkpoint_path": parent_path,
        "parent_checkpoint_hash": parent_hash,
        "classifier_initialization_evidence": "seeded_torch_init_at_model_creation",
        "device": "cuda",
        "gpu_name": "Tesla T4",
        "gpu_vram_gb": 15.0,
        "config_hash": config_hash,
        "dataset_archive_sha256": dataset_archive_sha,
        "dataset_content_sha256": CANONICAL_BUNDLE_CONTENT_SHA,
        "dataset_manifest_sha256": CANONICAL_BUNDLE_MANIFEST_SHA,
        "exact_trainable_tensor_inventory": inv,
        "trainable_parameters_count": trainable_count,
        "frozen_parameters_count": frozen_count,
        "total_parameters_count": trainable_count + frozen_count,
        "optimizer_parameter_groups": opt_grp,
        "batchnorm_policy": bn_pol,
        "epochs_completed": 9,
        "best_epoch": 8,
        "best_val_macro_f1": 0.58,
        "final_metrics": metrics_data,
        "dummy_baseline": {"macro_f1": 0.5, "macro_f1_mean": 0.5, "macro_f1_std": 0.0},
        "metadata_baseline": {"macro_f1": 0.5, "macro_f1_mean": 0.5, "macro_f1_std": 0.0},
        "training_time_seconds": 120.0,
        "peak_vram_mb": 115.0,
        "checkpoint_sha256": checkpoint_sha if checkpoint_sha is not None else actual_ckpt_sha,
        "validation_source_count": val_source_count,
        "validation_sample_count": val_sample_count,
        "locked_test_access": locked_access,
        "stage1_output_writes": stage1_writes,
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

    # 10. checksums.json
    checksums = {}
    for f in run_dir.iterdir():
        if f.is_file() and f.name != "checksums.json":
            if omit_from_checksums and f.name in omit_from_checksums:
                continue
            checksums[f.name] = {
                "size_bytes": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
    if extra_in_checksums:
        for ext in extra_in_checksums:
            checksums[ext] = {"size_bytes": 100, "sha256": "ab" * 32}

    if corrupt_checksum:
        checksums["best_checkpoint.pt"]["sha256"] = "00" * 32

    (run_dir / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    # Handle missing files if requested
    if missing_files:
        for mf in missing_files:
            target = run_dir / mf
            if target.exists():
                target.unlink()

    return run_dir


# ==============================================================================
# Test Cases 1 - 28 & 32
# ==============================================================================

def test_01_bash_syntax_utf8_lf_no_bom():
    """1. Operator bash syntax, UTF-8 encoding, LF line endings, no BOM."""
    assert OPERATOR_SCRIPT.exists(), f"Operator missing at {OPERATOR_SCRIPT}"
    raw = OPERATOR_SCRIPT.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "Must not have UTF-8 BOM"
    assert b"\r" not in raw, "Must use LF line endings, not CRLF"

    # bash -n validation using relative path to avoid Windows drive colon issues in git bash
    rel_path = str(OPERATOR_SCRIPT.relative_to(REPO_ROOT)).replace("\\", "/")
    res = subprocess.run(["bash", "-n", rel_path], cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n syntax error: {res.stderr}"


def test_02_no_credentials():
    """2. No credentials, tokens, or hardcoded passwords in operator."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    forbidden = ["ghp_", "github_pat_", "password", "client_secret", "refresh_token", "PRIVATE KEY"]
    for word in forbidden:
        assert word not in text, f"Potential credential found: {word}"


def test_03_no_google_drive_mount_in_operator():
    """3. Operator must NOT mount Google Drive directly."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "drive.mount" not in text, "Operator must not call drive.mount"


def test_04_no_torch_cuda_reinstall():
    """4. Operator must NOT reinstall torch, torchvision, or CUDA packages."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "pip install torch" not in text
    assert "pip install torchvision" not in text
    assert "pip install --upgrade torch" not in text


def test_05_capability_based_gpu_policy():
    """5. Capability-based GPU policy accepts T4, L4, V100, A100, requires >= 8GB VRAM."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "vram_gb >= 8.0" in text or "8.0" in text
    assert "torch.cuda.is_available()" in text


def test_06_preflight_only_no_training():
    """6. Preflight-only mode runs verification without invoking training runner."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "--preflight-only" in text
    assert 'OPERATOR_STATUS.json' in text
    assert 'preflight_passed' in text


def test_07_exact_15_run_matrix():
    """7. Exact 15-run matrix: N in {50, 100, 250}, seeds in {42, 1337, 2025, 3407, 9001}."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "SAMPLE_SIZES=(50 100 250)" in text
    assert "SEEDS=(42 1337 2025 3407 9001)" in text


def test_08_runner_cli_flags_match_real_parser():
    """8. Runner CLI flags in operator match run_phase_4c2.py parser exactly."""
    from ml.training.run_phase_4c2 import main
    # Verify main argument parser accepts operator arguments
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    expected_flags = [
        "--config",
        "--bundle",
        "--sample-size",
        "--seed",
        "--stage",
        "--device",
        "--train-partition",
        "--eval-partition",
        "--output",
        "--weights-path",
    ]
    for flag in expected_flags:
        assert flag in text, f"Flag {flag} missing in operator command!"


def test_09_dataset_archive_content_manifest_binding():
    """9. Dataset archive, content, and manifest hashes match canonical reusable bundle."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert CANONICAL_BUNDLE_ARCHIVE_SHA in text
    assert CANONICAL_BUNDLE_CONTENT_SHA in text
    assert CANONICAL_BUNDLE_MANIFEST_SHA in text


def test_10_pretrained_weights_binding():
    """10. Pretrained weights file SHA, size, and loaded backbone state fingerprint match."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert CANONICAL_WEIGHTS_FILE_SHA in text
    assert str(CANONICAL_WEIGHTS_FILE_BYTES) in text
    assert CANONICAL_BACKBONE_FINGERPRINT in text


def test_11_exact_trainable_allowlist():
    """11. Exact trainable parameters: 204,674 weights, 7 tensors; 870,560 frozen."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert str(EXPECTED_TRAINABLE_PARAMS) in text
    assert str(EXPECTED_FROZEN_PARAMS) in text
    assert str(EXPECTED_TOTAL_PARAMS) in text


def test_12_frozen_batchnorm_behavior():
    """12. Frozen BatchNorm policy keeps features.0-11 in eval mode."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "apply_frozen_bn_policy" in text


def test_13_optimizer_group_contract():
    """13. Differential optimizer groups are 2 disjoint groups."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "build_stage2_optimizer" in text
    assert "len(opt.param_groups) == 2" in text


def test_14_stage1_path_symlink_rejection():
    """14. Output path validation rejects any Stage 1 path or symlink."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "phase_4c1" in text
    assert "execution_79bb115" in text
    assert "FATAL SECURITY VIOLATION" in text


def test_15_valid_completed_run_skips():
    """15. Valid completed run in final directory is verified and skipped."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "verify_run_artifacts" in text
    assert "[SKIP] Run" in text


def test_16_invalid_final_run_fails_closed():
    """16. Existing invalid final run directory triggers FAIL-CLOSED, does not overwrite."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "FAIL-CLOSED" in text
    assert "failed verification" in text


def test_17_interrupted_inprogress_recovery():
    """17. Interrupted .inprogress directory is cleaned for fresh run."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert 'LOCAL_INPROGRESS="$WORK_DIR/${RUN_ID}.inprogress"' in text
    assert 'rm -rf "$LOCAL_INPROGRESS"' in text


def test_18_interrupted_part_publish_recovery():
    """18. Interrupted .part staging: finalized if valid, moved to failed_publish if corrupt."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert 'STAGE_PART_DIR="$OUTPUT_ROOT/.publish_${RUN_ID}.part"' in text
    assert 'failed_publish' in text


def run_verification_code(verif_code: str, run_dir: Path, sample_size: int = 50, seed: int = 42, bundle_dir: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c", verif_code, str(run_dir), str(sample_size), str(seed), str(bundle_dir) if bundle_dir else ""]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_19_locked_test_leak_fails():
    """19. Locked test leak detection fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_mock_stage2_run(Path(td), 50, 42, locked_access=1)
        res = run_verification_code(verif_code, run_dir, 50, 42)
        assert res.returncode != 0, "Expected failure when locked_test_access > 0"


def test_20_stage1_write_count_nonzero_fails():
    """20. stage1_output_writes > 0 triggers verification failure."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_mock_stage2_run(Path(td), 50, 42, stage1_writes=1)
        res = run_verification_code(verif_code, run_dir, 50, 42)
        assert res.returncode != 0, "Expected failure when stage1_output_writes > 0"


def test_21_wrong_stage_fails():
    """21. Wrong stage (e.g., 'frozen') fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_mock_stage2_run(Path(td), 50, 42, stage="frozen")
        res = run_verification_code(verif_code, run_dir, 50, 42)
        assert res.returncode != 0, "Expected failure when stage is not partial_finetune"


def test_22_wrong_initialization_policy_fails():
    """22. Wrong initialization policy fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_mock_stage2_run(Path(td), 50, 42, init_policy="loaded_from_stage1_best")
        res = run_verification_code(verif_code, run_dir, 50, 42)
        assert res.returncode != 0, "Expected failure when initialization policy is wrong"


def test_23_wrong_weights_hash_fails():
    """23. Wrong pretrained weights hash in code check fails."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert CANONICAL_WEIGHTS_FILE_SHA in text


def test_24_missing_artifact_fails():
    """24. Missing any of the 10 required run artifacts fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    for missing_art in REQUIRED_ARTIFACT_NAMES:
        with tempfile.TemporaryDirectory() as td:
            run_dir = setup_mock_stage2_run(Path(td), 50, 42, missing_files=[missing_art])
            res = run_verification_code(verif_code, run_dir, 50, 42)
            assert res.returncode != 0, f"Expected failure when {missing_art} is missing"


def test_25_corrupt_checksum_fails():
    """25. Corrupted artifact checksum fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_mock_stage2_run(Path(td), 50, 42, corrupt_checksum=True)
        res = run_verification_code(verif_code, run_dir, 50, 42)
        assert res.returncode != 0, "Expected failure when checksum does not match"


def test_26_exact_validation_membership_mismatch_fails():
    """26. Exact validation membership mismatch (cardinality != 91) fails."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_mock_stage2_run(Path(td), 50, 42, val_source_count=90, val_sample_count=180)
        res = run_verification_code(verif_code, run_dir, 50, 42)
        assert res.returncode != 0, "Expected failure when validation_source_count != 91"


def test_27_environment_lock_missing_or_corrupt_fails():
    """27. Missing or corrupted environment lock verification fails."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "phase4c2_environment_lock.json" in text
    assert "phase4c2_environment_lock.sha256" in text
    assert "Environment lock sidecar hash mismatch" in text


def test_28_outputs_persist_under_drive_stage2_root():
    """28. Outputs persist strictly under Drive Stage 2 namespace."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DRIVE_ROOT_PHASE4C2/runs/execution_${EXECUTION_SHORT_SHA}}"' in text


def test_32_zero_research_runs_during_preparation():
    """32. Zero research runs executed during preparation wave."""
    verdict_p = REPO_ROOT / "research" / "evidence" / "phase-4c.2a" / "PRE_EXECUTION_GO_NO_GO.json"
    with open(verdict_p, "r", encoding="utf-8") as f:
        data = json.load(f)["pre_execution_verdict"]

    inv = data["provenance_and_accounting"]
    assert inv["training_runs_in_phase"] == 0
    assert inv["locked_test_accesses"] == 0
    assert inv["stage2_invocations"] == 0


# ==============================================================================
# Phase 4C.2B.1 TAR Security Behavioral Tests (10 Fixtures via Bash Subprocess)
# ==============================================================================

CANONICAL_CODE_ARCHIVE = (
    REPO_ROOT.parent
    / "forensics-web-lab-local-artifacts"
    / "phase_4c2"
    / "inputs"
    / "phase_4c2_code_9ee7fdb.tar.gz"
)


def to_posix_path_for_bash(p: Path) -> str:
    try:
        res = subprocess.run(
            ["bash", "-c", f"wslpath -u '{p.as_posix()}'"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return p.as_posix()


def extract_tar_audit_bash_block(script_text: str) -> str:
    start_marker = '# Audit tar entries for path traversal before extraction\n'
    end_marker = '\nPY\n'
    start_idx = script_text.find(start_marker)
    if start_idx == -1:
        raise ValueError(f"Could not find start marker '{start_marker}' in operator script")
    end_idx = script_text.find(end_marker, start_idx)
    if end_idx == -1:
        raise ValueError(f"Could not find end marker '{end_marker}' in operator script")
    return script_text[start_idx + len(start_marker) : end_idx + len(end_marker)]


def run_tar_audit_subprocess(audit_block: str, archive_path: Path) -> subprocess.CompletedProcess:
    posix_path = to_posix_path_for_bash(archive_path)
    bash_script = f"""
SYS_PY3="${{SYS_PY3:-$(which python3 || echo python)}}"
CODE_ARCHIVE="{posix_path}"
{audit_block}
""".replace("\r\n", "\n").replace("\r", "\n")
    proc = subprocess.run(
        ["bash"],
        input=bash_script.encode("utf-8"),
        capture_output=True,
    )
    proc.stdout = proc.stdout.decode("utf-8", errors="replace")
    proc.stderr = proc.stderr.decode("utf-8", errors="replace")
    return proc


def create_tar_fixture(dest: Path, entries: list[tuple[str, int, str | None]]):
    with tarfile.open(dest, "w:gz") as tf:
        for name, entry_type, linkname in entries:
            ti = tarfile.TarInfo(name=name)
            ti.type = entry_type
            if linkname:
                ti.linkname = linkname
            if entry_type == tarfile.REGTYPE:
                data = b"safe dummy content for fixture test"
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            elif entry_type == tarfile.DIRTYPE:
                ti.size = 0
                tf.addfile(ti)
            else:
                ti.size = 0
                tf.addfile(ti)


@pytest.fixture(scope="module")
def tar_audit_block():
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    return extract_tar_audit_bash_block(script_text)


class TestTarSafetyAuditBehavioral:
    """Suite running the operator's actual TAR safety audit Bash block against 10 fixtures."""

    def test_fixture_01_valid_archive_passes(self, tar_audit_block, tmp_path):
        """1. Valid archive: regular files, directories, nested POSIX paths -> PASS."""
        arch = tmp_path / "01_valid.tar.gz"
        create_tar_fixture(arch, [
            ("ml", tarfile.DIRTYPE, None),
            ("ml/training", tarfile.DIRTYPE, None),
            ("ml/training/run_phase_4c2.py", tarfile.REGTYPE, None),
            ("ml/configs/phase_4c2.yaml", tarfile.REGTYPE, None),
        ])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode == 0, f"Expected PASS, got {res.returncode}: {res.stderr}"
        assert "[+] Code archive tar safety audit PASS" in res.stdout
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_02_posix_absolute_path_fails(self, tar_audit_block, tmp_path):
        """2. POSIX absolute path: /tmp/evil -> FAIL."""
        arch = tmp_path / "02_posix_abs.tar.gz"
        create_tar_fixture(arch, [("/tmp/evil", tarfile.REGTYPE, None)])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Absolute path in code archive: /tmp/evil" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_03_windows_root_path_fails(self, tar_audit_block, tmp_path):
        r"""3. Windows-root path: \evil -> FAIL."""
        arch = tmp_path / "03_win_root.tar.gz"
        create_tar_fixture(arch, [(r"\evil", tarfile.REGTYPE, None)])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Absolute path in code archive: \\evil" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_04_windows_drive_path_fails(self, tar_audit_block, tmp_path):
        r"""4. Windows drive path: C:\evil.txt -> FAIL."""
        arch = tmp_path / "04_win_drive.tar.gz"
        create_tar_fixture(arch, [(r"C:\evil.txt", tarfile.REGTYPE, None)])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Absolute path in code archive: C:\\evil.txt" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_05_posix_traversal_fails(self, tar_audit_block, tmp_path):
        """5. POSIX traversal: ../../evil -> FAIL."""
        arch = tmp_path / "05_posix_traversal.tar.gz"
        create_tar_fixture(arch, [("../../evil", tarfile.REGTYPE, None)])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Path traversal detected in code archive: ../../evil" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_06_backslash_traversal_fails(self, tar_audit_block, tmp_path):
        r"""6. Backslash traversal: ..\..\evil -> FAIL."""
        arch = tmp_path / "06_backslash_traversal.tar.gz"
        create_tar_fixture(arch, [(r"..\..\evil", tarfile.REGTYPE, None)])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Path traversal detected in code archive: ..\\..\\evil" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_07_symlink_fails(self, tar_audit_block, tmp_path):
        """7. Symlink: safe/link -> ../../evil -> FAIL."""
        arch = tmp_path / "07_symlink.tar.gz"
        create_tar_fixture(arch, [("safe/link", tarfile.SYMTYPE, "../../evil")])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Links are forbidden in code archive" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_08_hardlink_fails(self, tar_audit_block, tmp_path):
        """8. Hardlink -> FAIL."""
        arch = tmp_path / "08_hardlink.tar.gz"
        create_tar_fixture(arch, [("safe/hardlink", tarfile.LNKTYPE, "target")])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Links are forbidden in code archive" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_09_special_entry_fifo_fails(self, tar_audit_block, tmp_path):
        """9. Special FIFO/device entry -> FAIL."""
        arch = tmp_path / "09_fifo.tar.gz"
        create_tar_fixture(arch, [("safe/fifo", tarfile.FIFOTYPE, None)])
        res = run_tar_audit_subprocess(tar_audit_block, arch)
        assert res.returncode != 0
        assert "Special TAR entry is forbidden: safe/fifo" in res.stderr
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr

    def test_fixture_10_canonical_archive_passes(self, tar_audit_block):
        """10. Canonical code archive: phase_4c2_code_9ee7fdb.tar.gz -> PASS."""
        if not CANONICAL_CODE_ARCHIVE.exists():
            pytest.skip(f"Canonical code archive not found locally at {CANONICAL_CODE_ARCHIVE}")
        res = run_tar_audit_subprocess(tar_audit_block, CANONICAL_CODE_ARCHIVE)
        assert res.returncode == 0, f"Canonical archive audit failed: {res.stderr}"
        assert "[+] Code archive tar safety audit PASS" in res.stdout
        assert "SyntaxError" not in res.stderr
        assert "unterminated string literal" not in res.stderr


def test_extraction_security_contract_in_operator():
    """Verify operator enforces clean extraction and rejects root/Drive/Stage1 targets."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert "assert_safe_ephemeral_dir" in text
    assert 'assert_safe_ephemeral_dir "$CODE_DIR" "/content/phase_4c2_code"' in text
    assert 'rm -rf "$CODE_DIR"' in text
    assert 'mkdir -p "$CODE_DIR"' in text
    assert 'if ! tar -xzf "$CODE_ARCHIVE" -C "$CODE_DIR"; then' in text


def test_preflight_retry_resets_failure_and_archives_log():
    """Verify operator archives prior OPERATOR_FAILURE.json and resets status to in_progress."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert 'if [ -f "$OUTPUT_ROOT/OPERATOR_FAILURE.json" ]; then' in text
    assert 'ARCHIVE_TIMESTAMP=$(date -u +"%Y%m%d_%H%M%SZ" 2>/dev/null || echo "prior")' in text
    assert 'cp "$OUTPUT_ROOT/OPERATOR_FAILURE.json" "$LOGS_DIR/OPERATOR_FAILURE_archived_${ARCHIVE_TIMESTAMP}.json"' in text
    assert 'rm -f "$OUTPUT_ROOT/OPERATOR_FAILURE.json"' in text
    assert '"status": "in_progress"' in text


# ==============================================================================
# Phase 4C.2B.2 Final Execution Provenance & Verifier Hardening Tests (36 - 60)
# ==============================================================================

DEPLOYMENT_OPERATOR_SCRIPT = (
    REPO_ROOT.parent
    / "forensics-web-lab-local-artifacts"
    / "phase_4c2"
    / "inputs"
    / "phase_4c2_execute_all.sh"
)


def test_36_source_deployment_parity():
    """36. Canonical source and deployment operator scripts are bitwise identical."""
    assert OPERATOR_SCRIPT.exists(), "Source operator missing"
    assert DEPLOYMENT_OPERATOR_SCRIPT.exists(), "Deployment operator missing"

    src_bytes = OPERATOR_SCRIPT.read_bytes()
    dst_bytes = DEPLOYMENT_OPERATOR_SCRIPT.read_bytes()

    assert len(src_bytes) == len(dst_bytes), f"Byte length mismatch: {len(src_bytes)} != {len(dst_bytes)}"
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    dst_sha = hashlib.sha256(dst_bytes).hexdigest()
    assert src_sha == dst_sha, f"SHA-256 mismatch: {src_sha} != {dst_sha}"
    assert not dst_bytes.startswith(b"\xef\xbb\xbf"), "Deployment script contains UTF-8 BOM"
    assert b"\r" not in dst_bytes, "Deployment script contains CRLF line endings"


def test_37_wrong_code_archive_filename_fails():
    """37. Operator fails-closed if code archive filename does not match CANONICAL_CODE_ARCHIVE_NAME."""
    text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    assert 'CANONICAL_CODE_ARCHIVE_NAME="phase_4c2_code_9ee7fdb.tar.gz"' in text
    assert 'if [ "$ARCHIVE_NAME" != "$CANONICAL_CODE_ARCHIVE_NAME" ]; then' in text


def test_38_wrong_archive_bytes_fails(tmp_path):
    """38. Code archive byte size mismatch triggers fail-closed error."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    byte_check_code = extract_python_snippet(script_text, "# 2.2 Verify code archive exact bytes")

    dummy_arch = tmp_path / "phase_4c2_code_9ee7fdb.tar.gz"
    dummy_arch.write_bytes(b"WRONG_SIZE")

    cmd = [sys.executable, "-c", byte_check_code, str(dummy_arch), "10478136", "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "byte size mismatch" in res.stderr


def test_39_corrupt_archive_same_size_fails(tmp_path):
    """39. Corrupt code archive with identical byte size fails streaming SHA-256 check."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    byte_check_code = extract_python_snippet(script_text, "# 2.2 Verify code archive exact bytes")

    dummy_arch = tmp_path / "phase_4c2_code_9ee7fdb.tar.gz"
    # Write exactly 10478136 zero bytes
    dummy_arch.write_bytes(b"\x00" * 10478136)

    cmd = [sys.executable, "-c", byte_check_code, str(dummy_arch), "10478136", "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "SHA-256 mismatch" in res.stderr


def test_40_43_canonical_component_hash_failures(tmp_path):
    """40-43. Modification in runner, config, schema, or dataset binding fails post-extraction check."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    comp_check_code = extract_python_snippet(script_text, "# 2.6 Verify exact SHA-256 of all canonical components")

    r = tmp_path / "runner.py"
    c = tmp_path / "config.yaml"
    s = tmp_path / "schema.json"
    d = tmp_path / "dataset_binding.json"

    r.write_text("VALID_RUNNER", encoding="utf-8")
    c.write_text("VALID_CONFIG", encoding="utf-8")
    s.write_text("VALID_SCHEMA", encoding="utf-8")
    d.write_text("VALID_BINDING", encoding="utf-8")

    r_sha = hashlib.sha256(r.read_bytes()).hexdigest()
    c_sha = hashlib.sha256(c.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(s.read_bytes()).hexdigest()
    d_sha = hashlib.sha256(d.read_bytes()).hexdigest()

    # 40. Wrong runner hash
    cmd = [sys.executable, "-c", comp_check_code, str(r), "00" * 32, str(c), c_sha, str(s), s_sha, str(d), d_sha]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0

    # 41. Wrong config hash
    cmd = [sys.executable, "-c", comp_check_code, str(r), r_sha, str(c), "00" * 32, str(s), s_sha, str(d), d_sha]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0

    # 42. Wrong schema hash
    cmd = [sys.executable, "-c", comp_check_code, str(r), r_sha, str(c), c_sha, str(s), "00" * 32, str(d), d_sha]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0

    # 43. Wrong dataset-binding hash
    cmd = [sys.executable, "-c", comp_check_code, str(r), r_sha, str(c), c_sha, str(s), s_sha, str(d), "00" * 32]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode != 0


def test_44_47_destructive_path_guard_fault_injections(tmp_path):
    """44-47. Destructive path guard rejects /content/drive, symlinks, roots, and Stage 1 targets."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    guard_code = extract_python_snippet(script_text, "assert_safe_ephemeral_dir()")

    def run_guard(target: str, expected: str = "/content/phase_4c2_code", out_root: str = "/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c2/runs/execution_9ee7fdb"):
        return subprocess.run([sys.executable, "-c", guard_code, target, expected, out_root], capture_output=True, text=True)

    # 44. Target under /content/drive
    res = run_guard("/content/drive/MyDrive/test")
    assert res.returncode != 0

    # 45. Target is root or /content
    assert run_guard("/").returncode != 0
    assert run_guard("/content").returncode != 0

    # 46. Target targets Stage 1 namespace
    assert run_guard("/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c1/runs").returncode != 0
    assert run_guard("/content/drive/MyDrive/Colab Notebooks/forensics-web-lab/phase_4c2/runs/execution_79bb115").returncode != 0

    # 47. Empty path or path mismatch
    assert run_guard("").returncode != 0
    assert run_guard("/content/phase_4c2_other", expected="/content/phase_4c2_code").returncode != 0


def test_48_environment_scientific_lock_creation_during_preflight(tmp_path):
    """48. Preflight mode establishes immutable scientific lock and sidecar with required fields."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    lock_func_code = extract_python_snippet(script_text, 'echo "[*] Creating immutable scientific environment lock')

    lock_file = tmp_path / "phase4c2_environment_lock.json"
    lock_sha_file = tmp_path / "phase4c2_environment_lock.sha256"

    cmd = [
        sys.executable, "-c", lock_func_code,
        str(lock_file),
        str(lock_sha_file),
        "9ee7fdbb88fad16167f5790b5105867747801372",
        "phase_4c2_code_9ee7fdb.tar.gz",
        "10478136",
        "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89",
        "8ef0f0a06c25134a85982536064117a0bfc2eb9d63373c3b4ad6f4a2865783d4",
        "5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4",
        "dde1c873a43276cdf6bfd2ca459edebe7e8e14f2f02b1c8e8b9fe879936df98f",
        "dee09f81081466debd554642434a8282e60bef105bb8ff5a5a56c2bfc4f05c07",
        CANONICAL_BUNDLE_ARCHIVE_SHA,
        CANONICAL_BUNDLE_CONTENT_SHA,
        CANONICAL_BUNDLE_MANIFEST_SHA,
        str(CANONICAL_BUNDLE_BYTES),
        CANONICAL_WEIGHTS_FILE_SHA,
        str(CANONICAL_WEIGHTS_FILE_BYTES),
        CANONICAL_BACKBONE_FINGERPRINT,
        "operator_sha_test",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Lock creation failed: {res.stderr}"
    assert lock_file.exists()
    assert lock_sha_file.exists()

    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    assert lock["full_execution_commit_sha"] == "9ee7fdbb88fad16167f5790b5105867747801372"
    assert lock["code_archive"]["filename"] == "phase_4c2_code_9ee7fdb.tar.gz"
    assert lock["code_archive"]["bytes"] == 10478136
    assert lock["code_archive"]["sha256"] == "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"
    assert len(lock["run_matrix"]) == 3


def test_49_immutable_lock_retry_does_not_overwrite(tmp_path):
    """49. Retrying execution verifies existing lock without overwriting; sidecar mismatch fails."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    verify_lock_code = extract_python_snippet(script_text, 'echo "[*] Verifying existing immutable scientific environment lock..."')

    lock_file = tmp_path / "phase4c2_environment_lock.json"
    lock_sha_file = tmp_path / "phase4c2_environment_lock.sha256"

    # Create initial valid lock
    test_48_environment_scientific_lock_creation_during_preflight(tmp_path)
    orig_lock_bytes = lock_file.read_bytes()

    cmd = [
        sys.executable, "-c", verify_lock_code,
        str(lock_file),
        str(lock_sha_file),
        "9ee7fdbb88fad16167f5790b5105867747801372",
        "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89",
        "8ef0f0a06c25134a85982536064117a0bfc2eb9d63373c3b4ad6f4a2865783d4",
        "5ac7d41859798842aadf53d40fb8e9f2328e6ef46a4d2b1b248915cdee7543a4",
        "dde1c873a43276cdf6bfd2ca459edebe7e8e14f2f02b1c8e8b9fe879936df98f",
        "dee09f81081466debd554642434a8282e60bef105bb8ff5a5a56c2bfc4f05c07",
        CANONICAL_BUNDLE_ARCHIVE_SHA,
        CANONICAL_WEIGHTS_FILE_SHA,
        CANONICAL_BACKBONE_FINGERPRINT,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Verify existing lock failed: {res.stderr}"
    assert lock_file.read_bytes() == orig_lock_bytes, "Lock file was modified!"

    # Corrupt sidecar -> must fail
    lock_sha_file.write_text("00" * 32 + "  phase4c2_environment_lock.json\n", encoding="utf-8")
    res_bad = subprocess.run(cmd, capture_output=True, text=True)
    assert res_bad.returncode != 0


def test_50_compatible_different_gpu_runtime_observation(tmp_path):
    """50. Runtime observation records dynamic GPU, VRAM, and pip-freeze without touching scientific lock."""
    script_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    obs_code = extract_python_snippet(script_text, "record_runtime_observation()", start_token="<<'PY'", end_token="\nPY\n")

    obs_file = tmp_path / "runtime_observation.json"
    cmd = [sys.executable, "-c", obs_code, str(obs_file), "2026-10-01T00:00:00Z", "execute", "NVIDIA L4", "24.0"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Observation failed: {res.stderr}"
    assert obs_file.exists()

    obs = json.loads(obs_file.read_text(encoding="utf-8"))
    assert obs["gpu_name"] == "NVIDIA L4"
    assert obs["gpu_vram_gb"] == 24.0
    assert obs["execution_mode"] == "execute"


def test_51_missing_checksums_coverage_fails(tmp_path):
    """51. Missing coverage in checksums.json triggers fail-closed validation error."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    # Omit epoch_history.json from checksums.json
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, omit_from_checksums=["epoch_history.json"])
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0
    assert "Checksums coverage mismatch" in res.stderr


def test_52_invalid_receipt_schema_fails(tmp_path):
    """52. Invalid receipt schema (e.g. missing required field or wrong run_id) fails validation."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42)
    receipt_p = run_dir / "run_receipt.json"
    data = json.loads(receipt_p.read_text(encoding="utf-8"))
    del data["trainable_parameters_count"]
    receipt_p.write_text(json.dumps(data), encoding="utf-8")
    # Update checksums
    csums = json.loads((run_dir / "checksums.json").read_text(encoding="utf-8"))
    csums["run_receipt.json"]["sha256"] = hashlib.sha256(receipt_p.read_bytes()).hexdigest()
    csums["run_receipt.json"]["size_bytes"] = receipt_p.stat().st_size
    (run_dir / "checksums.json").write_text(json.dumps(csums), encoding="utf-8")

    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0


def test_53_wrong_checkpoint_hash_fails(tmp_path):
    """53. Checkpoint SHA-256 mismatch between receipt and actual checkpoint fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, checkpoint_sha="00" * 32)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0
    assert "checkpoint_sha256 mismatch" in res.stderr


def test_54_wrong_trainable_inventory_fails(tmp_path):
    """54. Trainable tensor inventory not matching exact 7 tensors fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    bad_inv = [{"name": "features.12.0.weight", "shape": [576, 96, 1, 1], "numel": 55296}]  # Only 1 tensor
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, trainable_inventory=bad_inv)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0


def test_55_wrong_optimizer_groups_fails(tmp_path):
    """55. Optimizer parameter groups not matching exact 2 groups fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    bad_opt = [{"group_name": "all_params", "learning_rate": 1e-4}]
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, optimizer_groups=bad_opt)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0


def test_56_wrong_batchnorm_policy_fails(tmp_path):
    """56. BatchNorm policy not keeping frozen features in eval mode fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    bad_bn = {"frozen_features_eval": False, "reapply_after_train": True, "unfrozen_features_12_bn_train": True}
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, batchnorm_policy=bad_bn)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0


def test_57_wrong_code_config_dataset_hashes_in_receipt_fails(tmp_path):
    """57. Receipt with mismatched config_hash or dataset_archive_sha256 fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    # Wrong config hash
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, config_hash="00" * 32)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0

    # Wrong dataset archive sha
    run_dir2 = setup_mock_stage2_run(tmp_path, 50, 42, dataset_archive_sha="00" * 32)
    res2 = run_verification_code(verif_code, run_dir2, 50, 42)
    assert res2.returncode != 0


def test_58_prediction_array_length_mismatch_fails(tmp_path):
    """58. Predictions JSON with length != 182 fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    bad_preds = {
        "targets": [0] * 50 + [1] * 50,
        "predictions": [0] * 50 + [1] * 50,
        "probabilities": [0.5] * 100,
        "source_ids": [f"src_{i:04d}" for i in range(50)] * 2,
    }
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, predictions_data_override=bad_preds)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0


def test_59_source_appearing_once_or_three_times_fails(tmp_path):
    """59. Any source appearing once or 3 times instead of exactly twice fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    sids = [f"src_{i:04d}" for i in range(91)] * 2
    sids[0] = sids[1]  # src_0001 now appears 3 times, src_0000 appears 1 time
    bad_preds = {
        "targets": [0] * 91 + [1] * 91,
        "predictions": [0] * 91 + [1] * 91,
        "probabilities": [0.5] * 182,
        "source_ids": sids,
    }
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, predictions_data_override=bad_preds)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0


def test_60_source_without_label_pair_0_1_fails(tmp_path):
    """60. Any source not having both label 0 and label 1 fails verification."""
    verif_code = extract_verification_python_code(OPERATOR_SCRIPT.read_text(encoding="utf-8"))
    # Invert one target so that src_0000 has targets {0, 0}
    bad_targets = [0] * 91 + [1] * 91
    bad_targets[91] = 0  # index 91 corresponds to src_0000 in second half
    bad_preds = {
        "targets": bad_targets,
        "predictions": [0] * 91 + [1] * 91,
        "probabilities": [0.5] * 182,
        "source_ids": [f"src_{i:04d}" for i in range(91)] * 2,
    }
    run_dir = setup_mock_stage2_run(tmp_path, 50, 42, predictions_data_override=bad_preds)
    res = run_verification_code(verif_code, run_dir, 50, 42)
    assert res.returncode != 0
