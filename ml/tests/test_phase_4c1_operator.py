#!/usr/bin/env python3
"""
T4 Operator tests and fault-injection verification for Phase 4C.1.
Verifies static constraints, fail-closed resume contract, and the 7 mandatory fault injections:
1. Execution-code binding correct -> PASS
2. Binding missing -> FAIL
3. Binding incorrect -> FAIL
4. Completed valid run -> resumed/skipped
5. Partial run -> fail-closed
6. Locked-test access != 0 -> FAIL
7. Stage 2 invocation != 0 -> FAIL
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parents[2]
PEER_ARTIFACTS_DIR = Path(__file__).parents[3] / "forensics-web-lab-local-artifacts"
LOCAL_ARTIFACTS_DIR = Path(os.environ.get("FORENSICS_LOCAL_ARTIFACTS", str(PEER_ARTIFACTS_DIR)))
OPERATOR_SCRIPT = LOCAL_ARTIFACTS_DIR / "phase_4c1" / "t4_transfer" / "phase_4c1_t4_execute_all_stage1.sh"
OPERATOR_SIDECAR = OPERATOR_SCRIPT.with_name(OPERATOR_SCRIPT.name + ".sha256")

EXPECTED_CODE_SHA = "79bb11527d900fd387de1f41f2010c4152b7fea7"
EXPECTED_BUNDLE_SHA = "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
EXPECTED_CODE_ARCHIVE_SHA = "5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5"
EXPECTED_OPERATOR_SHA = "1a7570d757ccfc1b471c636f64d0f001c3b6fcc9306b9894f94186f909f1f3c4"
EXPECTED_OPERATOR_BYTES = 21662

OPERATOR_EXISTS = OPERATOR_SCRIPT.exists()


def extract_verification_python_code(script_text: str) -> str:
    start_marker = "<<'PY'\n"
    end_marker = "\nPY\n"
    s_idx = script_text.find(start_marker)
    if s_idx == -1:
        raise ValueError("Could not find start marker for python code in operator script")
    e_idx = script_text.find(end_marker, s_idx)
    if e_idx == -1:
        raise ValueError("Could not find end marker for python code in operator script")
    return script_text[s_idx + len(start_marker):e_idx]


def run_py_verification(py_code: str, run_dir: Path, exp_size: int, exp_seed: int, exp_code_sha: str, exp_bundle_sha: str):
    cmd = [
        sys.executable,
        "-c",
        py_code,
        str(run_dir),
        str(exp_size),
        str(exp_seed),
        str(exp_code_sha),
        str(exp_bundle_sha),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    return cp.returncode == 0, cp.stdout, cp.stderr


def setup_mock_environment_and_run(base_dir: Path, size: int = 50, seed: int = 42):
    output_root = base_dir / "phase_4c1_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Environment lock in output_root
    env_lock = {
        "bundle_sha256": EXPECTED_BUNDLE_SHA,
        "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
        "execution_code_sha": EXPECTED_CODE_SHA,
        "gpu_model": "Tesla T4",
        "torch_version": "2.2.0",
    }
    lock_file = output_root / "phase4c1_environment_lock.json"
    lock_file.write_text(json.dumps(env_lock, indent=2), encoding="utf-8")
    lock_sha = hashlib.sha256(lock_file.read_bytes()).hexdigest()
    (output_root / "phase4c1_environment_lock.sha256").write_text(
        f"{lock_sha}  phase4c1_environment_lock.json\n", encoding="utf-8"
    )

    # 2. Run directory
    run_dir = output_root / f"n{size}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "best_checkpoint.pt").write_bytes(b"MOCK_CHECKPOINT_DATA_4B")
    (run_dir / "epoch_history.json").write_text(json.dumps([{"epoch": 1, "val_loss": 0.45}]), encoding="utf-8")

    targets = [0] * 91 + [1] * 91
    preds = [0] * 91 + [1] * 91
    probs = [0.1] * 91 + [0.9] * 91
    s_ids = [f"src_{i:04d}" for i in range(91)] * 2
    (run_dir / "predictions.json").write_text(
        json.dumps({
            "targets": targets,
            "predictions": preds,
            "probabilities": probs,
            "source_ids": s_ids,
        }),
        encoding="utf-8",
    )
    (run_dir / "training_history.csv").write_text("epoch,train_loss,val_loss\n1,0.5,0.45\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": f"n{size}_seed_{seed}", "macro_f1": 0.85}), encoding="utf-8"
    )

    env_content = json.dumps({"gpu": "Tesla T4", "python": "3.10.12"})
    (run_dir / "environment.json").write_text(env_content, encoding="utf-8")
    env_sha = hashlib.sha256(env_content.encode("utf-8")).hexdigest()

    (run_dir / "environment-binding.json").write_text(
        json.dumps({
            "run_id": f"n{size}_seed_{seed}",
            "sample_size": size,
            "seed": seed,
            "environment_sha256": env_sha,
            "timestamp_utc": "2026-09-30T00:00:00Z",
        }),
        encoding="utf-8",
    )

    # Real schema matching run_phase_4c1.py snapshot 79bb115
    receipt_data = {
        "run_id": f"n{size}_seed_{seed}",
        "timestamp_utc": "2026-09-30T00:00:00Z",
        "sample_size": size,
        "seed": seed,
        "stage": "frozen",
        "device": "cuda:0",
        "gpu_name": "Tesla T4",
        "gpu_vram_gb": 15.0,
        "config_hash": "1db90dd8c4767377232aaee02e92a35bbbddf5fee78446c58281aac58676a3b7",
        "bundle_sha256": EXPECTED_BUNDLE_SHA,
        "epochs_completed": 8,
        "best_epoch": 5,
        "best_val_macro_f1": 0.85,
        "final_metrics": {"macro_f1": 0.85},
        "dummy_baseline": {"macro_f1": 0.50},
        "metadata_baseline": {"macro_f1": 0.50},
        "training_time_seconds": 120.5,
        "peak_vram_mb": 500.0,
        "checkpoint_sha256": "abcdef123456",
        "locked_test_access": 0,
        "stage2_invocations": 0,
        "validation_source_count": 91,
        "status": "completed",
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

    # Real schema checksums: {"size_bytes": ..., "sha256": ...}
    csums = {}
    for p in run_dir.glob("*"):
        if p.is_file():
            csums[p.name] = {
                "size_bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
    (run_dir / "checksums.json").write_text(json.dumps(csums, indent=2), encoding="utf-8")

    return output_root, run_dir


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestOperatorStaticInvariants:
    """Static integrity assertions for operator script."""

    def test_file_exists_and_bytes(self):
        assert OPERATOR_SCRIPT.exists()
        raw = OPERATOR_SCRIPT.read_bytes()
        assert len(raw) == EXPECTED_OPERATOR_BYTES
        assert not raw.startswith(b"\xef\xbb\xbf"), "Must not have UTF-8 BOM"
        assert b"\r" not in raw, "Must use LF line endings"

    def test_sha256_and_sidecar(self):
        raw = OPERATOR_SCRIPT.read_bytes()
        act_sha = hashlib.sha256(raw).hexdigest()
        assert act_sha == EXPECTED_OPERATOR_SHA
        assert OPERATOR_SIDECAR.exists()
        sidecar_text = OPERATOR_SIDECAR.read_text(encoding="utf-8").strip()
        assert act_sha == sidecar_text.split()[0].lower()

    def test_safety_invariants(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "drive.mount" not in text
        assert "/content/drive" not in text
        assert "pip install torch" not in text
        assert "pip install --upgrade torch" not in text
        assert "--stage unfrozen" not in text
        assert "--eval-partition locked_test" not in text

        # Credentials check
        for kw in ["ghp_", "github_pat", "token=", "PRIVATE_KEY", "AWS_ACCESS", "password="]:
            assert kw not in text

        # Matrix check
        assert "SAMPLE_SIZES=(50 100 250)" in text
        assert "SEEDS=(42 1337 2025 3407 9001)" in text

        # Execution code SHA constant
        assert f'EXECUTION_CODE_SHA="{EXPECTED_CODE_SHA}"' in text


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestOperatorFaultInjections:
    """The 7 mandatory fault injection tests for the operator's verification logic."""

    @pytest.fixture(autouse=True)
    def setup_py_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.py_code = extract_verification_python_code(text)

    def test_fault_injection_1_correct_binding_passes(self):
        """1. Execution-code binding đúng -> PASS."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert ok, f"Expected PASS but got error:\n{err}\n{out}"
            assert "VERIFICATION_PASS" in out

    def test_fault_injection_2_binding_missing_fails(self):
        """2. Binding thiếu -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            lock_file = output_root / "phase4c1_environment_lock.json"
            lock_sidecar = output_root / "phase4c1_environment_lock.sha256"

            # Case 2a: Lock file deleted
            lock_file.unlink()
            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok, "Expected FAIL when environment lock is missing"

            # Case 2b: Lock file present but execution_code_sha missing
            lock_data = {
                "bundle_sha256": EXPECTED_BUNDLE_SHA,
                "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            }
            lock_file.write_text(json.dumps(lock_data), encoding="utf-8")
            lock_sidecar.write_text(
                f"{hashlib.sha256(lock_file.read_bytes()).hexdigest()}  phase4c1_environment_lock.json\n"
            )
            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok, "Expected FAIL when execution_code_sha field is missing"

    def test_fault_injection_3_binding_wrong_fails(self):
        """3. Binding sai -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            lock_file = output_root / "phase4c1_environment_lock.json"
            lock_sidecar = output_root / "phase4c1_environment_lock.sha256"

            lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
            lock_data["execution_code_sha"] = "0000000000000000000000000000000000000000"
            lock_file.write_text(json.dumps(lock_data), encoding="utf-8")
            lock_sidecar.write_text(
                f"{hashlib.sha256(lock_file.read_bytes()).hexdigest()}  phase4c1_environment_lock.json\n"
            )

            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok, "Expected FAIL when execution_code_sha is incorrect"

    def test_fault_injection_4_completed_valid_run_resumed(self):
        """4. Completed run hợp lệ -> được resume/skip."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert ok
            assert "VERIFICATION_PASS" in out

    def test_fault_injection_5_partial_run_fails_closed(self):
        """5. Partial run -> fail-closed."""
        with tempfile.TemporaryDirectory() as td:
            # Case 5a: Missing required checkpoint file
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            (run_dir / "best_checkpoint.pt").unlink()
            ok_5a, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok_5a, "Expected FAIL when checkpoint is missing"

            # Case 5b: Corrupted checkpoint failing checksums.json
            (run_dir / "best_checkpoint.pt").write_bytes(b"CORRUPTED_BYTES")
            ok_5b, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok_5b, "Expected FAIL when checksum mismatch occurs"

            # Case 5c: Incomplete status in run_receipt.json
            (run_dir / "best_checkpoint.pt").write_bytes(b"MOCK_CHECKPOINT_DATA_4B")
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            rcpt["status"] = "in_progress"
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok_5c, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok_5c, "Expected FAIL when receipt status is not completed"

    def test_fault_injection_6_locked_test_access_fails(self):
        """6. Locked-test access khác 0 -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            rcpt["locked_test_access"] = 1
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok, "Expected FAIL when locked_test_access > 0"

    def test_fault_injection_7_stage2_invocation_fails(self):
        """7. Stage 2 invocation khác 0 -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))

            # Case 7a: stage2_invocations = 1
            rcpt["stage2_invocations"] = 1
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok_7a, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok_7a, "Expected FAIL when stage2_invocations > 0"

            # Case 7b: stage_2_invocation = 1
            rcpt["stage2_invocations"] = 0
            rcpt["stage_2_invocation"] = 1
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok_7b, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_SHA
            )
            assert not ok_7b, "Expected FAIL when stage_2_invocation > 0"
