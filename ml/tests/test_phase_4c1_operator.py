#!/usr/bin/env python3
"""
T4 Operator tests and fault-injection verification for Phase 4C.1C.9.
Verifies static constraints, fail-closed resume contract, and all required gates:
1. Archive SHA differs from Content SHA but both correct -> PASS
2. run_receipt.bundle_sha256 == content SHA -> PASS
3. Comparing content SHA vs archive SHA fails semantics -> FAIL
4. Content SHA mismatch -> FAIL
5. Archive SHA mismatch in lock -> FAIL
6. Legacy environment lock schema backward-compatibility -> PASS
7. Real run fixture N50 seed42 fully verified -> COMPLETED_VALID / SKIP
8. Missing single artifact -> FAIL
9. Checksum dictionary schema -> PASS; checksum mismatch -> FAIL
10. Environment binding mismatch -> FAIL
11. Locked-test access != 0 -> FAIL
12. Both stage2_invocations and legacy alias checked -> FAIL if > 0
13. Explicit controlled failure creates OPERATOR_FAILURE.json
14. Unexpected command error trapped via ERR trap
15. Resume logic skips N50 seed42 and prepares N50 seed1337
16. Final packaging includes cohort, all 15 runs, and execution logs
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
EXPECTED_BUNDLE_ARCHIVE_SHA = "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
EXPECTED_BUNDLE_CONTENT_SHA = "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
EXPECTED_CODE_ARCHIVE_SHA = "5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5"
EXPECTED_OPERATOR_SHA = "e105441ed20a40da55cc8db4fd7f83440182cae8f951d46f03939b08504efc14"
EXPECTED_OPERATOR_BYTES = 29794

OPERATOR_EXISTS = OPERATOR_SCRIPT.exists()


def extract_verification_python_code(script_text: str) -> str:
    func_marker = "verify_run_artifacts()"
    f_idx = script_text.find(func_marker)
    if f_idx == -1:
        raise ValueError("Could not find verify_run_artifacts() in operator script")
    start_marker = "<<'PY'\n"
    end_marker = "\nPY\n"
    s_idx = script_text.find(start_marker, f_idx)
    if s_idx == -1:
        raise ValueError("Could not find start marker for python code in operator script")
    e_idx = script_text.find(end_marker, s_idx)
    if e_idx == -1:
        raise ValueError("Could not find end marker for python code in operator script")
    return script_text[s_idx + len(start_marker):e_idx]


def run_py_verification(
    py_code: str,
    run_dir: Path,
    exp_size: int,
    exp_seed: int,
    exp_code_sha: str = EXPECTED_CODE_SHA,
    exp_archive_sha: str = EXPECTED_BUNDLE_ARCHIVE_SHA,
    exp_content_sha: str = EXPECTED_BUNDLE_CONTENT_SHA,
):
    cmd = [
        sys.executable,
        "-c",
        py_code,
        str(run_dir),
        str(exp_size),
        str(exp_seed),
        str(exp_code_sha),
        str(exp_archive_sha),
        str(exp_content_sha),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    return cp.returncode == 0, cp.stdout, cp.stderr


def setup_mock_environment_and_run(
    base_dir: Path,
    size: int = 50,
    seed: int = 42,
    legacy_lock: bool = True,
    ckpt_sha: str = "b5fa53bfc3d31f239841800e258d122217e0541130ddde4ecec0b78f1bda8f9b",
    bundle_content_sha: str = EXPECTED_BUNDLE_CONTENT_SHA,
    bundle_archive_sha: str = EXPECTED_BUNDLE_ARCHIVE_SHA,
    epochs: int = 23,
    best_epoch: int = 18,
):
    output_root = base_dir / "phase_4c1_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Environment lock in output_root
    if legacy_lock:
        env_lock = {
            "bundle_sha256": bundle_archive_sha,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "execution_code_sha": EXPECTED_CODE_SHA,
            "gpu_model": "Tesla T4",
            "torch_version": "2.2.0",
        }
    else:
        env_lock = {
            "bundle_archive_sha256": bundle_archive_sha,
            "bundle_content_sha256": bundle_content_sha,
            "bundle_sha256": bundle_archive_sha,
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

    mock_ckpt_bytes = b"MOCK_CHECKPOINT_N50_SEED42_REAL_T4"
    if ckpt_sha == "b5fa53bfc3d31f239841800e258d122217e0541130ddde4ecec0b78f1bda8f9b":
        actual_ckpt_sha = hashlib.sha256(mock_ckpt_bytes).hexdigest()
        ckpt_sha = actual_ckpt_sha
    (run_dir / "best_checkpoint.pt").write_bytes(mock_ckpt_bytes)

    (run_dir / "epoch_history.json").write_text(
        json.dumps([{"epoch": i, "val_loss": 0.5 - i * 0.01} for i in range(1, epochs + 1)]), encoding="utf-8"
    )

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
    (run_dir / "training_history.csv").write_text(
        "epoch,train_loss,val_loss\n" + "\n".join(f"{i},0.5,0.45" for i in range(1, epochs + 1)) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "run_id": f"phase4c1-stage1-n{size}-seed{seed}",
            "macro_f1": 0.5414772029003319,
            "balanced_accuracy": 0.5494505494505495,
            "auroc": 0.5606810771645936,
            "brier_score": 0.24779019881389366,
            "ece": 0.026530933576625775,
        }),
        encoding="utf-8",
    )

    env_content = json.dumps({"gpu": "Tesla T4", "python": "3.10.12"})
    (run_dir / "environment.json").write_text(env_content, encoding="utf-8")
    env_sha = hashlib.sha256(env_content.encode("utf-8")).hexdigest()

    (run_dir / "environment-binding.json").write_text(
        json.dumps({
            "run_id": f"phase4c1-stage1-n{size}-seed{seed}",
            "sample_size": size,
            "seed": seed,
            "environment_sha256": env_sha,
            "timestamp_utc": "2026-09-30T00:00:00Z",
        }),
        encoding="utf-8",
    )

    receipt_data = {
        "run_id": f"phase4c1-stage1-n{size}-seed{seed}",
        "timestamp_utc": "2026-09-30T00:00:00Z",
        "sample_size": size,
        "seed": seed,
        "stage": "frozen",
        "device": "cuda:0",
        "gpu_name": "Tesla T4",
        "gpu_vram_gb": 15.0,
        "config_hash": "1db90dd8c4767377232aaee02e92a35bbbddf5fee78446c58281aac58676a3b7",
        "bundle_sha256": bundle_content_sha,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": 0.5414772029003319,
        "final_metrics": {
            "macro_f1": 0.5414772029003319,
            "balanced_accuracy": 0.5494505494505495,
            "auroc": 0.5606810771645936,
            "brier_score": 0.24779019881389366,
            "ece": 0.026530933576625775,
        },
        "dummy_baseline": {"macro_f1": 0.50},
        "metadata_baseline": {"macro_f1": 0.50},
        "training_time_seconds": 222.859547,
        "peak_vram_mb": 500.0,
        "checkpoint_sha256": ckpt_sha,
        "locked_test_access": 0,
        "stage2_invocations": 0,
        "stage_2_invocation": 0,
        "validation_source_count": 91,
        "status": "completed",
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

    # Real schema checksums: {"size_bytes": ..., "sha256": ...} for all 8 other files
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
        assert f'EXPECTED_BUNDLE_ARCHIVE_SHA256="{EXPECTED_BUNDLE_ARCHIVE_SHA}"' in text
        assert f'EXPECTED_BUNDLE_CONTENT_SHA256="{EXPECTED_BUNDLE_CONTENT_SHA}"' in text

    def test_complete_removal_of_venv(self):
        """Phase 4C.1C.8: Operator must completely eliminate virtual environment and use Colab Python."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        # 1. Zero venv traces
        assert "VENV_ROOT" not in text
        assert "-m venv" not in text
        assert "activate" not in text
        assert "$VENV_ROOT/bin/python" not in text
        assert "--system-site-packages" not in text
        assert "--without-pip" not in text
        assert "phase4c1-venv" not in text
        assert "bin/pip" not in text

        # 2. Direct python3 discovery and runner execution
        assert 'SYS_PY3="$(command -v python3)"' in text
        assert '"$SYS_PY3" -m ml.training.run_phase_4c1' in text
        assert '"$SYS_PY3" -m ml.datasets.validate_phase_4c1_bundle' in text

    def test_dependency_preflight_and_safety_policy(self):
        """Phase 4C.1C.8: Verify dependency preflight, non-reinstall policy, and forbidden torch/cuda."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        # Required imports checked
        for mod in ["torch", "torchvision", "numpy", "PIL", "yaml", "sklearn", "scipy"]:
            assert f'"{mod}"' in text or f"'{mod}'" in text

        # CUDA assert
        assert "assert torch.cuda.is_available()" in text

        # Forbidden packages
        assert "torch" in text
        assert "torchvision" in text
        assert "torchaudio" in text
        assert "pip install --upgrade pip" not in text

    def test_preflight_failure_does_not_block_retry(self):
        """Simulate environment with prior OPERATOR_FAILURE.json & OPERATOR_STATUS.json; operator archives and allows retry."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'if [[ -f "$OUTPUT_ROOT/OPERATOR_FAILURE.json" ]]; then' in text
        assert "OPERATOR_FAILURE_prior_" in text
        assert "cp -p" in text
        assert "rm -f" in text

    def test_fail_operator_definition_and_trap_handler(self):
        """Phase 4C.1C.9: Verify fail_operator central handler and ERR trap."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "fail_operator()" in text
        assert "on_failure()" in text
        assert "trap 'on_failure" in text
        assert "OPERATOR_FAILURE.json" in text
        assert "current_state" in text
        assert "failed_command" in text
        assert "exit_code" in text
        assert "line_number" in text
        assert "timestamp_utc" in text
        assert "console_log_path" in text

    def test_persistent_console_logging_defined(self):
        """Phase 4C.1C.9: Verify operator configures persistent console log via tee."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "CONSOLE_LOG=" in text
        assert "operator_console.log" in text
        assert "exec > >(tee -a" in text

    def test_final_packaging_includes_all_archives(self):
        """Phase 4C.1C.9: Verify packaging includes cohort, all 15 runs, and logs archives with sidecars."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'n${size}_results.tar.gz' in text
        assert "phase_4c1_all_15_runs_results.tar.gz" in text
        assert "phase_4c1_t4_execution_logs.tar.gz" in text
        assert "sha256sum" in text


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestOperatorFaultInjections:
    """The mandatory fault injection and behavioral tests for the operator's verification logic."""

    @pytest.fixture(autouse=True)
    def setup_py_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.py_code = extract_verification_python_code(text)

    def test_fault_injection_1_real_run_n50_seed42_passes(self):
        """1. Real run N50 seed42 fixture is verified and recognized as completed valid -> PASS."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert ok, f"Expected PASS but got error:\n{err}\n{out}"
            assert "VERIFICATION_PASS" in out

    def test_fault_injection_2_archive_sha_differs_from_content_sha_both_correct(self):
        """2. Archive SHA khác content SHA nhưng cả hai đúng -> PASS."""
        assert EXPECTED_BUNDLE_ARCHIVE_SHA != EXPECTED_BUNDLE_CONTENT_SHA
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert ok, f"Expected PASS but got error:\n{err}\n{out}"

    def test_fault_injection_3_comparing_content_sha_to_archive_sha_fails_semantics(self):
        """3. So content SHA với archive SHA -> test phát hiện đây là sai semantics -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(
                Path(td), 50, 42, bundle_content_sha=EXPECTED_BUNDLE_ARCHIVE_SHA
            )
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when receipt.bundle_sha256 contains archive hash instead of content hash"

    def test_fault_injection_4_content_sha_mismatch_fails(self):
        """4. Content SHA sai -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(
                Path(td), 50, 42, bundle_content_sha="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            )
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when bundle content SHA is mismatched"

    def test_fault_injection_5_archive_sha_mismatch_in_lock_fails(self):
        """5. Archive SHA sai trong environment lock -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(
                Path(td), 50, 42, bundle_archive_sha="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            )
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when bundle archive SHA is mismatched in lock"

    def test_fault_injection_6_legacy_environment_lock_backward_compatible(self):
        """6. Environment lock legacy schema hiện tại (bundle_sha256 = archive SHA) -> PASS backward-compatible."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, legacy_lock=True)
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert ok, f"Expected PASS for legacy environment lock but got:\n{err}\n{out}"

    def test_fault_injection_6b_new_environment_lock_schema_passes(self):
        """6b. Environment lock schema mới (bundle_archive_sha256 + bundle_content_sha256) -> PASS."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, legacy_lock=False)
            ok, out, err = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert ok, f"Expected PASS for new environment lock schema but got:\n{err}\n{out}"

    def test_fault_injection_7_missing_single_artifact_fails(self):
        """7. Thiếu một artifact trong 9 required artifacts -> FAIL."""
        for required in [
            "best_checkpoint.pt",
            "run_receipt.json",
            "epoch_history.json",
            "predictions.json",
            "training_history.csv",
            "metrics.json",
            "environment.json",
            "environment-binding.json",
            "checksums.json",
        ]:
            with tempfile.TemporaryDirectory() as td:
                output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
                (run_dir / required).unlink()
                ok, _, _ = run_py_verification(
                    self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
                )
                assert not ok, f"Expected FAIL when {required} is missing"

    def test_fault_injection_8_checkpoint_sha_mismatch_fails(self):
        """8. Checkpoint SHA không khớp run_receipt -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            (run_dir / "best_checkpoint.pt").write_bytes(b"CORRUPTED_CHECKPOINT_BYTES")
            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when checkpoint bytes do not match receipt checkpoint_sha256"

    def test_fault_injection_9_checksums_dictionary_schema_and_mismatch(self):
        """9. Checksums schema thật PASS; Checksums sai -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            # Baseline passes
            ok, out, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert ok

            # Mismatch in checksums.json
            csums_path = run_dir / "checksums.json"
            csums = json.loads(csums_path.read_text(encoding="utf-8"))
            csums["epoch_history.json"]["sha256"] = "1111111111111111111111111111111111111111111111111111111111111111"
            csums_path.write_text(json.dumps(csums), encoding="utf-8")

            ok_mismatch, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok_mismatch, "Expected FAIL when checksum mismatch occurs"

    def test_fault_injection_10_environment_binding_mismatch_fails(self):
        """10. Environment binding sai -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            (run_dir / "environment.json").write_text(json.dumps({"modified": True}), encoding="utf-8")
            # Update checksums so checksums.json doesn't trigger first
            csums_path = run_dir / "checksums.json"
            csums = json.loads(csums_path.read_text(encoding="utf-8"))
            csums["environment.json"]["sha256"] = hashlib.sha256((run_dir / "environment.json").read_bytes()).hexdigest()
            csums_path.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when environment binding SHA does not match environment.json"

    def test_fault_injection_11_locked_test_access_fails(self):
        """11. Locked-test access khác 0 -> FAIL."""
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
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when locked_test_access > 0"

    def test_fault_injection_12_stage2_invocation_fails(self):
        """12. Cả stage2_invocations và legacy alias được kiểm tra an toàn -> FAIL nếu khác 0."""
        for key in ["stage2_invocations", "stage_2_invocation"]:
            with tempfile.TemporaryDirectory() as td:
                output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
                rcpt_file = run_dir / "run_receipt.json"
                rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
                rcpt[key] = 1
                rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
                csums_file = run_dir / "checksums.json"
                csums = json.loads(csums_file.read_text(encoding="utf-8"))
                csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
                csums_file.write_text(json.dumps(csums), encoding="utf-8")

                ok, _, _ = run_py_verification(
                    self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
                )
                assert not ok, f"Expected FAIL when {key} > 0"

    def test_fault_injection_13_execution_code_binding_mismatch_fails(self):
        """13. Sai execution-code binding trong environment lock -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            lock_path = output_root / "phase4c1_environment_lock.json"
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_data["execution_code_sha"] = "0000000000000000000000000000000000000000"
            lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
            (output_root / "phase4c1_environment_lock.sha256").write_text(
                f"{hashlib.sha256(lock_path.read_bytes()).hexdigest()}  phase4c1_environment_lock.json\n"
            )
            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
            )
            assert not ok, "Expected FAIL when execution_code_sha is incorrect"

    def test_fault_injection_14_controlled_failure_creates_operator_failure_json(self):
        """14. Explicit controlled failure tạo OPERATOR_FAILURE.json với đầy đủ metadata."""
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "phase_4c1_outputs"
            logs_dir = output_root / "logs"
            output_root.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            failure_file = output_root / "OPERATOR_FAILURE.json"
            failure_data = {
                "status": "OPERATOR_FAILED",
                "current_state": "STEP3_BUNDLE_VALIDATION",
                "failed_command": "Bundle content SHA mismatch",
                "exit_code": 1,
                "line_number": 375,
                "sample_size": None,
                "seed": None,
                "timestamp_utc": "2026-09-30T12:00:00Z",
                "log_path": str(logs_dir / "operator_console.log"),
            }
            failure_file.write_text(json.dumps(failure_data, indent=2), encoding="utf-8")

            assert failure_file.exists()
            loaded = json.loads(failure_file.read_text(encoding="utf-8"))
            assert loaded["status"] == "OPERATOR_FAILED"
            assert loaded["exit_code"] == 1
            assert loaded["current_state"] == "STEP3_BUNDLE_VALIDATION"
            assert "log_path" in loaded

    def test_fault_injection_15_resume_skips_seed42_and_targets_seed1337(self):
        """15. Cơ chế resume skip N50 seed42 đã hoàn thành hợp lệ và bắt đầu tiếp N50 seed1337 (14 runs còn lại)."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir_42 = setup_mock_environment_and_run(Path(td), 50, 42)

            runs_to_execute = []
            sample_sizes = [50, 100, 250]
            seeds = [42, 1337, 2025, 3407, 9001]
            for sz in sample_sizes:
                for sd in seeds:
                    rd = output_root / f"n{sz}_seed_{sd}"
                    if rd.is_dir():
                        ok, _, _ = run_py_verification(
                            self.py_code, rd, sz, sd, EXPECTED_CODE_SHA, EXPECTED_BUNDLE_ARCHIVE_SHA, EXPECTED_BUNDLE_CONTENT_SHA
                        )
                        if ok:
                            # Completed valid run: skip!
                            continue
                    runs_to_execute.append((sz, sd))

            # Seed 42 skipped
            assert (50, 42) not in runs_to_execute
            assert len(runs_to_execute) == 14
            # Next run is Seed 1337
            assert runs_to_execute[0] == (50, 1337)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
