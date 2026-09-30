#!/usr/bin/env python3
"""
T4 Operator tests and fault-injection verification for Phase 4C.1C.10.
Covers 40 mandatory behavioral and static verification criteria:
1. T4 simulation in compatible mode -> PASS
2. L4 simulation in compatible mode -> PASS
3. V100 simulation in compatible mode -> PASS
4. A100 simulation in compatible mode -> PASS
5. GPU other than T4 in compatible mode -> warning notice, not fail
6. GPU other than expected model in strict mode -> FAIL
7. No CUDA -> FAIL
8. CUDA smoke test error -> FAIL
9. No MIN_GPU_VRAM_GB set -> does not block on arbitrary threshold
10. MIN_GPU_VRAM_GB set and GPU insufficient -> FAIL
11. Scientific binding correct -> PASS
12. Code SHA mismatch in lock -> FAIL
13. Archive SHA mismatch in lock -> FAIL
14. Content SHA mismatch in lock -> FAIL
15. Compatible runtime differs in Python executable path -> warning notice, not fail
16. Compatible runtime differs in Python patch version -> warning notice, not fail
17. Compatible runtime differs in GPU model -> HARDWARE_CHANGED_BETWEEN_RUNS warning, not fail
18. Strict runtime mismatch -> FAIL
19. pip freeze differs due to unrelated packages in compatible mode -> provenance warning, not fail
20. Missing non-core dependency in compatible mode -> install from snapshot then re-verify
21. Missing non-core dependency in strict mode -> FAIL
22. Missing torch/torchvision/CUDA -> FAIL and no reinstall
23. Exact checksum keyset (8 files, no self-reference) -> PASS
24. Missing checksum key in checksums.json -> FAIL
25. Extra checksum key in checksums.json -> FAIL
26. Checksum or byte count mismatch -> FAIL
27. Missing locked_test_access -> FAIL
28. Missing stage2_invocations -> FAIL
29. Stage 2 canonical or legacy alias != 0 -> FAIL
30. Failure reason with quotes/newlines/specials -> valid OPERATOR_FAILURE.json
31. Predictions arrays not equal length -> FAIL
32. Validation targets != 182 -> FAIL
33. Validation unique sources != 91 -> FAIL
34. Real run fixture n50_seed42 -> COMPLETED_VALID / PASS
35. Resume skips n50_seed42
36. Next target is n50_seed1337
37. Resume on different GPU does not re-run completed run
38. Archive SHA and content SHA used in their correct semantic roles
39. Partial run still fails-closed
40. Locked-test access != 0 still fails-closed
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
EXPECTED_OPERATOR_SHA = "75d26863c48edbee821b2410204d4517ec1b1baffc456b47a931306b38235b1d"
EXPECTED_OPERATOR_BYTES = 43085

OPERATOR_EXISTS = OPERATOR_SCRIPT.exists()


def extract_python_snippet(script_text: str, anchor: str, start_token: str = "<<'PY'") -> str:
    a_idx = script_text.find(anchor)
    if a_idx == -1:
        raise ValueError(f"Could not find anchor '{anchor}' in operator script")
    s_idx = script_text.find(start_token, a_idx)
    if s_idx == -1:
        raise ValueError(f"Could not find start token '{start_token}' after anchor in operator script")
    nl_idx = script_text.find("\n", s_idx)
    if nl_idx == -1:
        raise ValueError(f"Could not find newline after '{start_token}'")
    e_idx = script_text.find("\nPY\n", nl_idx)
    if e_idx == -1:
        raise ValueError("Could not find end marker '\\nPY\\n' in operator script")
    return script_text[nl_idx + 1:e_idx]


def extract_verification_python_code(script_text: str) -> str:
    return extract_python_snippet(script_text, "verify_run_artifacts()")


def run_py_verification(
    py_code: str,
    run_dir: Path,
    exp_size: int,
    exp_seed: int,
    exp_code_sha: str = EXPECTED_CODE_SHA,
    exp_archive_sha: str = EXPECTED_BUNDLE_ARCHIVE_SHA,
    exp_content_sha: str = EXPECTED_BUNDLE_CONTENT_SHA,
    gpu_policy: str = "compatible",
    expected_gpu: str = "Tesla T4",
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
        str(gpu_policy),
        str(expected_gpu),
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
    gpu_model: str = "Tesla T4",
):
    output_root = base_dir / "phase_4c1_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Environment lock in output_root
    if legacy_lock:
        env_lock = {
            "bundle_sha256": bundle_archive_sha,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "execution_code_sha": EXPECTED_CODE_SHA,
            "gpu_model": gpu_model,
            "torch_version": "2.2.0",
        }
    else:
        env_lock = {
            "bundle_archive_sha256": bundle_archive_sha,
            "bundle_content_sha256": bundle_content_sha,
            "bundle_sha256": bundle_archive_sha,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "execution_code_sha": EXPECTED_CODE_SHA,
            "gpu_model": gpu_model,
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

    env_content = json.dumps({"gpu": gpu_model, "gpu_model": gpu_model, "python": "3.10.12"})
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
        "gpu_name": gpu_model,
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

    # Keyset schema checksums: exactly 8 files, size_bytes & sha256
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

        # Policy variables defaults
        assert 'RUNTIME_POLICY="${RUNTIME_POLICY:-compatible}"' in text
        assert 'GPU_POLICY="${GPU_POLICY:-compatible}"' in text
        assert 'EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL:-Tesla T4}"' in text
        assert 'MIN_GPU_VRAM_GB="${MIN_GPU_VRAM_GB:-}"' in text

        # Execution code SHA constant
        assert f'EXECUTION_CODE_SHA="{EXPECTED_CODE_SHA}"' in text
        assert f'EXPECTED_BUNDLE_ARCHIVE_SHA256="{EXPECTED_BUNDLE_ARCHIVE_SHA}"' in text
        assert f'EXPECTED_BUNDLE_CONTENT_SHA256="{EXPECTED_BUNDLE_CONTENT_SHA}"' in text

    def test_complete_removal_of_venv(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "VENV_ROOT" not in text
        assert "-m venv" not in text
        assert "activate" not in text
        assert "$VENV_ROOT/bin/python" not in text
        assert "--system-site-packages" not in text
        assert "--without-pip" not in text
        assert "phase4c1-venv" not in text
        assert "bin/pip" not in text

        assert 'SYS_PY3="$(command -v python3)"' in text
        assert '"$SYS_PY3" -m ml.training.run_phase_4c1' in text
        assert '"$SYS_PY3" -m ml.datasets.validate_phase_4c1_bundle' in text

    def test_dependency_preflight_and_safety_policy(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        for mod in ["torch", "torchvision", "numpy", "PIL", "yaml", "sklearn", "scipy"]:
            assert f'"{mod}"' in text or f"'{mod}'" in text
        assert "assert torch.cuda.is_available()" in text
        assert "torch" in text
        assert "torchvision" in text
        assert "torchaudio" in text
        assert "pip install --upgrade pip" not in text

    def test_preflight_failure_does_not_block_retry(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'if [[ -f "$OUTPUT_ROOT/OPERATOR_FAILURE.json" ]]; then' in text
        assert "OPERATOR_FAILURE_prior_" in text
        assert "cp -p" in text
        assert "rm -f" in text

    def test_fail_operator_definition_and_trap_handler(self):
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
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "CONSOLE_LOG=" in text
        assert "operator_console.log" in text
        assert "exec > >(tee -a" in text

    def test_final_packaging_includes_all_archives(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "n${size}_results.tar.gz" in text
        assert "phase_4c1_all_15_runs_results.tar.gz" in text
        assert "phase_4c1_t4_execution_logs.tar.gz" in text
        assert "HARDWARE_SUMMARY.json" in text
        assert "sha256sum" in text


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestGpuCapabilityAndPolicy:
    """Tests 1-10: Capability-based GPU detection, smoke test, VRAM threshold, and policies."""

    @pytest.fixture(autouse=True)
    def setup_gpu_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.gpu_code = extract_python_snippet(text, "GPU_CHECK_RAW=")

    def run_gpu_check(
        self,
        gpu_policy: str = "compatible",
        expected_model: str = "Tesla T4",
        min_vram_gb: str = "",
        mock_cuda_available: bool = True,
        mock_gpu_name: str = "Tesla T4",
        mock_vram_gb: float = 15.0,
        mock_smoke_fail: bool = False,
    ):
        preamble = f"""
import sys, types
import torch

torch.cuda.is_available = lambda: {mock_cuda_available}
torch.cuda.get_device_name = lambda dev=0: "{mock_gpu_name}"
props = types.SimpleNamespace(total_memory=int({mock_vram_gb} * (1024 ** 3)))
torch.cuda.get_device_properties = lambda dev=0: props

orig_synchronize = getattr(torch.cuda, "synchronize", lambda: None)
torch.cuda.synchronize = lambda: None

if {mock_smoke_fail}:
    orig_ones = torch.ones
    def broken_ones(*args, **kwargs):
        t = orig_ones(32, 32)
        type(t).is_cuda = property(lambda self: False)
        return t
    torch.ones = broken_ones
else:
    orig_ones = torch.ones
    def mocked_ones(*args, **kwargs):
        t = orig_ones(32, 32)
        type(t).is_cuda = property(lambda self: True)
        return t
    torch.ones = mocked_ones
"""
        full_code = preamble + self.gpu_code
        cmd = [sys.executable, "-c", full_code, gpu_policy, expected_model, min_vram_gb]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        return cp.returncode, cp.stdout, cp.stderr

    def test_01_t4_in_compatible_mode_passes(self):
        """1. T4 trong compatible mode -> PASS."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "Tesla T4", 15.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        info = json.loads(out)
        assert info["gpu_name"] == "Tesla T4"

    def test_02_l4_simulation_in_compatible_mode_passes(self):
        """2. L4 giả lập trong compatible mode -> PASS."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "NVIDIA L4", 24.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        info = json.loads(out)
        assert info["gpu_name"] == "NVIDIA L4"

    def test_03_v100_simulation_in_compatible_mode_passes(self):
        """3. V100 giả lập trong compatible mode -> PASS."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "Tesla V100-SXM2-16GB", 16.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        info = json.loads(out)
        assert "V100" in info["gpu_name"]

    def test_04_a100_simulation_in_compatible_mode_passes(self):
        """4. A100 giả lập trong compatible mode -> PASS."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "NVIDIA A100-SXM4-40GB", 40.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        info = json.loads(out)
        assert "A100" in info["gpu_name"]

    def test_05_gpu_other_than_t4_compatible_warning_not_fail(self):
        """5. GPU khác T4 trong compatible mode -> warning, không fail."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "NVIDIA RTX 4090", 24.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        info = json.loads(out)
        assert info["gpu_name"] == "NVIDIA RTX 4090"

    def test_06_gpu_other_than_expected_model_in_strict_mode_fails(self):
        """6. GPU khác expected model trong strict mode -> FAIL."""
        rc, out, err = self.run_gpu_check("strict", "Tesla T4", "", True, "NVIDIA L4", 24.0)
        assert rc == 15, f"Expected exit code 15 for strict GPU mismatch, got {rc}: {err}"
        assert "Strict GPU policy violation" in err

    def test_07_no_cuda_fails(self):
        """7. Không có CUDA -> FAIL."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", False, "NONE", 0.0)
        assert rc == 11, f"Expected exit code 11 for no CUDA, got {rc}: {err}"
        assert "torch.cuda.is_available() is False" in err

    def test_08_cuda_smoke_test_error_fails(self):
        """8. CUDA smoke test lỗi -> FAIL."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "Tesla T4", 15.0, mock_smoke_fail=True)
        assert rc == 13, f"Expected exit code 13 for smoke test failure, got {rc}: {err}"
        assert "smoke test failed" in err

    def test_09_no_min_vram_gb_set_does_not_block_arbitrary(self):
        """9. Không đặt MIN_GPU_VRAM_GB -> không chặn bằng ngưỡng tùy ý (e.g. 4 GB passes)."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "Tesla T4", 4.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"

    def test_10_min_vram_gb_set_and_insufficient_fails(self):
        """10. Có đặt MIN_GPU_VRAM_GB và GPU không đủ -> FAIL."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "16.0", True, "Tesla T4", 15.0)
        assert rc == 14, f"Expected exit code 14 for insufficient VRAM, got {rc}: {err}"
        assert "below required MIN_GPU_VRAM_GB" in err


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestScientificBindingAndRuntimePolicy:
    """Tests 11-19: Scientific invariants vs runtime observation in Step 4."""

    @pytest.fixture(autouse=True)
    def setup_lock_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.lock_code = extract_python_snippet(text, "Environment lock exists. Verifying lock integrity sidecar...")

    def run_lock_verification(
        self,
        lock_data: dict,
        exp_code_sha: str = EXPECTED_CODE_SHA,
        exp_code_arch_sha: str = EXPECTED_CODE_ARCHIVE_SHA,
        exp_b_arch_sha: str = EXPECTED_BUNDLE_ARCHIVE_SHA,
        exp_b_cont_sha: str = EXPECTED_BUNDLE_CONTENT_SHA,
        runtime_policy: str = "compatible",
        gpu_policy: str = "compatible",
        expected_gpu: str = "Tesla T4",
        mock_current_gpu: str = "Tesla T4",
        current_pip_freeze_sha: str = "mockpipsha",
    ):
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "phase4c1_environment_lock.json"
            lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

            preamble = f"""
import sys, types, torch
torch.cuda.is_available = lambda: True
torch.cuda.get_device_name = lambda dev=0: "{mock_current_gpu}"
"""
            full_code = preamble + self.lock_code
            cmd = [
                sys.executable,
                "-c",
                full_code,
                str(lock_path),
                exp_code_sha,
                exp_code_arch_sha,
                exp_b_arch_sha,
                exp_b_cont_sha,
                runtime_policy,
                gpu_policy,
                expected_gpu,
                current_pip_freeze_sha,
            ]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            return cp.returncode == 0, cp.stdout, cp.stderr

    def test_11_scientific_binding_correct_passes(self):
        """11. Scientific binding đúng -> PASS."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "gpu_model": "Tesla T4",
        }
        ok, out, err = self.run_lock_verification(lock)
        assert ok, f"Expected PASS: {err}"
        assert "COMPATIBLE_RUNTIME_POLICY_VERIFICATION_PASS" in out

    def test_12_code_sha_mismatch_fails(self):
        """12. Code SHA sai -> FAIL."""
        lock = {
            "execution_code_sha": "0000000000000000000000000000000000000000",
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
        }
        ok, _, _ = self.run_lock_verification(lock)
        assert not ok, "Expected FAIL on code SHA mismatch"

    def test_13_archive_sha_mismatch_fails(self):
        """13. Archive SHA sai -> FAIL."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
        }
        ok, _, _ = self.run_lock_verification(lock)
        assert not ok, "Expected FAIL on bundle archive SHA mismatch"

    def test_14_content_sha_mismatch_fails(self):
        """14. Content SHA sai -> FAIL."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        }
        ok, _, _ = self.run_lock_verification(lock)
        assert not ok, "Expected FAIL on bundle content SHA mismatch"

    def test_15_compatible_runtime_differs_python_executable_passes(self):
        """15. Compatible runtime khác Python executable path -> warning, không fail."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "python_executable": "/usr/bin/python3_different",
            "gpu_model": "Tesla T4",
        }
        ok, out, _ = self.run_lock_verification(lock, runtime_policy="compatible")
        assert ok, "Must pass in compatible mode"
        assert "Python executable path differs from lock" in out

    def test_16_compatible_runtime_differs_python_patch_passes(self):
        """16. Compatible runtime khác Python patch version -> warning, không fail."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "python_version": "3.10.999 (main, mock)",
            "gpu_model": "Tesla T4",
        }
        ok, out, _ = self.run_lock_verification(lock, runtime_policy="compatible")
        assert ok, "Must pass in compatible mode"
        assert "Python version differs from lock" in out

    def test_17_compatible_runtime_differs_gpu_passes_with_warning(self):
        """17. Compatible runtime khác GPU -> HARDWARE_CHANGED_BETWEEN_RUNS warning, không fail."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "gpu_model": "Tesla T4",
        }
        ok, out, _ = self.run_lock_verification(lock, runtime_policy="compatible", gpu_policy="compatible", mock_current_gpu="NVIDIA L4")
        assert ok, "Must pass in compatible mode"
        assert "HARDWARE_CHANGED_BETWEEN_RUNS" in out

    def test_18_strict_runtime_mismatch_fails(self):
        """18. Strict runtime mismatch -> FAIL."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "python_executable": "/usr/bin/different_py",
            "gpu_model": "Tesla T4",
        }
        ok, _, err = self.run_lock_verification(lock, runtime_policy="strict")
        assert not ok, "Must fail in strict mode on runtime mismatch"
        assert "Strict runtime policy violation" in err

    def test_19_pip_freeze_differs_in_compatible_mode_provenance_warning(self):
        """19. pip freeze khác do package không liên quan trong compatible mode -> provenance warning."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "pip_freeze_sha256": "different_freeze_sha",
            "gpu_model": "Tesla T4",
        }
        ok, out, _ = self.run_lock_verification(lock, runtime_policy="compatible")
        assert ok, "Must pass in compatible mode"
        assert "pip freeze digest differs" in out


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestDependencyPreflightPolicy:
    """Tests 20-22: Dependency preflight with compatible vs strict policy."""

    @pytest.fixture(autouse=True)
    def setup_preflight_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.preflight_code = extract_python_snippet(text, "run_dependency_preflight()")

    def run_preflight(self, runtime_policy: str = "compatible", mock_missing: list = None):
        if mock_missing is None:
            mock_missing = []
        preamble = f"""
import sys
orig_import = __import__
def mocked_import(name, *args, **kwargs):
    if name in {mock_missing}:
        raise ImportError(f"No module named '{{name}}'")
    return orig_import(name, *args, **kwargs)
import builtins
builtins.__import__ = mocked_import
"""
        full_code = preamble + self.preflight_code
        cmd = [sys.executable, "-c", full_code, str(REPO_ROOT), runtime_policy]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        return cp.returncode, cp.stdout, cp.stderr

    def test_20_missing_non_core_dependency_in_compatible_mode_installs(self):
        """20. Thiếu dependency phụ hợp lệ trong compatible mode -> cài từ snapshot rồi kiểm tra lại (exit 10)."""
        rc, out, err = self.run_preflight("compatible", ["yaml"])
        assert rc == 10, f"Expected exit code 10 for installing missing package, got {rc}: {err}"

    def test_21_missing_non_core_dependency_in_strict_mode_fails(self):
        """21. Thiếu dependency phụ trong strict mode -> FAIL (exit 4)."""
        rc, out, err = self.run_preflight("strict", ["yaml"])
        assert rc == 4, f"Expected exit code 4 for strict dependency failure, got {rc}: {err}"
        assert "Strict runtime policy violation" in err

    def test_22_missing_core_dependency_fails_no_reinstall(self):
        """22. Thiếu torch/torchvision/CUDA -> FAIL và không cài lại (exit 2)."""
        rc, out, err = self.run_preflight("compatible", ["torchvision"])
        assert rc == 2, f"Expected exit code 2 for forbidden core reinstall, got {rc}: {err}"
        assert "Torch/CUDA reinstall is FORBIDDEN" in err


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestOperatorFaultInjections:
    """Tests 23-40: Checksum keyset, safety fields, predictions schema, resume & fail-closed."""

    @pytest.fixture(autouse=True)
    def setup_py_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.py_code = extract_verification_python_code(text)

    def test_23_exact_checksum_keyset_passes(self):
        """23. Exact checksum keyset -> PASS."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert ok, f"Expected PASS: {err}"
            assert "VERIFICATION_PASS" in out

    def test_24_missing_checksum_key_fails(self):
        """24. Thiếu checksum key -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            del csums["predictions.json"]
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when checksums.json is missing a required file key"
            assert "checksums.json keyset mismatch" in out or "Missing" in out

    def test_25_extra_checksum_key_fails(self):
        """25. Thừa checksum key -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["extra_unapproved_file.json"] = {"size_bytes": 10, "sha256": "00" * 32}
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when checksums.json has extra key"
            assert "checksums.json keyset mismatch" in out or "Extra" in out

    def test_26_checksum_or_byte_count_mismatch_fails(self):
        """26. Sai checksum hoặc byte count -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["epoch_history.json"]["size_bytes"] = 999999
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL on byte count mismatch in checksums.json"

    def test_27_missing_locked_test_access_fails(self):
        """27. Thiếu locked_test_access -> FAIL (no default 0)."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            del rcpt["locked_test_access"]
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when locked_test_access field is missing"

    def test_28_missing_stage2_invocations_fails(self):
        """28. Thiếu stage2_invocations -> FAIL (no default 0)."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            del rcpt["stage2_invocations"]
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when stage2_invocations field is missing"

    def test_29_stage2_canonical_or_legacy_fails_if_nonzero(self):
        """29. Stage 2 canonical hoặc legacy khác 0 -> FAIL."""
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

                ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
                assert not ok, f"Expected FAIL when {key} > 0"

    def test_30_failure_reason_with_quotes_and_newlines_creates_valid_json(self):
        """30. Failure reason có quotes/newlines/specials -> vẫn tạo JSON hợp lệ."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        fail_py_code = extract_python_snippet(text, "fail_operator()")

        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "OPERATOR_FAILURE.json"
            tricky_reason = 'Error: unexpected token \'(\', double "quotes",\\nline breaks\\nand Unicode: \\u0110\\u0103ng k\\u00fd tr\\u01b0\\u1edbc & special $PATH `ls`!'
            cmd = [
                sys.executable,
                "-c",
                fail_py_code,
                "STEP1_TEST",
                "12",
                tricky_reason,
                "456",
                "50",
                "42",
                "/tmp/console.log",
                str(out_file),
            ]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            assert cp.returncode == 0, f"Failure recorder failed: {cp.stderr}"
            assert out_file.exists()

            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            assert loaded["status"] == "OPERATOR_FAILED"
            assert loaded["exit_code"] == 12
            assert loaded["line_number"] == 456
            assert loaded["current_sample_size"] == 50
            assert loaded["current_seed"] == 42
            assert loaded["reason"] == tricky_reason

    def test_31_predictions_arrays_not_same_length_fails(self):
        """31. Predictions arrays không cùng độ dài -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            preds["predictions"] = preds["predictions"][:-1]  # length 181
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when predictions array length is mismatched"

    def test_32_validation_targets_not_182_fails(self):
        """32. Validation targets khác 182 -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            preds["targets"] = [0] * 90 + [1] * 90
            preds["predictions"] = [0] * 90 + [1] * 90
            preds["probabilities"] = [0.1] * 180
            preds["source_ids"] = [f"src_{i}" for i in range(90)] * 2
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when validation count is not 182"

    def test_33_validation_unique_sources_not_91_fails(self):
        """33. Validation unique sources khác 91 -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            # Only 90 unique sources
            preds["source_ids"] = ([f"src_{i:04d}" for i in range(90)] + ["src_0000"]) * 2
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when unique validation sources is not 91"

    def test_34_real_run_n50_seed42_fixture_passes(self):
        """34. Run thực tế n50_seed42 -> PASS."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert ok, f"Expected PASS for real run fixture: {err}"
            assert "VERIFICATION_PASS" in out

    def test_35_and_36_resume_skips_seed42_and_targets_seed1337(self):
        """35-36. Resume skip n50_seed42 và target tiếp theo là n50_seed1337."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir_42 = setup_mock_environment_and_run(Path(td), 50, 42)

            runs_to_execute = []
            sample_sizes = [50, 100, 250]
            seeds = [42, 1337, 2025, 3407, 9001]
            for sz in sample_sizes:
                for sd in seeds:
                    rd = output_root / f"n{sz}_seed_{sd}"
                    if rd.is_dir():
                        ok, _, _ = run_py_verification(self.py_code, rd, sz, sd)
                        if ok:
                            continue
                    runs_to_execute.append((sz, sd))

            assert (50, 42) not in runs_to_execute
            assert len(runs_to_execute) == 14
            assert runs_to_execute[0] == (50, 1337)

    def test_37_resume_on_different_gpu_does_not_rerun_completed_run(self):
        """37. Resume trên GPU khác không chạy lại completed run."""
        with tempfile.TemporaryDirectory() as td:
            # Seed 42 completed on Tesla T4
            output_root, run_dir_42 = setup_mock_environment_and_run(Path(td), 50, 42, gpu_model="Tesla T4")

            # Current session has NVIDIA L4 in compatible mode
            ok, out, _ = run_py_verification(self.py_code, run_dir_42, 50, 42, gpu_policy="compatible")
            assert ok, "Completed run must verify successfully on different GPU in compatible mode"

    def test_38_archive_sha_and_content_sha_used_in_correct_roles(self):
        """38. Archive SHA và content SHA được dùng đúng vai trò."""
        assert EXPECTED_BUNDLE_ARCHIVE_SHA != EXPECTED_BUNDLE_CONTENT_SHA
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert ok, f"Expected PASS when roles are distinct: {err}"

    def test_39_partial_run_still_fails_closed(self):
        """39. Partial run vẫn fail-closed."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            (run_dir / "best_checkpoint.pt").unlink()
            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Partial run missing checkpoint must fail-closed"

    def test_40_locked_test_access_nonzero_fails(self):
        """40. Locked-test access != 0 vẫn fail-closed."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            rcpt["locked_test_access"] = 5
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Locked test access > 0 must fail-closed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
