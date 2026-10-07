"""Tests for Phase 4C.1C.12 Operator Script: Static Invariants, GPU Capability,

Staging, Security, Checksums, and Resume Behavioral Tests.
"""

from __future__ import annotations

import csv
import io
import hashlib
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

pytestmark = pytest.mark.requires_research_artifact

REPO_ROOT = Path(__file__).parents[2]
PEER_ARTIFACTS_DIR = Path(__file__).parents[3] / "forensics-web-lab-local-artifacts"
LOCAL_ARTIFACTS_DIR = Path(os.environ.get("FORENSICS_LOCAL_ARTIFACTS", str(PEER_ARTIFACTS_DIR)))
OPERATOR_SCRIPT = LOCAL_ARTIFACTS_DIR / "phase_4c1" / "t4_transfer" / "phase_4c1_t4_execute_all_stage1.sh"
OPERATOR_SIDECAR = OPERATOR_SCRIPT.with_name(OPERATOR_SCRIPT.name + ".sha256")

EXPECTED_CODE_SHA = "79bb11527d900fd387de1f41f2010c4152b7fea7"
EXPECTED_BUNDLE_ARCHIVE_SHA = "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
EXPECTED_BUNDLE_CONTENT_SHA = "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
EXPECTED_CODE_ARCHIVE_SHA = "5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5"
EXPECTED_CONFIG_HASH = "e03c07dae05a2402416abb0c60480a8cd0a35d060de3bb138a09d0bec0a85dc9"
EXPECTED_REQS_SHA = "850478c0a9746b93354dd756ba428621a4aa48abab4e6aa5fa7e00669cea67a0"

EXPECTED_OPERATOR_SHA = "bae476db0296bd5634ad1002c230f72694de8e0e8cb7372d899165f1fa0759a9"
EXPECTED_OPERATOR_BYTES = 49684

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
    exp_config_hash: str = EXPECTED_CONFIG_HASH,
    bundle_root: Path = None,
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
        str(exp_config_hash),
        str(
            bundle_root
            if bundle_root is not None
            else (run_dir.parent / "bundle" if (run_dir.parent / "bundle").exists() else run_dir.parent.parent / "bundle")
        ),
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
    config_hash: str = EXPECTED_CONFIG_HASH,
    setup_bundle_manifest: bool = True,
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
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "execution_code_sha": EXPECTED_CODE_SHA,
            "gpu_model": gpu_model,
            "torch_version": "2.2.0",
            "config_hash": config_hash,
            "runtime_requirements_sha256": EXPECTED_REQS_SHA,
        }
    lock_file = output_root / "phase4c1_environment_lock.json"
    lock_file.write_text(json.dumps(env_lock, indent=2), encoding="utf-8")
    lock_sha = hashlib.sha256(lock_file.read_bytes()).hexdigest()
    (output_root / "phase4c1_environment_lock.sha256").write_text(
        f"{lock_sha}  phase4c1_environment_lock.json\n", encoding="utf-8"
    )

    # Optional Bundle Manifest
    bundle_root = base_dir / "bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)
    if setup_bundle_manifest:
        manifest_file = bundle_root / "manifest_pilot_a_option_p.csv"
        rows = [
            {"source_id": f"src_{i:04d}", "partition": "inner_validation"}
            for i in range(91)
        ] + [
            {"source_id": f"dev_{i:04d}", "partition": "development_train"}
            for i in range(50)
        ] + [
            {"source_id": f"lock_{i:04d}", "partition": "locked_test"}
            for i in range(30)
        ]
        with open(manifest_file, "w", newline="", encoding="utf-8") as mf:
            writer = csv.DictWriter(mf, fieldnames=["source_id", "partition"])
            writer.writeheader()
            writer.writerows(rows)

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
        "device": "cuda",
        "gpu_name": gpu_model,
        "gpu_vram_gb": 15.0,
        "config_hash": config_hash,
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
        assert "pip install torchvision" not in text

    def test_complete_removal_of_venv(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "-m venv" not in text
        assert "VENV_ROOT" not in text
        assert "VENV_PY" not in text

    def test_dependency_preflight_and_safety_policy(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "Torch/CUDA reinstall is FORBIDDEN" in text
        assert "pip install -r /tmp/phase4c1_missing_packages.txt" in text
        assert 'grep -Ei "^(torch|torchvision|torchaudio|cuda)"' in text

    def test_preflight_failure_does_not_block_retry(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "OPERATOR_FAILURE_prior_" in text
        assert 'cp -p "$OUTPUT_ROOT/OPERATOR_FAILURE.json"' in text

    def test_fail_operator_definition_and_trap_handler(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "fail_operator()" in text
        assert "trap 'fail_operator" in text

    def test_persistent_console_logging_defined(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "operator_console.log" in text
        assert 'exec > >(tee -a "$CONSOLE_LOG") 2>&1' in text

    def test_final_packaging_includes_all_archives(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "n${size}_results.tar.gz" in text
        assert "phase_4c1_all_15_runs_results.tar.gz" in text
        assert "phase_4c1_t4_execution_logs.tar.gz" in text

    def test_policy_validation_in_script(self):
        """14-15. Script validates GPU_POLICY and RUNTIME_POLICY fail-closed."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'case "$GPU_POLICY" in' in text
        assert 'case "$RUNTIME_POLICY" in' in text
        assert 'Invalid GPU_POLICY' in text
        assert 'Invalid RUNTIME_POLICY' in text

    def test_ephemeral_code_staging_defined(self):
        """1-2. Code archive is staged into clean mktemp directory."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'CODE_STAGE_DIR="$(mktemp -d /content/phase4c1-code-79bb115.XXXXXX)"' in text
        assert 'Auditing code archive tar entries for safe extraction' in text

    def test_no_phase_4c1c11_stale_string_in_operator(self):
        """1. Operator script must contain Phase 4C.1C.12 banner and zero Phase 4C.1C.11 strings."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "Phase 4C.1C.11" not in text
        assert "Phase 4C.1C.12 - Autonomous Resumable 15-Run Operator" in text

    def test_gpu_policy_defaults_to_compatible(self):
        """6. Operator script must default GPU_POLICY to compatible."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'GPU_POLICY="${GPU_POLICY:-compatible}"' in text
        assert 'RUNTIME_POLICY="${RUNTIME_POLICY:-compatible}"' in text

    def test_download_dir_is_under_output_root(self):
        """1. Output archives must be saved under persistent OUTPUT_ROOT/download, not ephemeral /content/download."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'DOWNLOAD_DIR="$OUTPUT_ROOT/download"' in text
        assert 'DOWNLOAD_DIR="/content/download"' not in text

    def test_err_trap_inherited_with_set_E(self):
        """4. Shell options must use set -Eeuo pipefail to inherit ERR trap in functions and subshells."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "set -Eeuo pipefail" in text
        assert "set -euo pipefail" not in text
        assert "trap - ERR" in text

    def test_fail_closed_manifest_check_no_fallback(self):
        """3. Exact bundle manifest must be present without cardinality-only fallback."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "if not bundle_manifest_p.exists():" in text
        assert "raise FileNotFoundError" in text
        assert "assert len(set(source_ids)) == 91" not in text

    def test_hardware_summary_fail_closed_no_or_true(self):
        """27. Hardware summary generation is fail-closed, does not ignore errors with || true."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'HARDWARE_SUMMARY.json' in text
        assert "<<'PY' || true" not in text


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestGpuCapabilityAndPolicy:
    """Tests 1-10: GPU capability detection, VRAM check, CUDA smoke test, and strict policy."""

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
        assert "NOTICE: Detected NVIDIA L4 differs from reference Tesla T4" in (out + err)
        info = json.loads(out)
        assert info["gpu_name"] == "NVIDIA L4"

    def test_03_v100_simulation_in_compatible_mode_passes(self):
        """3. V100 giả lập trong compatible mode -> PASS."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "Tesla V100-SXM2-16GB", 16.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        assert "NOTICE: Detected Tesla V100-SXM2-16GB differs from reference Tesla T4" in (out + err)
        info = json.loads(out)
        assert info["gpu_name"] == "Tesla V100-SXM2-16GB"

    def test_04_a100_simulation_in_compatible_mode_passes(self):
        """4. A100 giả lập trong compatible mode -> PASS."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "NVIDIA A100-SXM4-40GB", 40.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        assert "NOTICE: Detected NVIDIA A100-SXM4-40GB differs from reference Tesla T4" in (out + err)
        info = json.loads(out)
        assert info["gpu_name"] == "NVIDIA A100-SXM4-40GB"

    def test_05_gpu_other_than_t4_compatible_warning_not_fail(self):
        """5. GPU khác T4 trong compatible mode -> warning notice, không fail."""
        rc, out, err = self.run_gpu_check("compatible", "Tesla T4", "", True, "NVIDIA RTX 4090", 24.0)
        assert rc == 0, f"Expected 0, got {rc}: {err}"
        assert "NOTICE: Detected NVIDIA RTX 4090 differs from reference Tesla T4" in (out + err)

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

    def test_10b_min_vram_gb_invalid_negative_or_zero_fails(self):
        """17. MIN_GPU_VRAM_GB không hợp lệ (âm, 0, chuỗi) -> FAIL."""
        rc1, _, err1 = self.run_gpu_check("compatible", "Tesla T4", "-5.0", True, "Tesla T4", 15.0)
        assert rc1 == 14
        assert "positive number" in err1

        rc2, _, err2 = self.run_gpu_check("compatible", "Tesla T4", "0", True, "Tesla T4", 15.0)
        assert rc2 == 14

        rc3, _, err3 = self.run_gpu_check("compatible", "Tesla T4", "invalid_string", True, "Tesla T4", 15.0)
        assert rc3 == 14

    def test_10c_strict_a100_expected_and_actual_passes_no_compatible_msg(self):
        """18. Strict A100 expected + actual A100 -> PASS và không in compatible message."""
        rc, out, err = self.run_gpu_check("strict", "NVIDIA A100-SXM4-40GB", "", True, "NVIDIA A100-SXM4-40GB", 40.0)
        assert rc == 0, f"Expected PASS: {err}"
        assert "STRICT_GPU_POLICY_PASS" in (out + err)
        assert "Continuing under GPU_POLICY=compatible" not in (out + err)


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
        staged_req_sha: str = EXPECTED_REQS_SHA,
        staged_cfg_sha: str = EXPECTED_CONFIG_HASH,
        baseline_freeze_content: str = None,
        baseline_freeze_filename: str = None,
    ):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td)
            lock_path = output_root / "phase4c1_environment_lock.json"
            lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

            if baseline_freeze_content is not None:
                fname = baseline_freeze_filename or lock_data.get("baseline_pip_freeze_file") or "baseline-pip-freeze.txt"
                bff = output_root / fname
                bff.write_bytes(baseline_freeze_content.encode("utf-8"))

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
                staged_req_sha,
                staged_cfg_sha,
                str(output_root),
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

    def test_14b_requirements_hash_mismatch_fails(self):
        """10. Requirements hash sai -> FAIL trong cả compatible và strict."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "runtime_requirements_sha256": "bad_reqs_sha",
        }
        ok_comp, _, err_comp = self.run_lock_verification(lock, runtime_policy="compatible")
        assert not ok_comp, "Must fail on requirements hash mismatch in compatible mode"
        assert "Requirements SHA256 mismatch" in err_comp

        ok_strict, _, err_strict = self.run_lock_verification(lock, runtime_policy="strict")
        assert not ok_strict, "Must fail on requirements hash mismatch in strict mode"
        assert "Requirements SHA256 mismatch" in err_strict

    def test_14c_config_hash_mismatch_fails(self):
        """7. Config hash sai -> FAIL."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "config_hash": "bad_config_hash",
        }
        ok, _, err = self.run_lock_verification(lock, runtime_policy="compatible")
        assert not ok, "Must fail on config hash mismatch"
        assert "Config hash mismatch" in err

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
            "python_version": "3.10.999 (default, Jan 1 2026)",
            "gpu_model": "Tesla T4",
        }
        ok, out, _ = self.run_lock_verification(lock, runtime_policy="compatible")
        assert ok, "Must pass in compatible mode"
        assert "Python version differs from lock" in out

    def test_17_compatible_runtime_differs_gpu_passes_with_warning(self):
        """17. Compatible runtime khác GPU -> warning, không fail."""
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
        sample_freeze = "torch==2.2.0\nnumpy==1.26.0\n"
        freeze_sha = hashlib.sha256(sample_freeze.encode("utf-8")).hexdigest()
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "pip_freeze_sha256": freeze_sha,
            "gpu_model": "Tesla T4",
        }
        # Run with current session freeze different from baseline
        ok, out, _ = self.run_lock_verification(
            lock,
            runtime_policy="compatible",
            current_pip_freeze_sha="different_session_freeze_sha",
            baseline_freeze_content=sample_freeze,
        )
        assert ok, "Must pass in compatible mode"
        assert "pip freeze digest differs" in out

    def test_19b_legacy_baseline_pip_freeze_matches_passes(self):
        """21. Legacy baseline pip freeze (t4-pip-freeze.txt) khớp lock -> PASS."""
        sample_freeze = "torch==2.2.0\nnumpy==1.26.0\n"
        freeze_sha = hashlib.sha256(sample_freeze.encode("utf-8")).hexdigest()
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "pip_freeze_sha256": freeze_sha,
            "gpu_model": "Tesla T4",
        }
        ok, out, err = self.run_lock_verification(
            lock,
            baseline_freeze_content=sample_freeze,
            baseline_freeze_filename="t4-pip-freeze.txt",
        )
        assert ok, f"Expected PASS when legacy baseline freeze matches lock: {err}"
        assert "Baseline pip freeze verified" in out

    def test_19c_legacy_baseline_pip_freeze_mismatch_fails(self):
        """22. Legacy baseline pip freeze sai hash -> FAIL."""
        sample_freeze = "torch==2.2.0\nnumpy==1.26.0\n"
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "pip_freeze_sha256": "00" * 32,
            "gpu_model": "Tesla T4",
        }
        ok, _, err = self.run_lock_verification(
            lock,
            baseline_freeze_content=sample_freeze,
            baseline_freeze_filename="t4-pip-freeze.txt",
        )
        assert not ok, "Expected FAIL when baseline freeze file doesn't match lock sha"
        assert "Provenance corrupted" in err

    def test_19d_lock_has_freeze_sha_but_baseline_file_missing_fails(self):
        """5. Fail-closed: Lock có pip_freeze_sha256 nhưng không tìm thấy file baseline -> FAIL."""
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "pip_freeze_sha256": "aabb" * 16,
            "gpu_model": "Tesla T4",
        }
        # Notice: baseline_freeze_content is None, so neither t4-pip-freeze.txt nor baseline-pip-freeze.txt exists
        ok, _, err = self.run_lock_verification(lock, baseline_freeze_content=None)
        assert not ok, "Expected FAIL when baseline pip freeze file is missing"
        assert "Baseline pip freeze file required by environment lock not found" in err

    def test_19e_new_baseline_pip_freeze_matches_passes(self):
        """5. New session baseline pip freeze (baseline-pip-freeze.txt declared in lock) -> PASS."""
        sample_freeze = "torch==2.2.0\nscipy==1.12.0\n"
        freeze_sha = hashlib.sha256(sample_freeze.encode("utf-8")).hexdigest()
        lock = {
            "execution_code_sha": EXPECTED_CODE_SHA,
            "code_archive_sha256": EXPECTED_CODE_ARCHIVE_SHA,
            "bundle_archive_sha256": EXPECTED_BUNDLE_ARCHIVE_SHA,
            "bundle_content_sha256": EXPECTED_BUNDLE_CONTENT_SHA,
            "baseline_pip_freeze_file": "baseline-pip-freeze.txt",
            "pip_freeze_sha256": freeze_sha,
            "gpu_model": "Tesla T4",
        }
        ok, out, err = self.run_lock_verification(
            lock,
            baseline_freeze_content=sample_freeze,
            baseline_freeze_filename="baseline-pip-freeze.txt",
        )
        assert ok, f"Expected PASS when baseline-pip-freeze.txt matches lock: {err}"
        assert "Baseline pip freeze verified: baseline-pip-freeze.txt" in out


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestDependencyPreflightPolicy:
    """Tests 20-22: Dependency preflight with compatible vs strict policy."""

    @pytest.fixture(autouse=True)
    def setup_preflight_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.preflight_code = extract_python_snippet(text, "run_dependency_preflight()")

    def run_preflight(self, runtime_policy: str, missing_mods: list[str]):
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            ml_dir = repo_root / "ml"
            ml_dir.mkdir(parents=True, exist_ok=True)
            req_file = ml_dir / "requirements.txt"
            req_file.write_text("torch==2.2.0\ntorchvision==0.17.0\npyyaml==6.0.1\npillow==10.2.0\n", encoding="utf-8")

            simulated_missing = set(missing_mods)
            preamble = f"""
import sys, builtins
orig_import = builtins.__import__
simulated_missing = {repr(simulated_missing)}
def selective_import(name, *args, **kwargs):
    if name in simulated_missing:
        raise ImportError(f"Simulated missing module: {{name}}")
    return orig_import(name, *args, **kwargs)
builtins.__import__ = selective_import
"""
            full_code = preamble + self.preflight_code
            cmd = [sys.executable, "-c", full_code, str(repo_root), runtime_policy]
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
class TestCodeStagingAndSecurity:
    """Tests 3-5: Code archive validation for path traversal and absolute paths."""

    @pytest.fixture(autouse=True)
    def setup_security_code(self):
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        self.sec_code = extract_python_snippet(text, "Auditing code archive tar entries for safe extraction")

    def run_tar_audit(self, archive_path: Path):
        cmd = [sys.executable, "-c", self.sec_code, str(archive_path)]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        return cp.returncode == 0, cp.stdout, cp.stderr

    def test_safe_archive_passes(self):
        with tempfile.TemporaryDirectory() as td:
            tar_p = Path(td) / "safe.tar.gz"
            with tarfile.open(tar_p, "w:gz") as tf:
                f_p = Path(td) / "test.txt"
                f_p.write_text("hello", encoding="utf-8")
                tf.add(f_p, arcname="ml/test.txt")
            ok, _, _ = self.run_tar_audit(tar_p)
            assert ok

    def test_absolute_path_in_tar_fails(self):
        """3. TAR có absolute path -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            tar_p = Path(td) / "bad_abs.tar.gz"
            with tarfile.open(tar_p, "w:gz") as tf:
                ti = tarfile.TarInfo(name="/etc/passwd")
                ti.size = len(b"hello")
                tf.addfile(ti, io.BytesIO(b"hello"))
            ok, _, err = self.run_tar_audit(tar_p)
            assert not ok
            assert "Absolute path" in err

    def test_path_traversal_in_tar_fails(self):
        """4. TAR có .. traversal -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            tar_p = Path(td) / "bad_trav.tar.gz"
            with tarfile.open(tar_p, "w:gz") as tf:
                f_p = Path(td) / "test.txt"
                f_p.write_text("hello", encoding="utf-8")
                tf.add(f_p, arcname="../escape.txt")
            ok, _, err = self.run_tar_audit(tar_p)
            assert not ok
            assert "Path traversal" in err


@pytest.mark.skipif(not OPERATOR_EXISTS, reason="Local operator script not found")
class TestOperatorFaultInjections:
    """Tests 23-41: Checksum keyset, safety fields, predictions schema, resume & fail-closed."""

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

    def test_23b_checksum_dictionary_valid_passes(self):
        """11. Checksum dictionary đúng -> PASS."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert ok, f"Expected PASS for valid checksums: {err}"

    def test_23c_string_checksum_fails(self):
        """12. String checksum -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            # Replace dictionary entry with legacy string hash
            csums["predictions.json"] = "00" * 32
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when checksum entry is string instead of dictionary"
            assert "must be a dict" in out

    def test_23d_checksum_dictionary_missing_or_extra_field_fails(self):
        """13. Checksum dictionary thừa/thiếu field -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            # Missing sha256 field
            csums["predictions.json"] = {"size_bytes": 100}
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when checksum dict is missing sha256"
            assert "Invalid checksum entry keys" in out

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
            assert not ok, "Expected FAIL when recorded size doesn't match"
            assert "Size mismatch" in out

    def test_27_missing_locked_test_access_fails(self):
        """27. Thiếu locked_test_access -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            del rcpt["locked_test_access"]
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["size_bytes"] = rcpt_file.stat().st_size
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when locked_test_access is missing"

    def test_28_missing_stage2_invocations_fails(self):
        """28. Thiếu stage2_invocations -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            del rcpt["stage2_invocations"]
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["size_bytes"] = rcpt_file.stat().st_size
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when stage2_invocations is missing"

    def test_29_stage2_canonical_or_legacy_fails_if_nonzero(self):
        """29. Stage 2 canonical hoặc legacy khác 0 -> FAIL."""
        for field in ["stage2_invocations", "stage_2_invocation"]:
            with tempfile.TemporaryDirectory() as td:
                output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
                rcpt_file = run_dir / "run_receipt.json"
                rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
                rcpt[field] = 1
                rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
                csums_file = run_dir / "checksums.json"
                csums = json.loads(csums_file.read_text(encoding="utf-8"))
                csums["run_receipt.json"]["size_bytes"] = rcpt_file.stat().st_size
                csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
                csums_file.write_text(json.dumps(csums), encoding="utf-8")

                ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
                assert not ok, f"Expected FAIL when {field} is non-zero"

    def test_30_failure_reason_with_quotes_and_newlines_creates_valid_json(self):
        """30. Failure reason co quotes/newlines -> van tao JSON hop le."""
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "OPERATOR_FAILURE.json"
            complex_reason = 'Error in "step 1": line with quotes and\nnewline, plus special chars: \t < > `'
            py_code = (
                "import json, sys\n"
                "from datetime import datetime, timezone\n"
                "state, code, reason, line, size_str, seed_str, log_path, out_path = sys.argv[1:9]\n"
                "failure_doc = {\n"
                '    "status": "failed",\n'
                '    "current_state": state,\n'
                '    "exit_code": int(code) if code.isdigit() else 1,\n'
                '    "line_number": int(line) if line.isdigit() else None,\n'
                '    "failed_command": reason,\n'
                '    "reason": reason,\n'
                '    "current_sample_size": int(size_str) if size_str.isdigit() else None,\n'
                '    "current_seed": int(seed_str) if seed_str.isdigit() else None,\n'
                '    "console_log_path": log_path,\n'
                '    "timestamp_utc": datetime.now(timezone.utc).isoformat()\n'
                "}\n"
                'with open(out_path, "w", encoding="utf-8") as f:\n'
                "    json.dump(failure_doc, f, indent=2)\n"
            )
            cmd = [
                sys.executable,
                "-c",
                py_code,
                "PREFLIGHT", "1", complex_reason, "42", "50", "42", "/path/to/log", str(out_file)
            ]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            assert cp.returncode == 0
            assert out_file.exists()
            data = json.loads(out_file.read_text(encoding="utf-8"))
            assert data["reason"] == complex_reason

    def test_31_predictions_arrays_not_same_length_fails(self):
        """31. Predictions arrays không cùng độ dài -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            preds["predictions"] = preds["predictions"][:-1]  # 181 instead of 182
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when prediction arrays mismatch in length"

    def test_32_validation_targets_not_182_fails(self):
        """32. Validation targets khác 182 -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            for k in ["targets", "predictions", "probabilities", "source_ids"]:
                preds[k] = preds[k][:100]  # Only 100 entries
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
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
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when unique validation sources is not 91"

    def test_33b_prediction_sources_mismatch_bundle_manifest_fails(self):
        """24-25. Prediction source IDs mismatch bundle manifest inner_validation -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, setup_bundle_manifest=True)
            bundle_root = Path(td) / "bundle"
            # In predictions.json, change one source ID to unapproved source
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            preds["source_ids"][0] = "unapproved_src_9999"
            preds["source_ids"][91] = "unapproved_src_9999"
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42, bundle_root=bundle_root)
            assert not ok, "Expected FAIL when prediction sources don't match bundle manifest inner_validation"
            assert "Predictions source IDs mismatch" in (out + err)

    def test_33c_development_source_in_predictions_fails(self):
        """26. Development train source xuất hiện trong predictions -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, setup_bundle_manifest=True)
            bundle_root = Path(td) / "bundle"
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            preds["source_ids"][0] = "dev_0000"
            preds["source_ids"][91] = "dev_0000"
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, _ = run_py_verification(self.py_code, run_dir, 50, 42, bundle_root=bundle_root)
            assert not ok, "Expected FAIL when development source appears in validation predictions"

    def test_33d_config_hash_in_receipt_mismatch_expected_fails(self):
        """8. Run receipt config hash sai expected -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, config_hash="different_config_hash")
            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok, "Expected FAIL when receipt config_hash does not match expected_config_hash"

    def test_34_real_run_n50_seed42_fixture_passes(self):
        """28. Run thực tế n50_seed42 -> COMPLETED_VALID."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42)
            assert ok, f"Expected PASS for real run fixture: {err}"
            assert "VERIFICATION_PASS" in out

    def test_35_and_36_resume_skips_seed42_and_targets_seed1337(self):
        """29-30. Resume skip n50_seed42 và target tiếp theo là n50_seed1337."""
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
        """31 & 37. Resume trên GPU khác không chạy lại completed run và không sửa artifact."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir_42 = setup_mock_environment_and_run(Path(td), 50, 42, gpu_model="Tesla T4")
            # Snapshot artifact timestamps
            mtimes_before = {p.name: p.stat().st_mtime for p in run_dir_42.glob("*")}

            # Verifier running under compatible mode with expected_gpu="Tesla T4"
            ok, _, _ = run_py_verification(self.py_code, run_dir_42, 50, 42, gpu_policy="compatible")
            assert ok

            mtimes_after = {p.name: p.stat().st_mtime for p in run_dir_42.glob("*")}
            assert mtimes_before == mtimes_after, "Completed run artifacts must not be modified during resume check"

    def test_38_archive_sha_and_content_sha_used_in_correct_roles(self):
        """38. Archive SHA và content SHA được dùng đúng vai trò."""
        with tempfile.TemporaryDirectory() as td:
            # If content SHA passed where archive SHA expected -> FAIL
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            ok, _, _ = run_py_verification(
                self.py_code, run_dir, 50, 42,
                exp_archive_sha=EXPECTED_BUNDLE_CONTENT_SHA,
            )
            assert not ok, "Must fail if content SHA passed in place of archive SHA"

    def test_39_partial_run_still_fails_closed(self):
        """39. Partial run vẫn fail-closed."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            (run_dir / "best_checkpoint.pt").unlink()
            ok, out, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok
            assert "Required file missing: best_checkpoint.pt" in out

    def test_40_locked_test_access_nonzero_fails(self):
        """40. Locked-test access khác 0 vẫn fail-closed."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42)
            rcpt_file = run_dir / "run_receipt.json"
            rcpt = json.loads(rcpt_file.read_text(encoding="utf-8"))
            rcpt["locked_test_access"] = 1
            rcpt_file.write_text(json.dumps(rcpt), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["run_receipt.json"]["size_bytes"] = rcpt_file.stat().st_size
            csums["run_receipt.json"]["sha256"] = hashlib.sha256(rcpt_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, _, _ = run_py_verification(self.py_code, run_dir, 50, 42)
            assert not ok

    def test_41_hardware_summary_failure_fails_closed(self):
        """27. Hardware summary lỗi -> operator không tuyên bố complete."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'fail_operator "PACKAGING" 1 "Failed to generate HARDWARE_SUMMARY.json' in text


    def test_33e_missing_bundle_manifest_fails(self):
        """3. Missing canonical bundle manifest -> FAIL closed (no fallback)."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, setup_bundle_manifest=False)
            bundle_root = Path(td) / "bundle"
            # Explicitly ensure manifest is absent
            manifest_file = bundle_root / "manifest_pilot_a_option_p.csv"
            if manifest_file.exists():
                manifest_file.unlink()

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42, bundle_root=bundle_root)
            assert not ok, "Expected FAIL when canonical bundle manifest is missing"
            assert "Canonical bundle manifest not found" in (out + err)

    def test_33f_corrupt_manifest_schema_fails(self):
        """3. Corrupt bundle manifest schema -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, setup_bundle_manifest=False)
            bundle_root = Path(td) / "bundle"
            manifest_file = bundle_root / "manifest_pilot_a_option_p.csv"
            manifest_file.write_text("wrong_col1,wrong_col2\nval1,val2\n", encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42, bundle_root=bundle_root)
            assert not ok, "Expected FAIL when bundle manifest schema is corrupt"

    def test_33g_locked_test_source_in_predictions_fails(self):
        """3. Locked-test partition source appearing in predictions -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, setup_bundle_manifest=True)
            bundle_root = Path(td) / "bundle"
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            preds["source_ids"][0] = "lock_0000"
            preds["source_ids"][91] = "lock_0000"
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42, bundle_root=bundle_root)
            assert not ok, "Expected FAIL when locked-test source appears in validation predictions"

    def test_33h_cardinality_91_but_wrong_membership_fails(self):
        """3. Correct cardinality (91 unique sources) but wrong membership -> FAIL."""
        with tempfile.TemporaryDirectory() as td:
            output_root, run_dir = setup_mock_environment_and_run(Path(td), 50, 42, setup_bundle_manifest=True)
            bundle_root = Path(td) / "bundle"
            preds_file = run_dir / "predictions.json"
            preds = json.loads(preds_file.read_text(encoding="utf-8"))
            # Replace valid src_0000 with synthetic src_unregistered
            preds["source_ids"] = [
                "src_unregistered" if s == "src_0000" else s
                for s in preds["source_ids"]
            ]
            assert len(set(preds["source_ids"])) == 91
            preds_file.write_text(json.dumps(preds), encoding="utf-8")
            csums_file = run_dir / "checksums.json"
            csums = json.loads(csums_file.read_text(encoding="utf-8"))
            csums["predictions.json"]["size_bytes"] = preds_file.stat().st_size
            csums["predictions.json"]["sha256"] = hashlib.sha256(preds_file.read_bytes()).hexdigest()
            csums_file.write_text(json.dumps(csums), encoding="utf-8")

            ok, out, err = run_py_verification(self.py_code, run_dir, 50, 42, bundle_root=bundle_root)
            assert not ok, "Expected FAIL when membership does not exactly match inner_validation"
            assert "Predictions source IDs mismatch" in (out + err)

    def test_42_subshell_and_function_error_caught_by_E_trap(self, tmp_path):
        """4. Behavioral test: set -Eeuo pipefail ensures errors in functions or subshells trigger ERR trap."""
        failure_json = tmp_path / "OPERATOR_FAILURE.json"

        script = """set -Eeuo pipefail
SYS_PY=$(command -v python3 || command -v python)
fail_operator() {
    trap - ERR
    local state="$1"
    local code="$2"
    local reason="$3"
    local line="${4:-$LINENO}"
    "$SYS_PY" -c '
import sys, json
state, code, reason, line, out_p = sys.argv[1:6]
with open(out_p, "w") as f:
    json.dump({"status": "failed", "state": state, "code": int(code), "reason": reason, "line": int(line)}, f)
' "$state" "$code" "$reason" "$line" "OPERATOR_FAILURE.json"
    exit "$code"
}

trap 'fail_operator "SUBSHELL_TEST" $? "$BASH_COMMAND" $LINENO' ERR

inner_func() {
    ( exit 42 )
}

inner_func
""".replace("\r\n", "\n").encode("utf-8")

        cp = subprocess.run(["bash", "-s"], input=script, capture_output=True, cwd=str(tmp_path))
        assert cp.returncode == 42
        assert failure_json.exists()
        doc = json.loads(failure_json.read_text(encoding="utf-8"))
        assert doc["status"] == "failed"
        assert doc["code"] == 42
        assert doc["state"] == "SUBSHELL_TEST"

    def test_43_download_dir_is_persistent_output(self):
        """1. Operator script defines DOWNLOAD_DIR under persistent OUTPUT_ROOT."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert 'OUTPUT_ROOT="/content/phase_4c1_outputs"' in text
        assert 'DOWNLOAD_DIR="$OUTPUT_ROOT/download"' in text
        assert 'mkdir -p "$OUTPUT_ROOT" "$DOWNLOAD_DIR" "$LOGS_DIR"' in text
        assert '$DOWNLOAD_DIR/n${size}_results.tar.gz' in text
        assert '$DOWNLOAD_DIR/phase_4c1_all_15_runs_results.tar.gz' in text
        assert '$DOWNLOAD_DIR/phase_4c1_t4_execution_logs.tar.gz' in text

class TestPhase4C1C13PythonQuotingAndPreflightRegression:
    """Tests 44-48: Phase 4C.1C.13 Colab Python quoting hotfix and real preflight regression."""

    @pytest.fixture(autouse=True)
    def setup_operator(self):
        assert OPERATOR_SCRIPT.exists(), f"Operator script missing: {OPERATOR_SCRIPT}"
        self.text = OPERATOR_SCRIPT.read_text(encoding="utf-8")

        # Extract model and vram parser codes
        m_model = re.search(
            r'''ACTUAL_GPU_MODEL=\$\(["']?\$SYS_PY3["']?\s+-c\s+['"](.*?)['"]\s+"\$GPU_CHECK_RAW"\)''',
            self.text,
        )
        assert m_model is not None, "Could not find ACTUAL_GPU_MODEL parser in operator"
        self.model_code = m_model.group(1)

        m_vram = re.search(
            r'''ACTUAL_GPU_VRAM_GB=\$\(\s*["']?\$SYS_PY3["']?\s+-c\s+\\\s+['"](.*?)['"]\s+\\\s+"\$GPU_CHECK_RAW"\s*\)''',
            self.text,
        )
        assert m_vram is not None, "Could not find ACTUAL_GPU_VRAM_GB parser in operator"
        self.vram_code = m_vram.group(1)

    def test_44_gpu_json_parser_real_execution_t4(self):
        """1. Real execution of GPU JSON parser with T4 fixture returns model and 14.56 VRAM."""
        fixture_t4 = json.dumps({"gpu_name": "Tesla T4", "vram_gb": 14.56317138671875})
        res_m = subprocess.run([sys.executable, "-c", self.model_code, fixture_t4], capture_output=True, text=True)
        res_v = subprocess.run([sys.executable, "-c", self.vram_code, fixture_t4], capture_output=True, text=True)

        assert res_m.returncode == 0, f"Model parser failed with code {res_m.returncode}: {res_m.stderr}"
        assert res_m.stderr == "", f"Model parser produced unexpected stderr: {res_m.stderr}"
        assert res_m.stdout.strip() == "Tesla T4"

        assert res_v.returncode == 0, f"VRAM parser failed with code {res_v.returncode}: {res_v.stderr}"
        assert res_v.stderr == "", f"VRAM parser produced unexpected stderr: {res_v.stderr}"
        assert res_v.stdout.strip() == "14.56"
        assert "SyntaxError" not in res_v.stderr

    def test_45_gpu_json_parser_real_execution_l4(self):
        """2. Real execution of GPU JSON parser with L4 fixture returns model and 22.00 VRAM."""
        fixture_l4 = json.dumps({"gpu_name": "NVIDIA L4", "vram_gb": 22.0})
        res_m = subprocess.run([sys.executable, "-c", self.model_code, fixture_l4], capture_output=True, text=True)
        res_v = subprocess.run([sys.executable, "-c", self.vram_code, fixture_l4], capture_output=True, text=True)

        assert res_m.returncode == 0, f"Model parser failed: {res_m.stderr}"
        assert res_m.stderr == ""
        assert res_m.stdout.strip() == "NVIDIA L4"

        assert res_v.returncode == 0, f"VRAM parser failed: {res_v.stderr}"
        assert res_v.stderr == ""
        assert res_v.stdout.strip() == "22.00"
        assert "SyntaxError" not in res_v.stderr

    def test_46_gpu_json_parser_invalid_vram_fails_explicitly(self):
        """3. Invalid vram_gb values fail explicitly with non-zero exit code and error message."""
        invalid_fixtures = [
            json.dumps({"gpu_name": "Tesla T4", "vram_gb": "not_a_number"}),
            json.dumps({"gpu_name": "Tesla T4", "vram_gb": None}),
            json.dumps({"gpu_name": "Tesla T4"}),
        ]
        for fixture in invalid_fixtures:
            res = subprocess.run([sys.executable, "-c", self.vram_code, fixture], capture_output=True, text=True)
            assert res.returncode != 0, f"Expected non-zero return code for {fixture}, got {res.returncode}"
            assert res.stderr != "", "Expected error output in stderr"

    def test_47_scan_and_execute_all_operator_python_c_helpers(self):
        """4. Scan and execute all standalone python -c helpers in operator using suitable fixtures."""
        py_c_pattern = re.compile(r"""["']?\$SYS_PY3["']?\s+-c\s+(?:\\\s+)?(?:'([^']*)'|"([^"]*)")""", re.DOTALL)
        matches = list(py_c_pattern.finditer(self.text))
        assert len(matches) >= 4, f"Expected at least 4 python -c helpers, found {len(matches)}"

        fixture_gpu = json.dumps({"gpu_name": "Tesla T4", "vram_gb": 14.56317138671875})
        for idx, match in enumerate(matches, 1):
            code = match.group(1) if match.group(1) is not None else match.group(2)
            # Ensure no escaped quotes in single-quoted bash strings
            assert r'\"' not in code, f"Helper {idx} contains forbidden escaped quote \\\": {code}"

            if "gpu_name" in code or "vram_gb" in code:
                cp = subprocess.run([sys.executable, "-c", code, fixture_gpu], capture_output=True, text=True)
                assert cp.returncode == 0, f"Helper {idx} failed: {cp.stderr}"
                assert cp.stderr == ""
            elif "bundle_sha256" in code:
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
                    json.dump({"bundle_sha256": "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"}, f)
                    tmp_p = f.name
                try:
                    cp = subprocess.run([sys.executable, "-c", code, tmp_p], capture_output=True, text=True)
                    assert cp.returncode == 0, f"Helper {idx} failed: {cp.stderr}"
                    assert cp.stdout.strip() == "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
                finally:
                    Path(tmp_p).unlink()
            elif "datetime" in code:
                cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                assert cp.returncode == 0, f"Helper {idx} failed: {cp.stderr}"
                assert re.match(r"^\d{8}_\d{6}Z$", cp.stdout.strip())
            else:
                cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                assert cp.returncode == 0, f"Helper {idx} failed: {cp.stderr}"

    def test_48_preflight_regression_gpu_validation_log(self):
        """5. Preflight regression executes through GPU JSON generation, model/vram parsing, and logs Validated GPU."""
        fixture_t4 = json.dumps({"gpu_name": "Tesla T4", "vram_gb": 14.56317138671875})
        bash_script = f"""set -euo pipefail
SYS_PY3=$(command -v python3 || command -v python)
GPU_CHECK_RAW='{fixture_t4}'
ACTUAL_GPU_MODEL=$("$SYS_PY3" -c '{self.model_code}' "$GPU_CHECK_RAW")
ACTUAL_GPU_VRAM_GB=$(
    "$SYS_PY3" -c \
    '{self.vram_code}' \
    "$GPU_CHECK_RAW"
)
echo "[+] Validated GPU: $ACTUAL_GPU_MODEL ($ACTUAL_GPU_VRAM_GB GB VRAM)"
""".replace("\r\n", "\n").encode("utf-8")

        cp_bash = subprocess.run(["bash", "-s"], input=bash_script, capture_output=True)
        out_bash = cp_bash.stdout.decode("utf-8", errors="replace")
        err_bash = cp_bash.stderr.decode("utf-8", errors="replace")
        assert cp_bash.returncode == 0, f"Preflight bash failed ({cp_bash.returncode}): {err_bash}"
        assert "Validated GPU: Tesla T4 (14.56 GB VRAM)" in out_bash

    def test_operator_invokes_reusable_n250_validator_mode(self):
        """Phase 4C.1C.14: Operator invokes validate_learning_curve_bundle with --reusable-n250 in Step 3."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")

        # 1. Operator calls module ml.datasets.validate_learning_curve_bundle
        assert "ml.datasets.validate_learning_curve_bundle" in text

        # 2. Exactly passes --bundle "$BUNDLE_ROOT" and --reusable-n250
        assert '--bundle "$BUNDLE_ROOT"' in text
        assert "--reusable-n250" in text

        # 3. No canonical validator invocation lacks --reusable-n250
        assert 'validate_learning_curve_bundle.py --bundle "$BUNDLE_ROOT"\n' not in text
        assert 'validate_learning_curve_bundle.py --bundle "$BUNDLE_ROOT"' not in text

        # 4. Check real parser via subprocess --help
        cp_help = subprocess.run(
            [sys.executable, "-m", "ml.datasets.validate_learning_curve_bundle", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert cp_help.returncode == 0
        assert "--bundle" in cp_help.stdout
        assert "--reusable-n250" in cp_help.stdout

        # 5. If canonical reusable bundle exists locally, run behavioral validation test
        local_bundle = LOCAL_ARTIFACTS_DIR / "phase_4c1" / "t4_transfer" / "reusable_n250_bundle"
        if local_bundle.exists() and (local_bundle / "bundle_receipt.json").exists():
            cp_val = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ml.datasets.validate_learning_curve_bundle",
                    "--bundle",
                    str(local_bundle),
                    "--reusable-n250",
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            assert cp_val.returncode == 0, f"Validator failed: {cp_val.stderr}"
            assert "ALL VALIDATIONS PASSED" in cp_val.stdout
            assert "Locked-test rows/sources/files: ZERO" in cp_val.stdout
            assert "Cross-partition overlap: ZERO" in cp_val.stdout
            assert "Cohort cardinalities: 50 / 100 / 250 nested + 91 val" in cp_val.stdout
            assert "0 source ID mismatches" in cp_val.stdout

    def test_bundle_content_sha_variable_defined_before_use(self):
        """Phase 4C.1C.15: BUNDLE_CONTENT_SHA256 assigned before any use and EXTRACTED_CONTENT_SHA removed."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        assert "EXTRACTED_CONTENT_SHA" not in text

        assign_idx = text.find("BUNDLE_CONTENT_SHA256=")
        assert assign_idx != -1, "BUNDLE_CONTENT_SHA256= assignment not found"

        ref_matches = [m.start() for m in re.finditer(r"\$\{?BUNDLE_CONTENT_SHA256\}?", text)]
        assert len(ref_matches) > 0, "No references to BUNDLE_CONTENT_SHA256 found"
        for ref_pos in ref_matches:
            assert ref_pos > assign_idx, f"Reference at {ref_pos} occurs before assignment at {assign_idx}"

        assert r"^[0-9a-f]{64}$" in text
        assert '"$BUNDLE_CONTENT_SHA256" != "$EXPECTED_BUNDLE_CONTENT_SHA256"' in text

    def test_bundle_content_sha_real_execution_valid_fixture(self):
        """Phase 4C.1C.15: Real bash execution with valid bundle_receipt fixture succeeds with content hash."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
            json.dump({"bundle_sha256": "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"}, f)
            receipt_path = f.name
        try:
            bash_script = f"""set -Eeuo pipefail
SYS_PY3=$(command -v python3 || command -v python)
BUNDLE_RECEIPT="{receipt_path.replace(chr(92), '/')}"
BUNDLE_RECEIPT=$(wslpath -u "$BUNDLE_RECEIPT" 2>/dev/null || echo "$BUNDLE_RECEIPT")
EXPECTED_BUNDLE_CONTENT_SHA256="c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"

BUNDLE_CONTENT_SHA256=$(
    "$SYS_PY3" -c \\
    'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("bundle_sha256", ""))' \\
    "$BUNDLE_RECEIPT"
)

if [[ ! "$BUNDLE_CONTENT_SHA256" =~ ^[0-9a-f]{{64}}$ ]]; then
    echo "Invalid hash" >&2
    exit 3
fi

if [[ "$BUNDLE_CONTENT_SHA256" != "$EXPECTED_BUNDLE_CONTENT_SHA256" ]]; then
    echo "Mismatch" >&2
    exit 3
fi

echo "RESOLVED_CONTENT_SHA=$BUNDLE_CONTENT_SHA256"
""".replace("\r\n", "\n").replace("\r", "").encode("utf-8")

            cp = subprocess.run(["bash", "-s"], input=bash_script, capture_output=True)
            out = cp.stdout.decode("utf-8", errors="replace")
            err = cp.stderr.decode("utf-8", errors="replace")
            assert cp.returncode == 0, f"Valid fixture failed ({cp.returncode}): {err}"
            assert "RESOLVED_CONTENT_SHA=c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b" in out
            assert "unbound variable" not in err
        finally:
            Path(receipt_path).unlink()

    def test_bundle_content_sha_invalid_fixtures_fail(self):
        """Phase 4C.1C.15: Invalid bundle_receipt fixtures fail with non-zero exit code."""
        invalid_payloads = [
            {},  # missing
            {"bundle_sha256": ""},  # empty
            {"bundle_sha256": "abc123"},  # too short
            {"bundle_sha256": "z" * 64},  # non-hex
            {"bundle_sha256": "d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"},  # archive hash mismatch
        ]
        for p in invalid_payloads:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as f:
                json.dump(p, f)
                p_path = f.name
            try:
                b_script = f"""set -Eeuo pipefail
SYS_PY3=$(command -v python3 || command -v python)
BUNDLE_RECEIPT="{p_path.replace(chr(92), '/')}"
BUNDLE_RECEIPT=$(wslpath -u "$BUNDLE_RECEIPT" 2>/dev/null || echo "$BUNDLE_RECEIPT")
EXPECTED_BUNDLE_CONTENT_SHA256="c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"

BUNDLE_CONTENT_SHA256=$(
    "$SYS_PY3" -c \\
    'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("bundle_sha256", ""))' \\
    "$BUNDLE_RECEIPT"
)

if [[ ! "$BUNDLE_CONTENT_SHA256" =~ ^[0-9a-f]{{64}}$ ]]; then
    exit 3
fi

if [[ "$BUNDLE_CONTENT_SHA256" != "$EXPECTED_BUNDLE_CONTENT_SHA256" ]]; then
    exit 3
fi
""".replace("\r\n", "\n").replace("\r", "").encode("utf-8")
                cp_inv = subprocess.run(["bash", "-s"], input=b_script, capture_output=True)
                assert cp_inv.returncode != 0, f"Expected non-zero exit code for payload {p}"
            finally:
                Path(p_path).unlink()

    def test_unbound_uppercase_variables_static_scan(self):
        """Phase 4C.1C.15: Static scan verifies all uppercase variables referenced in operator are defined or allowlisted."""
        text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
        ref_pattern = re.compile(r'\$\{?([A-Z][A-Z0-9_]*)\}?')
        refs = set(ref_pattern.findall(text))

        assign_pattern = re.compile(r'(?:^|[;\s])([A-Z][A-Z0-9_]*)=', re.MULTILINE)
        assigns = set(assign_pattern.findall(text))
        assigns.update(re.findall(r'\bfor\s+([A-Z][A-Z0-9_]*)\b', text))
        assigns.update(re.findall(r'\bread\s+([A-Z][A-Z0-9_]*)\b', text))

        allowlist = {"BASH_COMMAND", "LINENO", "PATH"}
        unassigned = refs - assigns - allowlist
        assert len(unassigned) == 0, f"Found unassigned uppercase variables: {unassigned}"

        # Verify static scanner catches C.14's unassigned BUNDLE_CONTENT_SHA256
        c14_text = text.replace("BUNDLE_CONTENT_SHA256=$(", "EXTRACTED_CONTENT_SHA=$(")
        c14_assigns = set(assign_pattern.findall(c14_text))
        c14_unassigned = refs - c14_assigns - allowlist
        assert "BUNDLE_CONTENT_SHA256" in c14_unassigned, "Static scanner failed to detect missing BUNDLE_CONTENT_SHA256 on C.14"

    def test_step3_to_step4_behavioral_preflight_no_training(self):
        """Phase 4C.1C.15: Behavioral preflight from Step 3 to Step 4 executes cleanly without unbound variables and verifies all 10 observation arguments."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            fake_receipt = tmp_p / "bundle_receipt.json"
            fake_receipt.write_text(
                json.dumps({"bundle_sha256": "c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"}),
                encoding="utf-8",
            )

            step_script = f"""set -Eeuo pipefail
SYS_PY3=$(command -v python3 || command -v python)

# Preflight outputs simulated
ACTUAL_GPU_MODEL="Tesla T4"
RUNTIME_POLICY="compatible"
GPU_POLICY="compatible"
EXECUTION_CODE_SHA="79bb11527d900fd387de1f41f2010c4152b7fea7"
CODE_ACTUAL_SHA="5e775a7708d1555bf22f56dceb5358e26e07dd224c4b6186f575aff4d2b590d5"
BUNDLE_ARCHIVE_SHA256="d49a106f0c4991ca8d79776277cbf7331df209157725c438288720dc42226a27"
EXPECTED_BUNDLE_CONTENT_SHA256="c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b"
STAGED_REQ_SHA="850478c0a9746b93354dd756ba428621a4aa48abab4e6aa5fa7e00669cea67a0"
STAGED_CONFIG_SHA="e03c07dae05a2402416abb0c60480a8cd0a35d060de3bb138a09d0bec0a85dc9"
SESSION_PIP_FREEZE_SHA="11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
BUNDLE_RECEIPT="{fake_receipt.as_posix()}"
BUNDLE_RECEIPT=$(wslpath -u "$BUNDLE_RECEIPT" 2>/dev/null || echo "$BUNDLE_RECEIPT")

# Step 3 content SHA extraction (Phase 4C.1C.15 block)
BUNDLE_CONTENT_SHA256=$(
    "$SYS_PY3" -c \\
    'import json, sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("bundle_sha256", ""))' \\
    "$BUNDLE_RECEIPT"
)

if [[ ! "$BUNDLE_CONTENT_SHA256" =~ ^[0-9a-f]{{64}}$ ]]; then
    exit 3
fi

if [[ "$BUNDLE_CONTENT_SHA256" != "$EXPECTED_BUNDLE_CONTENT_SHA256" ]]; then
    exit 3
fi

# Step 4 runtime observation argument passing simulation
args=(
    "test_obs.json"
    "$SESSION_PIP_FREEZE_SHA"
    "$STAGED_REQ_SHA"
    "$STAGED_CONFIG_SHA"
    "$RUNTIME_POLICY"
    "$GPU_POLICY"
    "$ACTUAL_GPU_MODEL"
    "$EXECUTION_CODE_SHA"
    "$CODE_ACTUAL_SHA"
    "$BUNDLE_ARCHIVE_SHA256"
    "$BUNDLE_CONTENT_SHA256"
)

# Verify all 11 args non-empty
for idx in "${{!args[@]}}"; do
    val="${{args[$idx]}}"
    if [[ -z "$val" ]]; then
        echo "Arg $idx is empty" >&2
        exit 1
    fi
    echo "ARG_$idx=$val"
done
""".replace("\r\n", "\n").replace("\r", "").encode("utf-8")

            cp_step = subprocess.run(["bash", "-s"], input=step_script, capture_output=True)
            out_step = cp_step.stdout.decode("utf-8", errors="replace")
            err_step = cp_step.stderr.decode("utf-8", errors="replace")
            assert cp_step.returncode == 0, f"Step 3-4 simulation failed: {err_step}"
            assert "ARG_10=c365c812cc814097f11b9e5ed5c82e672015e2ba093f09f975df0a2a01229e9b" in out_step
            assert "unbound variable" not in err_step
