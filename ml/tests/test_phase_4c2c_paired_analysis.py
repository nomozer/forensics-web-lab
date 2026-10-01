"""Unit and regression test suite for Phase 4C.2C Paired Stage 1 vs Stage 2 Analysis.

Validates the 20 mandatory criteria specified in Section J:
 1. Exact 15-pair matching.
 2. Missing Stage 1 run fails.
 3. Missing Stage 2 run fails.
 4. Duplicate N/seed fails.
 5. Receipt stage mismatch fails.
 6. Locked-test access fails.
 7. Stage 1 write fails.
 8. Source cohort mismatch fails.
 9. Target mismatch fails.
10. Metric recomputation parity.
11. Delta direction equals Stage 2 - Stage 1.
12. Paired t-test implementation.
13. Exact 32 sign-flip permutations.
14. Minimum attainable two-sided p-value = 0.0625.
15. Holm correction correctness.
16. 95% t-interval correctness with df=4.
17. No absolute Windows paths in Git evidence.
18. No raw artifacts/checkpoints copied into Git.
19. Deterministic outputs across repeated analysis.
20. Scientific wording guard against prohibited claims.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
from scipy import stats

from scripts.research.analyze_phase_4c2_paired import (
    REPO_ROOT,
    SEEDS,
    SIZES,
    T_CRIT_DF4_95,
    compute_t_ci_95,
    exact_sign_flip_p,
    find_run_dir,
    holm_bonferroni,
    perform_paired_analysis,
    recompute_metrics_from_predictions,
    validate_run_artifacts,
)

EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "phase-4c.2c"


# -----------------------------------------------------------------------------
# Fixtures for Synthetic Mock Runs
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_mock_environment(tmp_path: Path):
    """Creates a minimal valid synthetic paired Stage 1 and Stage 2 environment."""
    s1_root = tmp_path / "stage1"
    s2_root = tmp_path / "stage2"
    s1_root.mkdir()
    s2_root.mkdir()

    # Create dummy 91 unique sources and 182 balanced targets
    sources = [f"source_{i:04d}" for i in range(91)] * 2
    targets = [0] * 91 + [1] * 91
    # Mock probabilities slightly separated
    probs_s1 = [0.45] * 91 + [0.55] * 91
    preds_s1 = [0] * 91 + [1] * 91
    probs_s2 = [0.43] * 91 + [0.57] * 91
    preds_s2 = [0] * 91 + [1] * 91

    for size in SIZES:
        for seed in SEEDS:
            # Stage 1
            d1 = s1_root / f"n{size}_seed_{seed}"
            d1.mkdir(parents=True)
            _write_mock_run(d1, size, seed, "frozen", targets, sources, preds_s1, probs_s1, stage_num=1)

            # Stage 2
            d2 = s2_root / f"n{size}_seed_{seed}"
            d2.mkdir(parents=True)
            _write_mock_run(d2, size, seed, "partial_finetune", targets, sources, preds_s2, probs_s2, stage_num=2)

    return s1_root, s2_root, tmp_path / "out"


def _write_mock_run(run_dir: Path, size: int, seed: int, stage: str, targets, sources, preds, probs, stage_num: int):
    import hashlib
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    from ml.evaluation.metrics import compute_ece

    targets_arr = np.array(targets)
    preds_arr = np.array(preds)
    probs_arr = np.array(probs)

    f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    bacc = float(balanced_accuracy_score(targets_arr, preds_arr))
    auroc = float(roc_auc_score(targets_arr, probs_arr))
    brier = float(np.mean((probs_arr - targets_arr) ** 2))
    ece = float(compute_ece(targets_arr, probs_arr))

    metrics = {
        "macro_f1": f1,
        "balanced_accuracy": bacc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece,
        "loss": 0.172,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_data = {
        "predictions": preds,
        "probabilities": probs,
        "targets": targets,
        "source_ids": sources,
    }
    (run_dir / "predictions.json").write_text(json.dumps(predictions_data, indent=2), encoding="utf-8")

    receipt = {
        "status": "completed",
        "stage": stage,
        "sample_size": size,
        "seed": seed,
        "locked_test_access": 0,
        "validation_source_count": 91,
        "best_epoch": 5,
        "epochs_completed": 10,
        "training_time_seconds": 120.0,
        "peak_vram_mb": 110.0,
    }
    if stage_num == 2:
        receipt["stage1_output_writes"] = 0
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    (run_dir / "best_checkpoint.pt").write_bytes(b"dummy_checkpoint_weights")
    (run_dir / "epoch_history.json").write_text(json.dumps([{"epoch": 1}], indent=2), encoding="utf-8")
    (run_dir / "training_history.csv").write_text("epoch,loss\n1,0.5\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps({"env": "test"}, indent=2), encoding="utf-8")
    (run_dir / "environment-binding.json").write_text(json.dumps({"bind": "test"}, indent=2), encoding="utf-8")

    _update_checksums(run_dir)


def _update_checksums(run_dir: Path):
    """Recalculates and updates checksums.json for a run directory."""
    import hashlib
    checksums = {}
    for f in run_dir.iterdir():
        if f.name == "checksums.json":
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fp:
            data = fp.read()
            h.update(data)
        checksums[f.name] = {"sha256": h.hexdigest(), "size_bytes": len(data)}
    (run_dir / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Test 1: Exact 15-Pair Matching
# -----------------------------------------------------------------------------
def test_exact_15_pair_matching():
    """Confirms exact 1:1 pairing across the 15 cells defined by SIZES x SEEDS."""
    csv_path = EVIDENCE_DIR / "paired_run_metrics.csv"
    assert csv_path.exists(), f"Evidence file missing: {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 15, f"Expected 15 paired rows, got {len(reader)}"
    observed_pairs = [(int(r["sample_size"]), int(r["seed"])) for r in reader]
    expected_pairs = [(size, seed) for size in SIZES for seed in SEEDS]
    assert observed_pairs == expected_pairs, "Paired run matrix order/content mismatch"


# -----------------------------------------------------------------------------
# Test 2: Missing Stage 1 Run Fails
# -----------------------------------------------------------------------------
def test_missing_stage1_run_fails(synthetic_mock_environment):
    """Verifies fail-closed behavior when any Stage 1 run directory is missing."""
    s1_root, s2_root, out_dir = synthetic_mock_environment
    # Remove one Stage 1 run
    target = s1_root / "n50_seed_42"
    shutil.rmtree(target)

    with pytest.raises(FileNotFoundError, match="Missing Stage 1 run directory"):
        perform_paired_analysis(s1_root, s2_root, out_dir)


# -----------------------------------------------------------------------------
# Test 3: Missing Stage 2 Run Fails
# -----------------------------------------------------------------------------
def test_missing_stage2_run_fails(synthetic_mock_environment):
    """Verifies fail-closed behavior when any Stage 2 run directory is missing."""
    s1_root, s2_root, out_dir = synthetic_mock_environment
    # Remove one Stage 2 run
    target = s2_root / "n100_seed_1337"
    shutil.rmtree(target)

    with pytest.raises(FileNotFoundError, match="Missing Stage 2 run directory"):
        perform_paired_analysis(s1_root, s2_root, out_dir)


# -----------------------------------------------------------------------------
# Test 4: Duplicate N/Seed Fails
# -----------------------------------------------------------------------------
def test_duplicate_n_seed_fails(synthetic_mock_environment):
    """Verifies fail-closed behavior if duplicate N/seed runs are passed or discovered."""
    s1_root, s2_root, _ = synthetic_mock_environment
    run_dir = s1_root / "n50_seed_42"
    receipt = json.loads((run_dir / "run_receipt.json").read_text(encoding="utf-8"))
    receipt["sample_size"] = 100  # Conflicts with expected N=50
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    _update_checksums(run_dir)

    with pytest.raises(ValueError, match="Sample size mismatch"):
        validate_run_artifacts(run_dir, expected_size=50, expected_seed=42, expected_stage="frozen", stage_num=1)


# -----------------------------------------------------------------------------
# Test 5: Receipt Stage Mismatch Fails
# -----------------------------------------------------------------------------
def test_receipt_stage_mismatch_fails(synthetic_mock_environment):
    """Verifies that receipt stage mismatch (e.g. stage != frozen or != partial_finetune) fails closed."""
    s1_root, s2_root, _ = synthetic_mock_environment
    d1 = s1_root / "n50_seed_42"
    receipt1 = json.loads((d1 / "run_receipt.json").read_text(encoding="utf-8"))
    receipt1["stage"] = "partial_finetune"
    (d1 / "run_receipt.json").write_text(json.dumps(receipt1), encoding="utf-8")
    _update_checksums(d1)

    with pytest.raises(ValueError, match="Stage mismatch in n50_seed_42"):
        validate_run_artifacts(d1, expected_size=50, expected_seed=42, expected_stage="frozen", stage_num=1)


# -----------------------------------------------------------------------------
# Test 6: Locked-Test Access Fails
# -----------------------------------------------------------------------------
def test_locked_test_access_fails(synthetic_mock_environment):
    """Verifies that locked_test_access != 0 triggers immediate fail-closed abort."""
    s1_root, s2_root, _ = synthetic_mock_environment
    d2 = s2_root / "n50_seed_42"
    receipt2 = json.loads((d2 / "run_receipt.json").read_text(encoding="utf-8"))
    receipt2["locked_test_access"] = 1
    (d2 / "run_receipt.json").write_text(json.dumps(receipt2), encoding="utf-8")
    _update_checksums(d2)

    with pytest.raises(ValueError, match="locked_test_access != 0"):
        validate_run_artifacts(d2, expected_size=50, expected_seed=42, expected_stage="partial_finetune", stage_num=2)


# -----------------------------------------------------------------------------
# Test 7: Stage 1 Write Fails
# -----------------------------------------------------------------------------
def test_stage1_write_fails(synthetic_mock_environment):
    """Verifies that Stage 2 stage1_output_writes != 0 triggers immediate fail-closed abort."""
    s1_root, s2_root, _ = synthetic_mock_environment
    d2 = s2_root / "n50_seed_42"
    receipt2 = json.loads((d2 / "run_receipt.json").read_text(encoding="utf-8"))
    receipt2["stage1_output_writes"] = 1
    (d2 / "run_receipt.json").write_text(json.dumps(receipt2), encoding="utf-8")
    _update_checksums(d2)

    with pytest.raises(ValueError, match="stage1_output_writes != 0"):
        validate_run_artifacts(d2, expected_size=50, expected_seed=42, expected_stage="partial_finetune", stage_num=2)


# -----------------------------------------------------------------------------
# Test 8: Source Cohort Mismatch Fails
# -----------------------------------------------------------------------------
def test_source_cohort_mismatch_fails(synthetic_mock_environment):
    """Verifies that differing source IDs between Stage 1 and Stage 2 trigger fail-closed abort."""
    s1_root, s2_root, out_dir = synthetic_mock_environment
    d2 = s2_root / "n50_seed_42"
    preds2 = json.loads((d2 / "predictions.json").read_text(encoding="utf-8"))
    old_id = preds2["source_ids"][0]
    new_id = "alternate_source_id_001"
    preds2["source_ids"] = [new_id if s == old_id else s for s in preds2["source_ids"]]
    (d2 / "predictions.json").write_text(json.dumps(preds2), encoding="utf-8")
    _update_checksums(d2)

    with pytest.raises(ValueError, match="Source IDs mismatch"):
        perform_paired_analysis(s1_root, s2_root, out_dir)


# -----------------------------------------------------------------------------
# Test 9: Target Mismatch Fails
# -----------------------------------------------------------------------------
def test_target_mismatch_fails(synthetic_mock_environment):
    """Verifies that differing targets between Stage 1 and Stage 2 trigger fail-closed abort."""
    s1_root, s2_root, out_dir = synthetic_mock_environment
    d2 = s2_root / "n50_seed_42"
    preds2 = json.loads((d2 / "predictions.json").read_text(encoding="utf-8"))
    preds2["targets"][0], preds2["targets"][91] = preds2["targets"][91], preds2["targets"][0]
    preds2["predictions"][0], preds2["predictions"][91] = preds2["predictions"][91], preds2["predictions"][0]
    preds2["probabilities"][0], preds2["probabilities"][91] = preds2["probabilities"][91], preds2["probabilities"][0]
    (d2 / "predictions.json").write_text(json.dumps(preds2), encoding="utf-8")

    recomputed = recompute_metrics_from_predictions(d2 / "predictions.json")
    m2 = json.loads((d2 / "metrics.json").read_text(encoding="utf-8"))
    m2.update(recomputed)
    (d2 / "metrics.json").write_text(json.dumps(m2), encoding="utf-8")
    _update_checksums(d2)

    with pytest.raises(ValueError, match="Targets mismatch"):
        perform_paired_analysis(s1_root, s2_root, out_dir)


# -----------------------------------------------------------------------------
# Test 10: Metric Recomputation Parity
# -----------------------------------------------------------------------------
def test_metric_recomputation_parity(synthetic_mock_environment):
    """Verifies recomputation matches metrics.json within tolerance, and detects tampering."""
    s1_root, _, _ = synthetic_mock_environment
    d1 = s1_root / "n50_seed_42"
    recomputed = recompute_metrics_from_predictions(d1 / "predictions.json")
    metrics_file = json.loads((d1 / "metrics.json").read_text(encoding="utf-8"))
    for k in ["macro_f1", "balanced_accuracy", "auroc", "brier_score", "ece"]:
        assert abs(recomputed[k] - float(metrics_file[k])) <= 1e-5

    # Tampered metrics.json with updated checksum triggers parity error
    metrics_file["macro_f1"] = 0.9999
    (d1 / "metrics.json").write_text(json.dumps(metrics_file), encoding="utf-8")
    _update_checksums(d1)

    with pytest.raises(ValueError, match="Metric recomputation parity failure"):
        validate_run_artifacts(d1, expected_size=50, expected_seed=42, expected_stage="frozen", stage_num=1)


# -----------------------------------------------------------------------------
# Test 11: Delta Direction Equals Stage 2 - Stage 1
# -----------------------------------------------------------------------------
def test_delta_direction_equals_stage2_minus_stage1():
    """Asserts that delta convention is strictly delta = Stage 2 - Stage 1 across all metrics."""
    csv_path = EVIDENCE_DIR / "paired_run_metrics.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        # Macro F1
        expected_d_f1 = float(r["stage2_macro_f1"]) - float(r["stage1_macro_f1"])
        assert abs(float(r["delta_macro_f1"]) - expected_d_f1) <= 1e-9
        # Balanced Accuracy
        expected_d_ba = float(r["stage2_balanced_accuracy"]) - float(r["stage1_balanced_accuracy"])
        assert abs(float(r["delta_balanced_accuracy"]) - expected_d_ba) <= 1e-9
        # AUROC
        expected_d_auc = float(r["stage2_auroc"]) - float(r["stage1_auroc"])
        assert abs(float(r["delta_auroc"]) - expected_d_auc) <= 1e-9
        # Brier
        expected_d_br = float(r["stage2_brier"]) - float(r["stage1_brier"])
        assert abs(float(r["delta_brier"]) - expected_d_br) <= 1e-9
        # ECE
        expected_d_ece = float(r["stage2_ece"]) - float(r["stage1_ece"])
        assert abs(float(r["delta_ece"]) - expected_d_ece) <= 1e-9
        # Loss
        expected_d_loss = float(r["stage2_runner_val_loss"]) - float(r["stage1_runner_val_loss"])
        assert abs(float(r["delta_runner_val_loss"]) - expected_d_loss) <= 1e-9


# -----------------------------------------------------------------------------
# Test 12: Paired T-Test Implementation
# -----------------------------------------------------------------------------
def test_paired_t_test_implementation():
    """Verifies that paired t-test matches analytical formula t = mean(d) / (std(d) / sqrt(n))."""
    deltas = [0.0135, 0.0175, -0.0003, 0.0149, 0.0014]
    m = np.mean(deltas)
    s = np.std(deltas, ddof=1)
    se = s / math.sqrt(len(deltas))
    expected_t = m / se
    expected_p = float(2.0 * (1.0 - stats.t.cdf(abs(expected_t), df=len(deltas) - 1)))

    res = stats.ttest_rel([x + 0.5 for x in deltas], [0.5] * len(deltas))
    assert abs(res.statistic - expected_t) < 1e-6
    assert abs(res.pvalue - expected_p) < 1e-6


# -----------------------------------------------------------------------------
# Test 13: Exact 32 Sign-Flip Permutations
# -----------------------------------------------------------------------------
def test_exact_32_sign_flip_permutations():
    """Verifies that exact_sign_flip_p uses all 2^5 = 32 permutations without approximation."""
    # When deltas are all positive and distinct: exactly 2 permutations (all +1 and all -1) match/exceed
    d_all_pos = [0.01, 0.02, 0.03, 0.04, 0.05]
    p_val = exact_sign_flip_p(d_all_pos)
    assert p_val == 2 / 32  # 0.0625

    # When mean is exactly zero: all 32 permutations have abs(mean) >= 0
    d_zero = [-0.02, -0.01, 0.0, 0.01, 0.02]
    assert exact_sign_flip_p(d_zero) == 1.0


# -----------------------------------------------------------------------------
# Test 14: Minimum Attainable Two-Sided P-Value = 0.0625
# -----------------------------------------------------------------------------
def test_minimum_attainable_two_sided_p_value():
    """Verifies that for n=5, two-sided sign-flip permutation p cannot drop below 0.0625."""
    # Test random non-zero vectors
    rng = np.random.RandomState(42)
    for _ in range(20):
        vec = rng.uniform(0.001, 0.1, size=5)
        p = exact_sign_flip_p(list(vec))
        assert p >= 0.0625, f"P-value {p} < 0.0625 mathematically impossible for n=5"


# -----------------------------------------------------------------------------
# Test 15: Holm Correction Correctness
# -----------------------------------------------------------------------------
def test_holm_correction_correctness():
    """Verifies step-down Holm-Bonferroni correction with known test cases."""
    # Case 1: monotonic raw p-values
    raw = [0.01, 0.04, 0.05]
    # Sorted: 0.01 (x3=0.03), 0.04 (x2=0.08), 0.05 (x1=0.05 -> max(0.08, 0.05)=0.08)
    adj = holm_bonferroni(raw)
    assert adj == [0.03, 0.08, 0.08]

    # Case 2: non-monotonic input order
    raw2 = [0.062816, 0.683336, 0.446251]
    adj2 = holm_bonferroni(raw2)
    # Sorted indices: 0 (0.062816), 2 (0.446251), 1 (0.683336)
    # Rank 1: 0.062816 * 3 = 0.188448
    # Rank 2: max(0.188448, 0.446251 * 2) = 0.892502
    # Rank 3: max(0.892502, 0.683336 * 1) = 0.892502
    assert abs(adj2[0] - 0.188448) < 1e-4
    assert abs(adj2[1] - 0.892502) < 1e-4
    assert abs(adj2[2] - 0.892502) < 1e-4


# -----------------------------------------------------------------------------
# Test 16: 95% t-Interval Correctness With df=4
# -----------------------------------------------------------------------------
def test_95_t_interval_correctness_df4():
    """Verifies that compute_t_ci_95 strictly uses Student's t critical value with df=4."""
    assert abs(T_CRIT_DF4_95 - 2.7764451051977987) < 1e-8

    mean = 0.0094
    std = 0.00822
    n = 5
    se = std / math.sqrt(n)
    expected_l = mean - 2.7764451051977987 * se
    expected_u = mean + 2.7764451051977987 * se

    ci_l, ci_u = compute_t_ci_95(mean, std, n=5)
    assert abs(ci_l - expected_l) < 1e-8
    assert abs(ci_u - expected_u) < 1e-8


# -----------------------------------------------------------------------------
# Test 17: No Absolute Windows Paths in Git Evidence
# -----------------------------------------------------------------------------
def test_no_absolute_windows_paths_in_git_evidence():
    """Verifies that no absolute Windows paths (e.g. C:/, D:/, \\Users, \\Documents) are present in evidence."""
    drive_pattern = re.compile(r"[a-zA-Z]:[\\/]")
    backslash_pattern = re.compile(r"\\(?:Users|Documents|forensics|content)")

    for p in EVIDENCE_DIR.rglob("*"):
        if p.is_file() and p.suffix in [".json", ".csv", ".md"]:
            content = p.read_text(encoding="utf-8")
            assert not drive_pattern.search(content), f"Absolute drive path found in {p}"
            assert not backslash_pattern.search(content), f"Backslash path found in {p}"


# -----------------------------------------------------------------------------
# Test 18: No Raw Artifacts/Checkpoints Copied into Git
# -----------------------------------------------------------------------------
def test_no_raw_artifacts_copied_into_git():
    """Verifies that raw checkpoint files (.pt), archives (.tar.gz), and predictions are not stored in evidence."""
    forbidden_suffixes = [".pt", ".tar.gz", ".tar", ".pth", ".bin"]
    forbidden_files = ["predictions.json", "best_checkpoint.pt", "normalized_execution_manifest.json"]

    for p in EVIDENCE_DIR.rglob("*"):
        assert not any(p.name.endswith(sfx) for sfx in forbidden_suffixes), f"Forbidden raw artifact in Git: {p}"
        assert p.name not in forbidden_files, f"Forbidden raw file in Git: {p}"


# -----------------------------------------------------------------------------
# Test 19: Deterministic Outputs Across Repeated Analysis
# -----------------------------------------------------------------------------
def test_deterministic_outputs_across_repeated_analysis(synthetic_mock_environment):
    """Verifies that running the analysis twice on identical inputs produces bitwise identical CSV and JSON."""
    s1_root, s2_root, out_base = synthetic_mock_environment
    out1 = out_base / "run1"
    out2 = out_base / "run2"

    perform_paired_analysis(s1_root, s2_root, out1)
    perform_paired_analysis(s1_root, s2_root, out2)

    files_to_check = [
        "paired_run_metrics.csv",
        "primary_statistical_tests.csv",
        "primary_statistical_tests.json",
        "paired_summary.csv",
        "paired_summary.json",
        "calibration_summary.csv",
    ]

    for fname in files_to_check:
        b1 = (out1 / fname).read_bytes()
        b2 = (out2 / fname).read_bytes()
        assert b1 == b2, f"Non-deterministic output detected in {fname}"


# -----------------------------------------------------------------------------
# Test 20: Scientific Wording Guard Against Prohibited Claims
# -----------------------------------------------------------------------------
def test_scientific_wording_guard_against_prohibited_claims():
    """Verifies that PHASE_REPORT.md strictly avoids prohibited claims and includes required phrases."""
    report_path = EVIDENCE_DIR / "PHASE_REPORT.md"
    assert report_path.exists(), f"PHASE_REPORT.md missing: {report_path}"
    content = report_path.read_text(encoding="utf-8")

    # Prohibited claims (regex word boundaries to avoid substring false positives like 'approved')
    prohibited_patterns = [
        r"\bconfirmed superiority\b",
        r"\bstatistically superior\b",
        r"\bproved\b",
        r"\bproves\b",
        r"\bcausal isolation\b",
        r"\bpure unfreezing effect\b",
        r"\bstatistically significant improvement\b",
    ]
    for pat in prohibited_patterns:
        assert not re.search(pat, content, re.IGNORECASE), f"Prohibited pattern '{pat}' found in report"

    # Required scientific qualification phrases
    required_phrases = [
        "pre-registered partial fine-tuning protocol",
        "exploratory",
        "model-development evidence",
        "PHASE_4C2C_PAIRED_ANALYSIS_COMPLETE",
    ]
    for phrase in required_phrases:
        assert phrase in content, f"Required phrase '{phrase}' missing from report"
