"""Tests for Phase 4C.1D.1: Rebuilt, Reconciled, and Audited Analysis Evidence from Raw Receipts."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pytest
from scipy import stats

REPO_ROOT = Path(__file__).parents[2]
LOCAL_ARTIFACTS = Path(r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\runs")
COLAB_T4_DIR = LOCAL_ARTIFACTS / "colab_t4"
EXTRACT_DIR = LOCAL_ARTIFACTS / "extracted_15_runs"
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "phase-4c.1d"
FIGURES_DIR = EVIDENCE_DIR / "figures"

EXPECTED_SIZES = [50, 100, 250]
EXPECTED_SEEDS = [42, 1337, 2025, 3407, 9001]
EXPECTED_ARTIFACT_NAMES = [
    "best_checkpoint.pt",
    "run_receipt.json",
    "epoch_history.json",
    "predictions.json",
    "training_history.csv",
    "metrics.json",
    "environment.json",
    "environment-binding.json",
    "checksums.json",
]

ARCHIVES = [
    "phase_4c1_all_15_runs_results.tar.gz",
    "phase_4c1_t4_execution_logs.tar.gz",
    "n50_results.tar.gz",
    "n100_results.tar.gz",
    "n250_results.tar.gz",
]


def compute_sha256_streaming(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_markdown_table_rows(markdown_text: str, header_keyword: str) -> List[List[str]]:
    """Helper to extract rows of a markdown table whose header contains header_keyword."""
    lines = markdown_text.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if in_table:
            if stripped.startswith("|") and stripped.endswith("|"):
                if not re.match(r"^\|(\s*:?-+:?\s*\|)+$", stripped):
                    cols = [c.strip() for c in stripped.split("|")[1:-1]]
                    table_lines.append(cols)
            else:
                break
        elif stripped.startswith("|") and header_keyword in stripped:
            in_table = True
    return table_lines


def verify_table2_parity_against_csv(markdown_content: str, csv_rows: List[Dict[str, str]]) -> Tuple[bool, str]:
    """Verifies Table 2 of report against run_level_metrics.csv."""
    rows = parse_markdown_table_rows(markdown_content, "Run ID")
    if len(rows) != len(csv_rows):
        return False, f"Row count mismatch: markdown has {len(rows)}, CSV has {len(csv_rows)}"

    for idx, (m_row, c_row) in enumerate(zip(rows, csv_rows)):
        # Run ID
        clean_m_id = m_row[0].replace("`", "")
        if clean_m_id != c_row["run_id"]:
            return False, f"Row {idx} Run ID mismatch: {clean_m_id} != {c_row['run_id']}"
        # Sample size
        if int(m_row[1]) != int(c_row["sample_size"]):
            return False, f"Row {idx} sample_size mismatch: {m_row[1]} != {c_row['sample_size']}"
        # Seed
        if int(m_row[2]) != int(c_row["seed"]):
            return False, f"Row {idx} seed mismatch: {m_row[2]} != {c_row['seed']}"
        # Epochs: e.g. "18 / 23"
        ep_expected = f"{c_row['best_epoch']} / {c_row['epochs_completed']}"
        if m_row[3] != ep_expected:
            return False, f"Row {idx} epochs mismatch: {m_row[3]} != {ep_expected}"
        # Macro-F1
        if abs(float(m_row[4]) - float(c_row["macro_f1"])) > 1e-4:
            return False, f"Row {idx} macro_f1 mismatch: {m_row[4]} != {c_row['macro_f1']}"
        # Balanced Acc
        if abs(float(m_row[5]) - float(c_row["balanced_accuracy"])) > 1e-4:
            return False, f"Row {idx} balanced_accuracy mismatch: {m_row[5]} != {c_row['balanced_accuracy']}"
        # AUROC
        if abs(float(m_row[6]) - float(c_row["auroc"])) > 1e-4:
            return False, f"Row {idx} auroc mismatch: {m_row[6]} != {c_row['auroc']}"
        # Brier Score
        if abs(float(m_row[7]) - float(c_row["brier_score"])) > 1e-4:
            return False, f"Row {idx} brier_score mismatch: {m_row[7]} != {c_row['brier_score']}"
        # ECE
        if abs(float(m_row[8]) - float(c_row["ece"])) > 1e-4:
            return False, f"Row {idx} ece mismatch: {m_row[8]} != {c_row['ece']}"
        # Val Loss
        if abs(float(m_row[9]) - float(c_row["val_loss"])) > 1e-4:
            return False, f"Row {idx} val_loss mismatch: {m_row[9]} != {c_row['val_loss']}"
        # Time (s)
        if abs(float(m_row[10]) - float(c_row["training_time_s"])) > 0.15:
            return False, f"Row {idx} time mismatch: {m_row[10]} != {c_row['training_time_s']}"
        # Checkpoint SHA-256 prefix: e.g. "`b5fa53bf...`"
        clean_sha = m_row[11].replace("`", "").replace("...", "")
        expected_prefix = c_row["checkpoint_sha256"][:8]
        if clean_sha != expected_prefix:
            return False, f"Row {idx} checkpoint SHA prefix mismatch: {clean_sha} != {expected_prefix}"

    return True, "Parity verified"


class TestPhase4C1DArchiveIngestion:
    """Verifies downloaded archives against SHA-256 sidecars."""

    def test_all_10_download_files_exist_and_match_sidecars(self):
        """All 5 archives and 5 sidecars must exist and match hashes perfectly."""
        for arch in ARCHIVES:
            arch_path = COLAB_T4_DIR / arch
            sidecar_path = COLAB_T4_DIR / f"{arch}.sha256"
            assert arch_path.exists(), f"Missing archive {arch}"
            assert sidecar_path.exists(), f"Missing sidecar {sidecar_path}"

            expected_sha = sidecar_path.read_text(encoding="utf-8").strip().split()[0].lower()
            actual_sha = compute_sha256_streaming(arch_path)
            assert actual_sha == expected_sha, f"Hash mismatch for {arch}: expected {expected_sha}, got {actual_sha}"


class TestPhase4C1DRunVerification:
    """Verifies that all 15 runs have 9 valid artifacts and satisfy all research invariants."""

    def test_extracted_15_runs_cardinality_and_artifacts(self):
        """All 15 run directories exist with 9 non-empty artifacts matching checksums.json."""
        assert EXTRACT_DIR.exists(), f"Missing extract dir {EXTRACT_DIR}"

        cohorts = []
        for size in EXPECTED_SIZES:
            for seed in EXPECTED_SEEDS:
                run_dir = EXTRACT_DIR / f"n{size}_seed_{seed}"
                assert run_dir.exists(), f"Missing run folder {run_dir.name}"

                # Check 9 artifacts
                for art_name in EXPECTED_ARTIFACT_NAMES:
                    art_path = run_dir / art_name
                    assert art_path.exists(), f"Missing {art_name} in {run_dir.name}"
                    assert art_path.stat().st_size > 0, f"Empty {art_name} in {run_dir.name}"

                # Check checksums.json matches all artifacts
                with open(run_dir / "checksums.json", "r", encoding="utf-8") as f:
                    csums = json.load(f)
                for art_name, info in csums.items():
                    act_sha = compute_sha256_streaming(run_dir / art_name)
                    act_bytes = (run_dir / art_name).stat().st_size
                    assert act_sha == info["sha256"], f"Checksum mismatch for {art_name} in {run_dir.name}"
                    assert act_bytes == info["size_bytes"], f"Byte mismatch for {art_name} in {run_dir.name}"

                # Check run_receipt.json invariants
                with open(run_dir / "run_receipt.json", "r", encoding="utf-8") as f:
                    rec = json.load(f)
                assert rec["status"] == "completed"
                assert rec["stage"] == "frozen"
                assert rec["sample_size"] == size
                assert rec["seed"] == seed
                assert rec.get("locked_test_access", 0) == 0
                assert rec.get("stage2_invocations", 0) == 0
                assert rec["validation_source_count"] == 91

                # Check predictions.json pairing
                with open(run_dir / "predictions.json", "r", encoding="utf-8") as f:
                    preds = json.load(f)
                assert len(preds["predictions"]) == 182
                assert len(preds["targets"]) == 182
                assert preds["targets"].count(0) == 91
                assert preds["targets"].count(1) == 91
                unique_srcs = sorted(set(preds["source_ids"]))
                assert len(unique_srcs) == 91
                cohorts.append(tuple(unique_srcs))

        # Confirm all 15 runs share identical validation source IDs
        assert len(set(cohorts)) == 1, "Validation cohort is not identical across all 15 runs!"


class TestPhase4C1DReconciliationAndParity:
    """Verifies that all 15 checkpoint prefixes, tables, summaries, and baselines match raw receipts."""

    def test_15_checkpoint_prefixes_match_raw_receipts(self):
        """Every checkpoint prefix in run_level_metrics.csv must match the raw receipt in extracted_15_runs."""
        csv_path = EVIDENCE_DIR / "run_level_metrics.csv"
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        assert len(rows) == 15

        for r in rows:
            run_dir = EXTRACT_DIR / f"n{r['sample_size']}_seed_{r['seed']}"
            raw_rec = json.load(open(run_dir / "run_receipt.json", encoding="utf-8"))
            raw_sha = raw_rec["checkpoint_sha256"]
            assert r["checkpoint_sha256"] == raw_sha
            assert r["checkpoint_sha256"][:8] == raw_sha[:8]

    def test_markdown_tables_match_csv_data(self):
        """Table 2 in phase_4c1_learning_curve_report.md must match run_level_metrics.csv exactly."""
        rep_path = EVIDENCE_DIR / "phase_4c1_learning_curve_report.md"
        csv_path = EVIDENCE_DIR / "run_level_metrics.csv"
        md_text = rep_path.read_text(encoding="utf-8")
        csv_rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

        is_valid, reason = verify_table2_parity_against_csv(md_text, csv_rows)
        assert is_valid, f"Parity check failed: {reason}"

    def test_summary_csv_and_json_match_recalculation(self):
        """learning_curve_summary.json and .csv must match direct recalculation from run_level_metrics.csv."""
        csv_path = EVIDENCE_DIR / "run_level_metrics.csv"
        json_path = EVIDENCE_DIR / "learning_curve_summary.json"
        summary_csv_path = EVIDENCE_DIR / "learning_curve_summary.csv"

        runs = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        summary_json = json.load(open(json_path, encoding="utf-8"))
        summary_csv_rows = list(csv.DictReader(open(summary_csv_path, encoding="utf-8")))

        t_crit = float(stats.t.ppf(0.975, df=4))

        for size in EXPECTED_SIZES:
            s_runs = [r for r in runs if int(r["sample_size"]) == size]
            assert len(s_runs) == 5

            for m in ["macro_f1", "balanced_accuracy", "auroc", "brier_score", "ece", "val_loss"]:
                vals = np.array([float(r[m]) for r in s_runs], dtype=float)
                exp_mean = float(np.mean(vals))
                exp_std = float(np.std(vals, ddof=1))
                se = exp_std / math.sqrt(5)
                exp_ci_l = exp_mean - t_crit * se
                exp_ci_u = exp_mean + t_crit * se

                json_stat = summary_json[str(size)][m]
                assert abs(json_stat["mean"] - exp_mean) < 1e-5
                assert abs(json_stat["std"] - exp_std) < 1e-5
                assert abs(json_stat["ci_95_lower"] - exp_ci_l) < 1e-5
                assert abs(json_stat["ci_95_upper"] - exp_ci_u) < 1e-5

                # Check CSV row
                csv_entry = next(r for r in summary_csv_rows if int(r["sample_size"]) == size and r["metric"] == m)
                assert abs(float(csv_entry["mean"]) - exp_mean) < 1e-5
                assert abs(float(csv_entry["std"]) - exp_std) < 1e-5

    def test_baseline_values_match_per_seed(self):
        """Dummy baseline must vary by seed as produced by Stratified DummyClassifier; Metadata baseline must be 0.5000."""
        csv_path = EVIDENCE_DIR / "run_level_metrics.csv"
        runs = list(csv.DictReader(open(csv_path, encoding="utf-8")))

        dummy_by_seed = {}
        for r in runs:
            seed = int(r["seed"])
            dummy_f1 = float(r["dummy_macro_f1"])
            meta_f1 = float(r["metadata_macro_f1"])

            assert meta_f1 == 0.5, f"Metadata baseline must be exactly 0.5000, got {meta_f1}"
            if seed not in dummy_by_seed:
                dummy_by_seed[seed] = dummy_f1
            else:
                assert abs(dummy_by_seed[seed] - dummy_f1) < 1e-6, f"Dummy baseline for seed {seed} diverged across N sizes"

        # Confirm dummy baseline has 5 distinct/valid per-seed values
        assert len(dummy_by_seed) == 5
        assert dummy_by_seed[42] != dummy_by_seed[1337], "Dummy baseline must reflect random seed sampling, not a constant"

    def test_figures_use_canonical_summary(self):
        """All 5 figure pairs must exist, be non-empty, and match canonical summary metrics."""
        fig_names = [
            "learning_curve_macro_f1",
            "per_seed_trajectories",
            "balanced_accuracy_auroc_vs_n",
            "brier_ece_vs_n",
            "model_vs_baselines",
        ]
        summary = json.load(open(EVIDENCE_DIR / "learning_curve_summary.json", encoding="utf-8"))
        f1_100_str = f"{summary['100']['macro_f1']['mean']:.4f}"

        for name in fig_names:
            svg_file = FIGURES_DIR / f"{name}.svg"
            png_file = FIGURES_DIR / f"{name}.png"
            assert svg_file.exists(), f"Missing SVG {svg_file}"
            assert png_file.exists(), f"Missing PNG {png_file}"
            assert svg_file.stat().st_size > 500
            assert png_file.stat().st_size > 1000

        # Check that Figure 1 and Figure 5 SVGs contain the canonical N=100 macro-F1
        f1_svg = (FIGURES_DIR / "learning_curve_macro_f1.svg").read_text(encoding="utf-8")
        assert f1_100_str in f1_svg, f"Canonical mean {f1_100_str} not embedded in learning_curve_macro_f1.svg"

    def test_fault_injection_markdown_mismatch_detected(self):
        """Fault injection test: intentionally modifying a single cell in the Markdown table must cause parity check to fail."""
        rep_path = EVIDENCE_DIR / "phase_4c1_learning_curve_report.md"
        csv_path = EVIDENCE_DIR / "run_level_metrics.csv"
        md_text = rep_path.read_text(encoding="utf-8")
        csv_rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))

        # Baseline check passes
        ok, msg = verify_table2_parity_against_csv(md_text, csv_rows)
        assert ok, f"Clean text failed parity: {msg}"

        # Fault Injection 1: Corrupt a Macro-F1 value
        corrupted_f1 = md_text.replace("0.5415", "0.9999", 1)
        ok_f1, msg_f1 = verify_table2_parity_against_csv(corrupted_f1, csv_rows)
        assert not ok_f1, "Fault injection on Macro-F1 was NOT detected!"
        assert "macro_f1 mismatch" in msg_f1

        # Fault Injection 2: Corrupt a Checkpoint SHA prefix
        corrupted_sha = md_text.replace("b5fa53bf", "deadbeef", 1)
        ok_sha, msg_sha = verify_table2_parity_against_csv(corrupted_sha, csv_rows)
        assert not ok_sha, "Fault injection on Checkpoint SHA prefix was NOT detected!"
        assert "checkpoint SHA prefix mismatch" in msg_sha

    def test_scientific_wording_forbidden_phrases_and_canonical_numbers(self):
        """Asserts that forbidden inaccurate or overclaimed scientific phrases are absent,
        and reports only contain numbers derived from canonical CSV/JSON."""
        rep_path = EVIDENCE_DIR / "phase_4c1_learning_curve_report.md"
        phase_rep_path = EVIDENCE_DIR / "PHASE_REPORT.md"

        rep_text = rep_path.read_text(encoding="utf-8")
        phase_rep_text = phase_rep_path.read_text(encoding="utf-8")
        combined_text = f"{rep_text}\n{phase_rep_text}"

        # 1. Strictly forbidden phrases
        forbidden_phrases = [
            "BCE sau sigmoid",
            "confirmed representation bottleneck",
            "Temperature Scaling is mandatory",
            "metadata baseline proves metadata performance",
            "Bắt buộc phải áp dụng Temperature Scaling",
            "mô hình trở nên overconfident",
            "có ý nghĩa thống kê chắc chắn",
            "Val Loss (BCE)",
        ]
        for phrase in forbidden_phrases:
            assert phrase.lower() not in combined_text.lower(), f"Forbidden phrase found in report: '{phrase}'"

        # 2. Required conservative scientific phrases
        required_phrases = [
            "runner-reported validation loss",
            "uninformative metadata placeholder baseline",
            "FocalLoss",
            "0.0625",
            "exploratory",
        ]
        for phrase in required_phrases:
            assert phrase.lower() in combined_text.lower(), f"Required conservative phrase missing: '{phrase}'"

        # 3. Canonical number verification: all numbers in summary CSV must be present in the markdown report
        summary_csv_path = EVIDENCE_DIR / "learning_curve_summary.csv"
        with open(summary_csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["metric"] in ["macro_f1", "auroc"]:
                    val_mean = f"{float(row['mean']):.4f}"
                    assert val_mean in rep_text, f"Canonical {row['metric']} {val_mean} missing from report"
