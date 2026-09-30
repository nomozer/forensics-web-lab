"""Tests for Phase 4C.1D: Ingestion, Verification and Learning Curve Analysis of 15 Stage-1 Runs."""

import csv
import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).parents[2]
COLAB_T4_DIR = Path(r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\runs\colab_t4")
EXTRACT_DIR = Path(r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\runs\extracted_15_runs")
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

class TestPhase4C1DAnalysisArtifacts:
    """Verifies that all required CSV, JSON, and figure artifacts are generated and valid."""

    def test_run_level_metrics_csv_validity(self):
        csv_path = EVIDENCE_DIR / "run_level_metrics.csv"
        assert csv_path.exists()
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        assert len(rows) == 15
        
        # Verify sizes and seeds
        sizes = [int(r["sample_size"]) for r in rows]
        assert sizes.count(50) == 5
        assert sizes.count(100) == 5
        assert sizes.count(250) == 5

    def test_learning_curve_summary_json_and_csv_validity(self):
        json_path = EVIDENCE_DIR / "learning_curve_summary.json"
        csv_path = EVIDENCE_DIR / "learning_curve_summary.csv"
        assert json_path.exists()
        assert csv_path.exists()
        
        data = json.load(open(json_path, encoding="utf-8"))
        for s in ["50", "100", "250"]:
            assert s in data
            assert "macro_f1" in data[s]
            assert data[s]["macro_f1"]["n_seeds"] == 5
            # Confirm CI is ordered
            assert data[s]["macro_f1"]["ci_95_lower"] < data[s]["macro_f1"]["mean"] < data[s]["macro_f1"]["ci_95_upper"]

    def test_paired_seed_deltas_correctness(self):
        p_path = EVIDENCE_DIR / "paired_seed_deltas.csv"
        assert p_path.exists()
        p_rows = list(csv.DictReader(open(p_path, encoding="utf-8")))
        assert len(p_rows) == 15 # 3 comparisons x 5 seeds
        
        # N100 minus N50 Macro-F1 deltas must all be positive
        n100_50_f1_deltas = [float(r["delta_macro_f1"]) for r in p_rows if r["comparison"] == "n100_minus_n50"]
        assert len(n100_50_f1_deltas) == 5
        assert all(d > 0 for d in n100_50_f1_deltas), "All 5 seeds must show positive delta from N=50 to N=100"

    def test_figures_exist_and_non_empty(self):
        fig_names = [
            "learning_curve_macro_f1",
            "per_seed_trajectories",
            "balanced_accuracy_auroc_vs_n",
            "brier_ece_vs_n",
            "model_vs_baselines",
        ]
        for name in fig_names:
            svg_file = FIGURES_DIR / f"{name}.svg"
            png_file = FIGURES_DIR / f"{name}.png"
            assert svg_file.exists(), f"Missing SVG {svg_file}"
            assert png_file.exists(), f"Missing PNG {png_file}"
            assert svg_file.stat().st_size > 500, f"SVG {svg_file} too small"
            assert png_file.stat().st_size > 1000, f"PNG {png_file} too small"
