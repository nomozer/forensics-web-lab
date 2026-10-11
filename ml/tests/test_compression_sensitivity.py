"""
Unit test suite for So-Fake manifest v2 corrections and paired compression/resize sensitivity experiment.
Validates:
1. Manifest v2 integrity (300 distinct sample IDs and group IDs, no truncation collapsing).
2. Evaluation receipt v2 schema, corrected generator breakdowns by (gen, class), and taxonomy sensitivity.
3. Compression sensitivity receipt schema, C1/C2 pipeline divergence, Brier benchmark vs uniform baseline.
4. Raw predictions schema across 3 conditions, 2 cohorts, and 6 models.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"

MANIFEST_V2_PATH = EVIDENCE_DIR / "so_fake_probe_manifest_v2.json"
ISOLATION_V2_PATH = EVIDENCE_DIR / "so_fake_isolation_audit_v2.json"
EVAL_RECEIPT_V2_PATH = EVIDENCE_DIR / "so_fake_probe_evaluation_receipt_v2.json"
SENSITIVITY_RECEIPT_PATH = EVIDENCE_DIR / "compression_sensitivity_receipt.json"
SENSITIVITY_PREDS_PATH = EVIDENCE_DIR / "compression_sensitivity_predictions.json"


def test_manifest_v2_integrity_and_no_collapsing():
    assert MANIFEST_V2_PATH.exists(), f"Missing {MANIFEST_V2_PATH}"
    with open(MANIFEST_V2_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["schema_version"] == "2.0.0"
    assert m["total_samples"] == 300
    assert m["distinct_sample_ids"] == 300
    assert m["distinct_source_groups"] == 300

    records = m["records"]
    assert len(records) == 300

    seen_sids = set()
    seen_gids = set()
    for r in records:
        sid = r["sample_id"]
        gid = r["source_group_id"]
        assert sid not in seen_sids
        assert gid not in seen_gids
        seen_sids.add(sid)
        seen_gids.add(gid)
        assert r["source_independence_status"] == "unverified"
        assert not gid.endswith("_object_part_synt")  # ensures truncation bug is fixed

    assert len(seen_sids) == 300
    assert len(seen_gids) == 300


def test_isolation_audit_v2():
    assert ISOLATION_V2_PATH.exists(), f"Missing {ISOLATION_V2_PATH}"
    with open(ISOLATION_V2_PATH, "r", encoding="utf-8") as f:
        iso = json.load(f)

    assert iso["status"] == "ISOLATION_AUDIT_PASS"
    assert iso["collision_count"] == 0
    assert iso["distinct_source_groups"] == 300


def test_eval_receipt_v2_generator_breakdown_and_taxonomy():
    assert EVAL_RECEIPT_V2_PATH.exists(), f"Missing {EVAL_RECEIPT_V2_PATH}"
    with open(EVAL_RECEIPT_V2_PATH, "r", encoding="utf-8") as f:
        r = json.load(f)

    assert r["schema_version"] == "2.0.0"
    assert r["status"] == "SO_FAKE_EVALUATION_V2_SUCCESS"

    # Generator breakdown must distinguish flux.1-dev and latent_diffusion by ground truth label
    gb = r["generator_breakdown_by_class_n_ge_5"]
    assert "flux.1-dev [true: ai_edited]" in gb
    assert "flux.1-dev [true: fully_generated]" in gb
    assert gb["flux.1-dev [true: ai_edited]"]["sample_count"] == 44
    assert gb["flux.1-dev [true: fully_generated]"]["sample_count"] == 15

    assert "latent_diffusion [true: ai_edited]" in gb
    assert "latent_diffusion [true: fully_generated]" in gb
    assert gb["latent_diffusion [true: ai_edited]"]["sample_count"] == 12
    assert gb["latent_diffusion [true: fully_generated]"]["sample_count"] == 20

    # Taxonomy sensitivity for OpenForensics_fake
    tax = r["taxonomy_sensitivity_openforensics"]
    assert tax["excluded_count"] == 9
    assert tax["retained_count"] == 291
    assert "A0-global" in tax["models_comparison"]
    for m in ["A0-global", "A0-local", "A1-global", "A1-local", "A0-visual", "A1-visual"]:
        assert "macro_f1_full_n300" in tax["models_comparison"][m]
        assert "macro_f1_excl_n291" in tax["models_comparison"][m]


def test_compression_sensitivity_receipt_and_predictions():
    assert SENSITIVITY_RECEIPT_PATH.exists(), f"Missing {SENSITIVITY_RECEIPT_PATH}"
    assert SENSITIVITY_PREDS_PATH.exists(), f"Missing {SENSITIVITY_PREDS_PATH}"

    with open(SENSITIVITY_RECEIPT_PATH, "r", encoding="utf-8") as f:
        rec = json.load(f)

    with open(SENSITIVITY_PREDS_PATH, "r", encoding="utf-8") as f:
        preds = json.load(f)

    expected_cohorts = ["so_fake_probe", "development_holdout"]
    expected_models = ["A0-global", "A0-local", "A1-global", "A1-local", "A0-visual", "A1-visual"]
    expected_conds = ["C0", "C1", "C2"]

    for c in expected_cohorts:
        assert c in rec["metrics_by_cohort"]
        assert c in preds["results_by_cohort"]

        for m in expected_models:
            assert m in rec["metrics_by_cohort"][c]
            m_metrics = rec["metrics_by_cohort"][c][m]

            for cond in expected_conds:
                assert cond in m_metrics
                assert cond in preds["results_by_cohort"][c][m]
                c_data = m_metrics[cond]["metrics"]
                assert 0.0 <= c_data["macro_f1"] <= 1.0
                assert 0.0 <= c_data["balanced_accuracy"] <= 1.0
                assert c_data["multiclass_brier"] >= 0.0

            # Paired comparisons against C0
            comps = m_metrics["paired_comparisons_vs_C0"]
            assert "C1_minus_C0" in comps
            assert "C2_minus_C0" in comps
            assert comps["C1_minus_C0"]["decision_flip_rate"] >= 0.0
            assert comps["C2_minus_C0"]["decision_flip_rate"] >= 0.0

    # Verify Brier benchmark vs uniform baseline:
    # On So-Fake probe, all models have Brier > 0.6667 (worse than uniform guessing)
    for m in expected_models:
        sf_c0_brier = rec["metrics_by_cohort"]["so_fake_probe"][m]["C0"]["metrics"]["multiclass_brier"]
        assert sf_c0_brier > 0.6667, f"Expected Brier > 0.6667 for {m} on So-Fake, got {sf_c0_brier}"

    # Verify C2 exhibits higher authentic FP rate than C0 on DSP models (e.g. A0-global)
    a0_c0_fp = rec["metrics_by_cohort"]["so_fake_probe"]["A0-global"]["C0"]["metrics"]["authentic_control"]["false_positive_ai_rate"]
    a0_c2_fp = rec["metrics_by_cohort"]["so_fake_probe"]["A0-global"]["C2"]["metrics"]["authentic_control"]["false_positive_ai_rate"]
    assert a0_c2_fp > a0_c0_fp, "Expected C2 canvas JPEG to trigger higher false positive rate on DSP model"
