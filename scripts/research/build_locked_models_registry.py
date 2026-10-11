#!/usr/bin/env python3
"""Build and seal the locked models registry for the external generalization probe.

Registers:
- A0-global (592-D)
- A0-local (640-D)
- A1-global (592-D)
- A1-local (640-D)
- A0-visual (576-D) [RQ2 control]
- A1-visual (576-D) [RQ2 control]

Documents exact SHA-256 hashes, feature order, scalers, coefficients, intercepts,
classes, and temperature provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("locked_models_registry")

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "research" / "evidence" / "three_class_preparation"
REGISTRY_PATH = EVIDENCE_DIR / "locked_models_registry.json"

CLASSES = ["authentic", "ai_edited", "fully_generated"]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    logger.info("=== Building Locked Models Registry ===")

    # Model paths
    p_a0_glob = EVIDENCE_DIR / "three_class_visual_dsp_model.json"
    p_a0_loc = EVIDENCE_DIR / "three_class_local_dsp_a0_model.json"
    p_a1_glob = EVIDENCE_DIR / "three_class_visual_dsp_a1_model.json"
    p_a1_loc = EVIDENCE_DIR / "three_class_local_dsp_a1_model.json"
    p_a0_vis = EVIDENCE_DIR / "three_class_visual_only_model.json"
    p_a1_vis = EVIDENCE_DIR / "three_class_visual_only_a1_model.json"

    for p in [p_a0_glob, p_a0_loc, p_a1_glob, p_a1_loc, p_a0_vis, p_a1_vis]:
        assert p.exists(), f"Model file missing: {p}"

    # Load data
    d_a0_glob = json.loads(p_a0_glob.read_text(encoding="utf-8"))
    d_a0_loc = json.loads(p_a0_loc.read_text(encoding="utf-8"))
    d_a1_glob = json.loads(p_a1_glob.read_text(encoding="utf-8"))
    d_a1_loc = json.loads(p_a1_loc.read_text(encoding="utf-8"))
    d_a0_vis = json.loads(p_a0_vis.read_text(encoding="utf-8"))
    d_a1_vis = json.loads(p_a1_vis.read_text(encoding="utf-8"))

    # Scaler reconstruction for A0-global (separate visual and dsp dicts in historical file)
    scaler_mean_a0_glob = d_a0_glob["scaler_visual"]["mean"] + d_a0_glob["scaler_dsp"]["mean"]
    scaler_scale_a0_glob = d_a0_glob["scaler_visual"]["scale"] + d_a0_glob["scaler_dsp"]["scale"]

    # Scaler reconstruction for A0-vis
    scaler_mean_a0_vis = d_a0_vis["scaler"]["mean"]
    scaler_scale_a0_vis = d_a0_vis["scaler"]["scale"]

    registry = {
        "schema_version": "1.0.0",
        "description": "Sealed registry of six locked three-class models for external generalization evaluation.",
        "target_classes": CLASSES,
        "models": {
            "A0-global": {
                "config_id": "A0_GLOBAL_DSP_592D",
                "training_condition": "A0_pilot_N750_uniform_weights",
                "feature_dim": 592,
                "feature_layout": "visual_576 + global_dsp_16",
                "model_artifact": "research/evidence/three_class_preparation/three_class_visual_dsp_model.json",
                "model_sha256": compute_sha256(p_a0_glob),
                "coefficients": d_a0_glob["coefficients"],
                "intercept": d_a0_glob["intercept"],
                "scaler_mean": scaler_mean_a0_glob,
                "scaler_scale": scaler_scale_a0_glob,
                "calibrated_temperature": 5.9577,  # Reconciled grouped-CV temperature
                "temperature_provenance": {
                    "reconciled_grouped_cv": 5.9577,
                    "historical_un_grouped": 8.1321,
                    "local_dsp_script_run": 5.8907,
                    "explanation": (
                        "5.9577 is the reconciled temperature from StratifiedGroupKFold(5, shuffle=True) with dual scalers. "
                        "5.8907 is from single-scaler StratifiedGroupKFold(5, shuffle=False) in train_and_evaluate_local_dsp.py. "
                        "8.1321 is historical non-grouped CV. Both 5.9577 and 5.8907 are valid; we evaluate primary using 5.9577 "
                        "and record 5.8907 sensitivity."
                    ),
                },
            },
            "A0-local": {
                "config_id": "A0_LOCAL_DSP_640D",
                "training_condition": "A0_pilot_N750_uniform_weights",
                "feature_dim": 640,
                "feature_layout": "visual_576 + global_dsp_16 + local_dsp_summaries_48",
                "model_artifact": "research/evidence/three_class_preparation/three_class_local_dsp_a0_model.json",
                "model_sha256": compute_sha256(p_a0_loc),
                "coefficients": d_a0_loc["weights"],
                "intercept": d_a0_loc["intercept"],
                "scaler_mean": d_a0_loc["scaler_mean"],
                "scaler_scale": d_a0_loc["scaler_scale"],
                "calibrated_temperature": d_a0_loc["temperature"],  # 6.0142
                "temperature_provenance": {
                    "calibrated_oof": round(d_a0_loc["temperature"], 4),
                    "explanation": "Calibrated temperature from StratifiedGroupKFold(5) on 750 development train samples.",
                },
            },
            "A1-global": {
                "config_id": "A1_GLOBAL_DSP_592D",
                "training_condition": "A1_supplemented_N1000_sample_weights_balanced",
                "feature_dim": 592,
                "feature_layout": "visual_576 + global_dsp_16",
                "model_artifact": "research/evidence/three_class_preparation/three_class_visual_dsp_a1_model.json",
                "model_sha256": compute_sha256(p_a1_glob),
                "coefficients": d_a1_glob["weights"],
                "intercept": d_a1_glob.get("intercept", d_a1_glob.get("bias")),
                "scaler_mean": d_a1_glob["scaler_mean"],
                "scaler_scale": d_a1_glob["scaler_scale"],
                "calibrated_temperature": 4.9391,  # from three_class_ablation_evaluation_receipt.json / model
                "temperature_provenance": {
                    "ablation_receipt_temperature": 4.9391,
                    "local_dsp_script_temperature": 5.1400,
                    "explanation": (
                        "4.9391 is from StratifiedGroupKFold(5, shuffle=True) with dual scalers in train_and_evaluate_ablation_a1.py. "
                        "5.1400 is from single-scaler StratifiedGroupKFold(5, shuffle=False) in train_and_evaluate_local_dsp.py."
                    ),
                },
            },
            "A1-local": {
                "config_id": "A1_LOCAL_DSP_640D",
                "training_condition": "A1_supplemented_N1000_sample_weights_balanced",
                "feature_dim": 640,
                "feature_layout": "visual_576 + global_dsp_16 + local_dsp_summaries_48",
                "model_artifact": "research/evidence/three_class_preparation/three_class_local_dsp_a1_model.json",
                "model_sha256": compute_sha256(p_a1_loc),
                "coefficients": d_a1_loc["weights"],
                "intercept": d_a1_loc["intercept"],
                "scaler_mean": d_a1_loc["scaler_mean"],
                "scaler_scale": d_a1_loc["scaler_scale"],
                "calibrated_temperature": d_a1_loc["temperature"],  # 5.1689
                "temperature_provenance": {
                    "calibrated_oof": round(d_a1_loc["temperature"], 4),
                    "explanation": "Calibrated temperature from StratifiedGroupKFold(5) on 1,000 supplemented samples.",
                },
            },
            "A0-visual": {
                "config_id": "A0_VISUAL_ONLY_576D",
                "training_condition": "A0_pilot_N750_uniform_weights",
                "feature_dim": 576,
                "feature_layout": "visual_576",
                "model_artifact": "research/evidence/three_class_preparation/three_class_visual_only_model.json",
                "model_sha256": compute_sha256(p_a0_vis),
                "coefficients": d_a0_vis["coefficients"],
                "intercept": d_a0_vis["intercept"],
                "scaler_mean": scaler_mean_a0_vis,
                "scaler_scale": scaler_scale_a0_vis,
                "calibrated_temperature": 5.9818,  # Reconciled grouped-CV
                "temperature_provenance": {
                    "reconciled_grouped_cv": 5.9818,
                    "historical_un_grouped": 8.3734,
                },
            },
            "A1-visual": {
                "config_id": "A1_VISUAL_ONLY_576D",
                "training_condition": "A1_supplemented_N1000_sample_weights_balanced",
                "feature_dim": 576,
                "feature_layout": "visual_576",
                "model_artifact": "research/evidence/three_class_preparation/three_class_visual_only_a1_model.json",
                "model_sha256": compute_sha256(p_a1_vis),
                "coefficients": d_a1_vis["weights"],
                "intercept": d_a1_vis.get("intercept", d_a1_vis.get("bias")),
                "scaler_mean": d_a1_vis["scaler_mean"],
                "scaler_scale": d_a1_vis["scaler_scale"],
                "calibrated_temperature": d_a1_vis["temperature"],  # 5.1813
                "temperature_provenance": {
                    "calibrated_oof": 5.1813,
                },
            },
        },
        "reporting_corrections": {
            "a0_local_minus_global_delta": {
                "value": "+0.0167",
                "formula": "Macro-F1(A0-local: 0.5552) - Macro-F1(A0-global: 0.5385) = +0.0167 (+1.67 percentage points)",
                "note": "Correction of earlier text printing +0.0168 resulting from unrounded float subtraction before display formatting.",
            }
        },
    }

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info(f"Registry successfully written to: {REGISTRY_PATH}")
    for name, m in registry["models"].items():
        logger.info(f"  {name}: {m['config_id']}, dim={m['feature_dim']}, temp={m['calibrated_temperature']}, SHA={m['model_sha256'][:16]}...")


if __name__ == "__main__":
    main()
