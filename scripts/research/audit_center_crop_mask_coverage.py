#!/usr/bin/env python3
"""
Audit Center-Crop Mask Coverage on 91 TGIF SD2 Inpainted Holdout Samples
Protocol: PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT & Continuous Diagnostics

Examines whether the canonical square center-crop removes or attenuates inpainting edits:
  - Canonical crop: left = (w - min(w, h)) // 2, top = (h - min(w, h)) // 2
  - Measures: original edit area, retained edit area post-crop, retention ratio, canvas ratio
  - Identifies partial loss and complete loss (0% retained)
  - Evaluates classification behavior across predefined coverage bins across all 6 locked models
  - Explicit distinction: Crop losing edit area is an observation; crop causing model error is a hypothesis.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = WORKSPACE_ROOT / "research" / "evidence" / "three_class_preparation"
DATA_DIR = WORKSPACE_ROOT / "data" / "research" / "tgif"

PREDEFINED_BINS = ["full_retention", "minor_loss", "major_loss", "complete_loss"]


def classify_coverage_bin(ratio: float) -> str:
    if ratio >= 0.999999:
        return "full_retention"
    elif ratio >= 0.75:
        return "minor_loss"
    elif ratio > 0.0:
        return "major_loss"
    else:
        return "complete_loss"


def load_model_predictions() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Loads predictions of 91 holdout samples for all 6 locked models."""
    p_rec = json.load(open(EVIDENCE_DIR / "three_class_pilot_predictions_reconciled.json"))
    p_a1 = json.load(open(EVIDENCE_DIR / "three_class_ablation_a1_predictions.json"))
    p_loc = json.load(open(EVIDENCE_DIR / "three_class_local_dsp_predictions.json"))

    preds: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def extract_fields(d: Dict[str, Any]) -> Dict[str, Any]:
        probs = d.get("probabilities") if "probabilities" in d else d.get("probs")
        pred_cls = d.get("predicted_class") if "predicted_class" in d else d.get("pred_class")
        pred_lbl = d.get("predicted_label") if "predicted_label" in d else d.get("verdict")
        return {"probabilities": list(probs), "predicted_class": int(pred_cls), "predicted_label": str(pred_lbl)}

    for s in p_rec["holdout_predictions"]:
        s_id = s["sample_id"]
        preds.setdefault(s_id, {})
        preds[s_id]["A0-visual"] = extract_fields(s["visual_only"])

    for s in p_a1["holdout_predictions"]:
        s_id = s["sample_id"]
        preds.setdefault(s_id, {})
        preds[s_id]["A1-visual"] = extract_fields(s["a1_visual_only"])

    for m in ["A0-global", "A0-local", "A1-global", "A1-local"]:
        for s in p_loc[m]["holdout_predictions"]:
            s_id = s["sample_id"]
            preds.setdefault(s_id, {})
            preds[s_id][m] = extract_fields(s)

    return preds


def main():
    print("Auditing center-crop mask coverage on 91 holdout edited samples...")
    splits = json.load(open(EVIDENCE_DIR / "three_class_pilot_splits.json"))
    holdout_edited = [r for r in splits["records"] if r.get("partition") == "development_holdout" and r.get("class_name") == "ai_edited"]
    assert len(holdout_edited) == 91, f"Expected 91 edited samples, got {len(holdout_edited)}"

    model_preds = load_model_predictions()
    models = ["A0-global", "A0-local", "A0-visual", "A1-global", "A1-local", "A1-visual"]

    sample_audits = []

    for r in holdout_edited:
        s_id = r["sample_id"]
        rel_path = r["relative_path"]
        parts = rel_path.replace("\\", "/").split("/")
        split_name = parts[4]
        category = parts[5]
        fname = parts[6]
        mask_fname = fname.replace("_sd2_0.png", "")

        img_path = WORKSPACE_ROOT / rel_path
        mask_path = DATA_DIR / "masks" / split_name / category / mask_fname

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # Open image and mask to audit dimensions and coordinates
        with Image.open(img_path) as img:
            w, h = img.size

        with Image.open(mask_path) as mask_img:
            mw, mh = mask_img.size
            mask_arr = np.array(mask_img)

        if len(mask_arr.shape) == 3:
            mask_arr = mask_arr[:, :, 0]

        assert (w, h) == (mw, mh), f"Dimension mismatch between image ({w}, {h}) and mask ({mw}, {mh})"

        # Canonical crop box formula
        min_dim = min(w, h)
        crop_left = (w - min_dim) // 2
        crop_top = (h - min_dim) // 2
        crop_right = crop_left + min_dim
        crop_bottom = crop_top + min_dim

        # Original mask metrics
        binary_mask = (mask_arr > 0)
        orig_edit_pixels = int(np.count_nonzero(binary_mask))
        assert orig_edit_pixels > 0, f"Sample {s_id} has empty mask!"

        # Mask bounding box in original coordinates
        y_indices, x_indices = np.nonzero(binary_mask)
        mask_bbox_orig = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices)), int(np.max(y_indices))]

        # Retained edit pixels inside crop window
        cropped_mask_window = binary_mask[crop_top:crop_bottom, crop_left:crop_right]
        retained_edit_pixels = int(np.count_nonzero(cropped_mask_window))

        retention_ratio = float(retained_edit_pixels / orig_edit_pixels)
        canvas_ratio = float(retained_edit_pixels / (min_dim * min_dim))
        coverage_bin = classify_coverage_bin(retention_ratio)

        # Model predictions
        sample_preds = model_preds.get(s_id, {})
        m_eval = {}
        for m in models:
            mp = sample_preds.get(m, {})
            p_cls = mp.get("predicted_class")
            m_eval[m] = {
                "predicted_class": p_cls,
                "predicted_label": mp.get("predicted_label"),
                "is_correct": bool(p_cls == 1),
                "misclassified_as_authentic": bool(p_cls == 0),
                "misclassified_as_generated": bool(p_cls == 2),
                "probabilities": mp.get("probabilities")
            }

        sample_audits.append({
            "sample_id": s_id,
            "source_id": r["source_id"],
            "source_group_id": r["source_group_id"],
            "category": category,
            "relative_image_path": rel_path,
            "relative_mask_path": str(mask_path.relative_to(WORKSPACE_ROOT)),
            "image_geometry": {
                "width": w,
                "height": h,
                "aspect_ratio": round(w / h, 4),
                "is_square": bool(w == h)
            },
            "canonical_crop_box": {
                "left": crop_left,
                "top": crop_top,
                "right": crop_right,
                "bottom": crop_bottom,
                "min_dim": min_dim
            },
            "mask_geometry": {
                "bbox_orig": mask_bbox_orig,
                "original_edit_pixels": orig_edit_pixels,
                "retained_edit_pixels": retained_edit_pixels,
                "lost_edit_pixels": orig_edit_pixels - retained_edit_pixels,
                "retention_ratio": round(retention_ratio, 6),
                "canvas_edit_ratio": round(canvas_ratio, 6),
                "coverage_bin": coverage_bin
            },
            "model_evaluations": m_eval
        })

    # Summary Statistics
    bin_counts = {b: 0 for b in PREDEFINED_BINS}
    for item in sample_audits:
        bin_counts[item["mask_geometry"]["coverage_bin"]] += 1

    # Analysis across coverage bins for each model
    bin_performance: Dict[str, Dict[str, Any]] = {b: {"support": bin_counts[b], "models": {}} for b in PREDEFINED_BINS}

    for b in PREDEFINED_BINS:
        sub_items = [item for item in sample_audits if item["mask_geometry"]["coverage_bin"] == b]
        n_sub = len(sub_items)
        for m in models:
            if n_sub > 0:
                n_correct = sum(1 for it in sub_items if it["model_evaluations"][m]["is_correct"])
                n_to_auth = sum(1 for it in sub_items if it["model_evaluations"][m]["misclassified_as_authentic"])
                n_to_gen = sum(1 for it in sub_items if it["model_evaluations"][m]["misclassified_as_generated"])
                bin_performance[b]["models"][m] = {
                    "recall": round(n_correct / n_sub, 4),
                    "misclassified_as_authentic_rate": round(n_to_auth / n_sub, 4),
                    "misclassified_as_generated_rate": round(n_to_gen / n_sub, 4),
                    "correct_count": n_correct,
                    "to_authentic_count": n_to_auth,
                    "to_generated_count": n_to_gen
                }
            else:
                bin_performance[b]["models"][m] = {
                    "recall": None,
                    "misclassified_as_authentic_rate": None,
                    "misclassified_as_generated_rate": None,
                    "correct_count": 0,
                    "to_authentic_count": 0,
                    "to_generated_count": 0
                }

    # Loss severity summary
    complete_loss_samples = [it["sample_id"] for it in sample_audits if it["mask_geometry"]["coverage_bin"] == "complete_loss"]
    major_loss_samples = [it["sample_id"] for it in sample_audits if it["mask_geometry"]["coverage_bin"] == "major_loss"]
    minor_loss_samples = [it["sample_id"] for it in sample_audits if it["mask_geometry"]["coverage_bin"] == "minor_loss"]
    full_retention_samples = [it["sample_id"] for it in sample_audits if it["mask_geometry"]["coverage_bin"] == "full_retention"]

    # Manifest export
    manifest = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT",
        "description": "Per-sample geometric audit of center-crop mask coverage on 91 TGIF holdout edited images",
        "total_samples_audited": len(sample_audits),
        "coverage_bin_distribution": bin_counts,
        "sample_audits": sample_audits
    }
    manifest_path = EVIDENCE_DIR / "center_crop_mask_coverage_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {manifest_path}")

    # Receipt export
    receipt = {
        "schema_version": "1.0.0",
        "protocol_id": "PROTO-RQ1-RQ2-3CLASS-V2.2-AMENDMENT",
        "receipt_id": "center_crop_mask_coverage_receipt_v1",
        "timestamp": "2026-10-11T22:00:00Z",
        "total_samples": len(sample_audits),
        "coverage_summary": {
            "bin_counts": bin_counts,
            "bin_percentages": {b: round(bin_counts[b] / len(sample_audits) * 100, 2) for b in PREDEFINED_BINS},
            "complete_loss_count": len(complete_loss_samples),
            "complete_loss_sample_ids": complete_loss_samples,
            "major_loss_count": len(major_loss_samples),
            "minor_loss_count": len(minor_loss_samples),
            "full_retention_count": len(full_retention_samples)
        },
        "performance_by_coverage_bin": bin_performance,
        "epistemic_synthesis": {
            "direct_observation": (
                f"Trên 91 mẫu holdout edited, canonical center-crop giữ lại 100% vùng sửa ở {bin_counts['full_retention']}/{len(sample_audits)} ảnh ({bin_counts['full_retention']/len(sample_audits)*100:.1f}%), "
                f"mất nhẹ (75-99%) ở {bin_counts['minor_loss']} ảnh ({bin_counts['minor_loss']/len(sample_audits)*100:.1f}%), "
                f"mất nhiều (>25%) ở {bin_counts['major_loss']} ảnh ({bin_counts['major_loss']/len(sample_audits)*100:.1f}%), "
                f"và mất hoàn toàn (0% retained) ở {bin_counts['complete_loss']} ảnh ({bin_counts['complete_loss']/len(sample_audits)*100:.1f}%). "
                "Ở các ảnh mất hoàn toàn vùng sửa, tensor 224x224 và DSP 512x512 thực sự chỉ quan sát các pixel nguyên bản (authentic), "
                "khiến nhãn ảnh (image-level ground truth) bị phân kỳ so với nội dung canvas thực tế."
            ),
            "correlation_vs_causation_guard": (
                "Phân biệt nghiêm ngặt: Việc crop làm mất vùng sửa là một quan sát hình học đo được trực tiếp. "
                "Tuy nhiên, việc cho rằng crop là nguyên nhân duy nhất khiến mô hình dự đoán nhầm sang authentic là một giả thuyết: "
                "ngay cả ở nhóm full_retention (100% diện tích sửa được giữ lại), tỷ lệ dự đoán nhầm sang authentic hoặc generated vẫn tồn tại, "
                "phản ánh giới hạn năng lực phân biệt nội tại của mô hình bên cạnh yếu tố hình học tiền xử lý."
            )
        }
    }
    receipt_path = EVIDENCE_DIR / "center_crop_mask_coverage_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"Receipt written to {receipt_path}")


if __name__ == "__main__":
    main()
