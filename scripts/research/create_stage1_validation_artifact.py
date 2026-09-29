#!/usr/bin/env python3
"""
Create Stage 1 validation-only predictions artifact from the full predictions.
Filters to only inner_validation partition (91 source_ids, 182 samples).
"""

import sys
sys.path.insert(0, r"D:\Documents\forensics-web-lab")

import json
import csv
import hashlib
from pathlib import Path

# Paths
FULL_PREDICTIONS_PATH = r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\runs\n50_seed42\predictions.json"
MANIFEST_PATH = r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\smoke_n50_seed42\extracted_bundle\manifest_pilot_a_option_p.csv"
OUTPUT_DIR = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.2"

def load_manifest():
    """Load manifest and get inner_validation source_ids."""
    inner_val_sources = set()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["partition"] == "inner_validation":
                inner_val_sources.add(row["source_id"])
    return inner_val_sources

def load_full_predictions():
    """Load the full predictions from the run."""
    with open(FULL_PREDICTIONS_PATH, "r") as f:
        data = json.load(f)
    return data

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Loading manifest...")
    inner_val_sources = load_manifest()
    print(f"Found {len(inner_val_sources)} inner_validation source_ids")
    
    print("Loading full predictions...")
    full_data = load_full_predictions()
    
    predictions = full_data["predictions"]
    probabilities = full_data["probabilities"]
    targets = full_data["targets"]
    source_ids = full_data["source_ids"]
    
    print(f"Full data: {len(predictions)} predictions, {len(set(source_ids))} unique source_ids")
    
    # Filter to inner_validation only
    val_predictions = []
    val_probabilities = []
    val_targets = []
    val_source_ids = []
    
    for i, sid in enumerate(source_ids):
        if sid in inner_val_sources:
            val_predictions.append(predictions[i])
            val_probabilities.append(probabilities[i])
            val_targets.append(targets[i])
            val_source_ids.append(sid)
    
    print(f"Validation-only: {len(val_predictions)} predictions, {len(set(val_source_ids))} unique source_ids")
    
    # Verify structure: each source_id should appear exactly twice (authentic + ai_edited)
    from collections import Counter
    source_counts = Counter(val_source_ids)
    for sid, count in source_counts.items():
        if count != 2:
            print(f"WARNING: source_id {sid} has {count} samples (expected 2)")
    
    # Create detailed records
    records = []
    for i in range(len(val_predictions)):
        records.append({
            "sample_id": f"{val_source_ids[i]}_{'authentic' if val_targets[i] == 0 else 'ai_edited'}",
            "source_id": val_source_ids[i],
            "partition": "inner_validation",
            "y_true": val_targets[i],
            "y_pred": val_predictions[i],
            "probability_ai_edited": val_probabilities[i],
            "seed": 42,
            "method": "Frozen_MobileNetV3_Small_Stage1"
        })
    
    # Save validation-only predictions
    output_path = Path(OUTPUT_DIR) / "predictions-stage1-seed42-validation-only.json"
    with open(output_path, "w") as f:
        json.dump({
            "records": records,
            "summary": {
                "total_records": len(records),
                "unique_source_ids": len(set(val_source_ids)),
                "authentic_count": sum(1 for r in records if r["y_true"] == 0),
                "ai_edited_count": sum(1 for r in records if r["y_true"] == 1),
                "partitions": ["inner_validation"],
                "development_records": 0,
                "locked_test_records": 0
            }
        }, f, indent=2)
    print(f"Saved to {output_path}")
    
    # Create receipt
    receipt = {
        "source_artifact": FULL_PREDICTIONS_PATH,
        "source_sha256": compute_sha256(Path(FULL_PREDICTIONS_PATH)),
        "source_total_records": len(predictions),
        "source_unique_source_ids": len(set(source_ids)),
        "filter_criterion": "partition == inner_validation",
        "manifest_used": MANIFEST_PATH,
        "output_artifact": str(output_path),
        "output_sha256": compute_sha256(output_path),
        "output_records": len(records),
        "output_unique_source_ids": len(set(val_source_ids)),
        "output_authentic_count": sum(1 for r in records if r["y_true"] == 0),
        "output_ai_edited_count": sum(1 for r in records if r["y_true"] == 1)
    }
    
    receipt_path = Path(OUTPUT_DIR) / "stage1-prediction-receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved receipt to {receipt_path}")
    
    # Verify metrics match reported values
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    import numpy as np
    from ml.evaluation.metrics import compute_ece
    
    y_true = np.array([r["y_true"] for r in records])
    y_pred = np.array([r["y_pred"] for r in records])
    y_prob = np.array([r["probability_ai_edited"] for r in records])
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob))
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = compute_ece(y_true, y_prob)
    
    print(f"\nVerification metrics:")
    print(f"  Macro-F1: {macro_f1:.12f} (expected: 0.565816095425034)")
    print(f"  Balanced Acc: {balanced_acc:.12f} (expected: 0.565934)")
    print(f"  AUROC: {auroc:.12f} (expected: 0.611762)")
    print(f"  Brier: {brier:.12f}")
    print(f"  ECE: {ece:.12f}")
    
    # Check tolerance
    expected_f1 = 0.565816095425034
    expected_ba = 0.565934
    expected_auroc = 0.611762
    
    assert abs(macro_f1 - expected_f1) < 1e-10, f"Macro-F1 mismatch: {macro_f1} vs {expected_f1}"
    assert abs(balanced_acc - expected_ba) < 1e-10, f"Balanced Acc mismatch: {balanced_acc} vs {expected_ba}"
    assert abs(auroc - expected_auroc) < 1e-10, f"AUROC mismatch: {auroc} vs {expected_auroc}"
    
    print("\nAll metrics verified within tolerance!")

if __name__ == "__main__":
    main()