#!/usr/bin/env python3
"""
Create dummy baseline prediction artifact for Phase 4C.1B.6R.1

Generates predictions for StratifiedDummyClassifier on inner_validation partition
using the same protocol as the training runner:
- Fit on development_train class distribution
- Predict on inner_validation (1 canonical variant per source: 91 source_ids x 2 = 182 samples)
"""

import sys
sys.path.insert(0, r"D:\Documents\forensics-web-lab")

import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyClassifier

MANIFEST_PATH = r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\smoke_n50_seed42\extracted_bundle\manifest_pilot_a_option_p.csv"
REPO_ROOT = r"D:\Documents\forensics-web-lab-local-artifacts\phase_4c1\smoke_n50_seed42\extracted_bundle"
OUTPUT_DIR = r"D:\Documents\forensics-web-lab\research\evidence\phase-4c.1b.6r.1"

def load_manifest(partition):
    """Load manifest and extract rows for given partition."""
    instances = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["partition"] == partition:
                instances.append(row)
    return instances

def create_dummy_predictions(dev_instances, val_instances, random_state=42):
    """
    Create dummy predictions:
    - Fit DummyClassifier on development_train class distribution
    - Predict on inner_validation
    """
    # Build training labels from development_train (1 authentic + 1 ai_edited per source)
    y_train = []
    for inst in dev_instances:
        y_train.append(0)  # authentic
        y_train.append(1)  # ai_edited
    y_train = np.array(y_train)
    
    # Fit dummy on training class distribution
    dummy = DummyClassifier(strategy="stratified", random_state=random_state)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    
    # Predict on inner_validation (1 authentic + 1 ai_edited per source)
    y_true = []
    source_ids = []
    sample_ids = []
    
    for inst in val_instances:
        source_id = inst["source_id"]
        # authentic sample
        y_true.append(0)
        source_ids.append(source_id)
        sample_ids.append(f"{source_id}_authentic")
        # ai_edited sample
        y_true.append(1)
        source_ids.append(source_id)
        sample_ids.append(f"{source_id}_ai_edited")
    
    y_true = np.array(y_true)
    source_ids = np.array(source_ids)
    sample_ids = np.array(sample_ids)
    
    # Predict
    y_pred = dummy.predict(np.zeros((len(y_true), 1)))
    probs = dummy.predict_proba(np.zeros((len(y_true), 1)))
    y_prob = probs[:, 1]  # probability of class 1 (ai_edited)
    
    # Create prediction records
    predictions = []
    for i in range(len(y_true)):
        predictions.append({
            "sample_id": sample_ids[i],
            "source_id": source_ids[i],
            "partition": "inner_validation",
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "probability": float(y_prob[i]),
            "method": "StratifiedDummyClassifier",
            "random_seed": random_state
        })
    
    return predictions

def compute_metrics(predictions):
    """Compute metrics from predictions."""
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    import numpy as np
    
    y_true = np.array([p["y_true"] for p in predictions])
    y_pred = np.array([p["y_pred"] for p in predictions])
    y_prob = np.array([p["probability"] for p in predictions])
    
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5
    brier = float(np.mean((y_prob - y_true) ** 2))
    
    # ECE
    from ml.evaluation.metrics import compute_ece
    ece = compute_ece(y_true, y_prob)
    
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "auroc": auroc,
        "brier_score": brier,
        "ece": ece
    }

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Loading development_train manifest...")
    dev_instances = load_manifest("development_train")
    print(f"Found {len(dev_instances)} development_train source_ids")
    
    print("Loading inner_validation manifest...")
    val_instances = load_manifest("inner_validation")
    print(f"Found {len(val_instances)} inner_validation source_ids")
    
    print("Creating dummy predictions...")
    predictions = create_dummy_predictions(dev_instances, val_instances, random_state=42)
    print(f"Generated {len(predictions)} predictions ({len(predictions)//2} source_ids x 2)")
    
    # Compute metrics
    metrics = compute_metrics(predictions)
    print(f"Metrics: {metrics}")
    
    # Save predictions
    pred_path = Path(OUTPUT_DIR) / "predictions-dummy-seed42.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {pred_path}")
    
    # Save metrics
    metrics_path = Path(OUTPUT_DIR) / "dummy-baseline-metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Create receipt
    receipt = {
        "implementation_path": "scripts/research/create_dummy_baseline_artifact.py",
        "reproduction_command": "python scripts/research/create_dummy_baseline_artifact.py",
        "random_seed": 42,
        "input_manifest": MANIFEST_PATH,
        "input_manifest_sha256": compute_sha256(Path(MANIFEST_PATH)),
        "output_predictions_sha256": compute_sha256(pred_path),
        "output_metrics_sha256": compute_sha256(metrics_path),
        "source_count": len(val_instances),
        "sample_count": len(predictions),
        "class_distribution": {
            "authentic": len([p for p in predictions if p["y_true"] == 0]),
            "ai_edited": len([p for p in predictions if p["y_true"] == 1])
        },
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"]
    }
    
    receipt_path = Path(OUTPUT_DIR) / "dummy-baseline-receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Saved receipt to {receipt_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()