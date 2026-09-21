import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from ml.datasets.manifest import DatasetRecord, ManifestManager
from ml.datasets.split import split_manifest_by_group
from ml.training.loss import FocalLoss
from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
from ml.evaluation.calibration import compute_ece, TemperatureScaler

def test_mobilenetv3_forward_shape():
    model = MobileNetV3Forensics(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out.shape == (2, 3)
    assert model.count_parameters() < 15_000_000  # Stays well within budget

def test_focal_loss_computation():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    loss = loss_fn(logits, targets)

    assert loss.item() > 0.0
    assert torch.isfinite(loss)

def test_manifest_csv_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_manifest.csv"
        records = [
            DatasetRecord(
                sample_id="test_001",
                source_id="src_001",
                image_path="test_001.jpg",
                label="fully_generated",
                generator="stable_diffusion",
                generator_version="1.5",
                edit_type="full_synthesis",
                mask_path="",
                dataset_name="GenImage",
                dataset_version="1.0",
                split="train",
                license="CC-BY-NC 4.0",
                width=512,
                height=512,
                sha256="abc123",
            )
        ]

        ManifestManager.save_csv(records, csv_path)
        loaded = ManifestManager.load_csv(csv_path)

        assert len(loaded) == 1
        assert loaded[0].sample_id == "test_001"
        assert loaded[0].label == "fully_generated"

def test_group_split_prevents_leakage():
    records = [
        DatasetRecord(
            sample_id=f"sample_{i}",
            source_id="common_parent_1",
            image_path=f"img_{i}.jpg",
            label="ai_edited" if i % 2 == 0 else "authentic",
            generator="camera",
            generator_version="1.0",
            edit_type="none",
            mask_path="",
            dataset_name="TestSet",
            dataset_version="1.0",
            split="train",
            license="MIT",
            width=256,
            height=256,
            sha256="",
        )
        for i in range(10)
    ]

    split_records = split_manifest_by_group(records, train_ratio=0.7, val_ratio=0.15)
    splits = {r.split for r in split_records}
    # All records belonging to common_parent_1 must be in the exact same split!
    assert len(splits) == 1

def test_temperature_scaler_and_ece():
    np.random.seed(42)
    logits = np.array([[2.0, 0.5, 0.1], [0.1, 2.5, 0.2], [0.3, 0.2, 2.2]])
    labels = np.array([0, 1, 2])

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)

    probs = scaler.predict_proba(logits)
    ece = compute_ece(probs, labels, num_bins=5)

    assert 0.0 <= ece <= 1.0
    assert scaler.temperature > 0.0
