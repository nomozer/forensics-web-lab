"""Synthetic stand-in for the frozen 4C.3B world used by robustness tests.

Builds 341 tiny generated image pairs, a fake (non-reference) feature extractor,
ablation-format feature caches, real ablation early-fusion fold fits, a real
calibrated late-fusion run and its analysis summary, plus a robustness protocol
copy whose bindings point at these synthetic artifacts. Nothing here is real data.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
ROBUSTNESS_PROTOCOL = REPO_ROOT / "ml/configs/development_robustness_protocol.yaml"
LATE_FUSION_PROTOCOL = REPO_ROOT / "ml/configs/calibrated_late_fusion_protocol.yaml"
ABLATION_PROTOCOL = REPO_ROOT / "ml/configs/visual_dsp_ablation_protocol.yaml"
N_SOURCES = 341


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeExtractor:
    """Cheap deterministic stand-in for MobileNetV3 + DSP; sensitive to pixels and resolution."""

    is_reference_pipeline = False

    def __init__(self) -> None:
        rng = np.random.default_rng(1234)
        self.visual_projection = rng.normal(scale=0.4, size=(68, 576))
        self.dsp_projection = rng.normal(scale=0.4, size=(68, 16))
        self.calls = 0

    @staticmethod
    def _descriptor(image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        gray = np.asarray(rgb, dtype=np.float64).mean(axis=2) / 255.0
        small = np.asarray(rgb.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        laplacian = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
        stats = [laplacian.var() * 50.0, np.abs(np.diff(gray, axis=1)).mean() * 10.0, gray.std(), gray.shape[0] / 32.0]
        return np.concatenate([small.ravel(), stats])

    def extract(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        descriptors = np.stack([self._descriptor(image) for image in images])
        visual = np.tanh(descriptors @ self.visual_projection).astype(np.float32)
        dsp = np.tanh(descriptors @ self.dsp_projection).astype(np.float32)
        return visual, dsp

    def describe(self) -> dict[str, Any]:
        return {"extractor": "synthetic_fake_extractor", "device": "cpu"}


def _write_images(data_root: Path) -> list[dict[str, str]]:
    images = data_root / "images"
    images.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:32, 0:32]
    rows = []
    for i in range(N_SOURCES):
        rng = np.random.default_rng(i)
        base = (
            0.5 + 0.3 * np.sin(xx / (3 + i % 5)) * np.cos(yy / (4 + i % 3))
        )[..., None] * np.array([1.0, 0.9 - 0.1 * (i % 3), 0.8])
        base = base + rng.normal(scale=0.02, size=(32, 32, 3))
        edited = base.copy()
        if rng.random() < 0.8:  # label signal that is detectable but not perfect
            edited[8:24, 8:24] += rng.normal(scale=0.12, size=(16, 16, 3))
        paths = {}
        source = f"src_{i:03d}"
        for name, array in (("authentic", base), ("edited", edited)):
            path = images / f"{source}_{name}.png"
            Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8), "RGB").save(path)
            paths[name] = path
        rows.append(
            {
                "source_id": source,
                "partition": "development_train" if i < 250 else "inner_validation",
                "authentic_path": f"images/{source}_authentic.png",
                "canonical_edit_path": f"images/{source}_edited.png",
                "authentic_sha256": _sha256(paths["authentic"]),
                "canonical_edit_sha256": _sha256(paths["edited"]),
            }
        )
    rows.append(
        {
            "source_id": "locked_000",
            "partition": "locked_test",
            "authentic_path": "locked/never_read.png",
            "canonical_edit_path": "locked/never_read_edit.png",
            "authentic_sha256": "0" * 64,
            "canonical_edit_sha256": "1" * 64,
        }
    )
    return rows


def _write_cache(shared: Path, name: str, features: np.ndarray, source_ids: list[str], extra: dict | None = None) -> None:
    from ml.training.nested_cv_development import source_membership_commitment

    cache = shared / f"{name}_features.pt"
    torch.save({"features": torch.from_numpy(features), "source_ids": list(source_ids)}, cache)
    receipt = {
        "sources": len(source_ids),
        "samples": 2 * len(source_ids),
        "feature_dimension": int(features.shape[-1]),
        "source_membership_commitment": source_membership_commitment(source_ids),
        "cache_sha256": _sha256(cache),
        **(extra or {}),
    }
    (shared / f"{name}_features_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def build_robustness_fixture(root: Path) -> dict[str, Any]:
    from ml.training.calibrated_late_fusion import (
        build_run_bindings,
        feature_array_commitment,
        run_matrix,
        sha256_text_file,
    )
    from ml.training.calibrated_late_fusion import load_protocol as load_late_protocol
    from ml.training.development_robustness import load_condition_image, scope_samples
    from ml.training.dsp_features import DSP_FEATURE_NAMES
    from ml.training.run_visual_dsp_ablation import OuterFitSpec, execute_outer_fit
    from ml.training.visual_dsp_ablation import (
        build_grouped_nested_folds,
        load_development_pairs,
        public_fold_lock,
    )
    from scripts.research.analyze_calibrated_late_fusion import run_analysis

    data_root = root / "bundle"
    rows = _write_images(data_root)
    manifest = data_root / "manifest_pilot_a_option_p.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pairs = load_development_pairs(manifest_path=manifest, data_root=data_root, verify_hashes=True)
    source_ids = [p.source_id for p in pairs]
    folds = build_grouped_nested_folds(source_ids, outer_folds=5, inner_folds=4, outer_seed=42, inner_seed=1337)
    extractor = FakeExtractor()
    samples = scope_samples(pairs, folds, "full")
    visual, dsp = extractor.extract([load_condition_image(s.path, "original")[0] for s in samples])
    visual = visual.reshape(N_SOURCES, 2, -1)
    dsp = dsp.reshape(N_SOURCES, 2, -1)

    ablation_dir = root / "ablation_run"
    shared = ablation_dir / "shared"
    shared.mkdir(parents=True)
    _write_cache(shared, "visual", visual, source_ids)
    _write_cache(shared, "dsp", dsp, source_ids, {"feature_names": list(DSP_FEATURE_NAMES)})
    ablation_protocol = yaml.safe_load(ABLATION_PROTOCOL.read_text(encoding="utf-8"))
    fusion = np.concatenate([visual, dsp], axis=-1)
    for k in range(5):
        execute_outer_fit(
            spec=OuterFitSpec("visual_dsp_fusion", k),
            protocol=ablation_protocol,
            features=fusion,
            pairs=pairs,
            nested_folds=folds,
            output_root=ablation_dir,
        )

    late_dir = root / "late_fusion_run"
    late_protocol = load_late_protocol(LATE_FUSION_PROTOCOL)
    bindings = build_run_bindings(
        protocol_path=LATE_FUSION_PROTOCOL,
        data_origin="synthetic_fixture",
        source_ids=source_ids,
        visual_features=visual,
        dsp_features=dsp,
        fold_lock=public_fold_lock(folds),
    )
    receipts = run_matrix(
        output_dir=late_dir, outer_folds=range(5), folds=folds, visual_features=visual, dsp_features=dsp,
        source_ids=source_ids, protocol=late_protocol, bindings=bindings, log=lambda _: None,
    )
    summary = run_analysis(
        output_dir=late_dir, report_dir=root / "late_fusion_report", protocol_path=LATE_FUSION_PROTOCOL,
        ablation_dir=ablation_dir,
    )
    summary_path = root / "late_fusion_report" / "analysis_summary.json"

    protocol = yaml.safe_load(ROBUSTNESS_PROTOCOL.read_text(encoding="utf-8"))
    late_spec = protocol["frozen_models"]["late_fusion"]
    late_spec["run_manifest_sha256"] = sha256_text_file(late_dir / "run_manifest.json")
    late_spec["fold_predictions_sha256"] = [r["predictions_sha256"] for r in receipts]
    late_spec["reference_summary_path"] = str(summary_path)
    late_spec["reference_summary_sha256"] = sha256_text_file(summary_path)
    protocol["frozen_models"]["feature_caches"] = {
        "visual_feature_commitment": feature_array_commitment(visual),
        "dsp_feature_commitment": feature_array_commitment(dsp),
    }
    protocol["artifact_bindings"]["dataset_archive"]["manifest_sha256"] = _sha256(manifest)
    protocol_path = root / "robustness_protocol_synthetic.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    return {
        "root": root,
        "data_root": data_root,
        "manifest": manifest,
        "ablation_dir": ablation_dir,
        "late_fusion_dir": late_dir,
        "protocol_path": protocol_path,
        "late_summary": summary,
        "pairs": pairs,
        "folds": folds,
        "extractor_factory": FakeExtractor,
    }
