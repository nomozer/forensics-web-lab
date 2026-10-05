"""Development robustness of the frozen Phase 4C.3B models under image transforms.

Every outer-test image of the 341 development sources is decoded, transformed by
one locked condition (JPEG re-encode and/or bicubic down-scaling), its visual and
DSP features are re-extracted from the transformed pixels, and it is scored by
the already-fitted models of its own outer fold:

  visual_calibrated    late-fusion fold model: visual scaler/LR, temperature
  late_fusion_stacked  late-fusion fold model: both scorers, temperatures, stacker
  early_fusion         ablation visual_dsp_fusion fold model: scaler/LR on 592-d

Nothing is fitted here. Fitted estimators are rebuilt from stored parameters and
only `transform` / `decision_function` / `predict_proba` are called. Before any
transformed condition runs, the original condition must reproduce the stored
4C.3B predictions (the original-reproduction gate).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from ml.training.calibrated_late_fusion import (
    FOLD_RECEIPT_NAME,
    MODEL_NAME,
    PREDICTIONS_NAME,
    RUN_MANIFEST_NAME,
    atomic_write_json,
    atomic_write_text,
    feature_array_commitment,
    fold_directory,
    sha256_bytes,
    sha256_file,
    sha256_text_file,
    sigmoid,
    utc_now,
    verify_fold_artifacts,
)
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "1.0.0"
EXPERIMENT_ID = "development_robustness"
PROTOCOL_STATUS = "DEVELOPMENT_PROTOCOL_LOCKED_PRE_EXECUTION"
RECIPE_IDS = ("visual_calibrated", "early_fusion", "late_fusion_stacked")
CANONICAL_CONDITIONS: dict[str, tuple[dict[str, Any], ...]] = {
    "original": (),
    "jpeg_q95": ({"op": "jpeg", "quality": 95},),
    "jpeg_q75": ({"op": "jpeg", "quality": 75},),
    "jpeg_q50": ({"op": "jpeg", "quality": 50},),
    "resize_0.5": ({"op": "resize", "scale": 0.5},),
    "resize_0.5_jpeg_q75": ({"op": "resize", "scale": 0.5}, {"op": "jpeg", "quality": 75}),
}
CONDITION_IDS = tuple(CANONICAL_CONDITIONS)
LABEL_NAMES = ("authentic", "ai_edited")
DATA_ORIGIN_REAL = "development_real"
DATA_ORIGIN_SYNTHETIC = "synthetic_only"
SCOPES = ("pilot", "full")
EARLY_FUSION_RECIPE_DIR = "visual_dsp_fusion"
PREDICTION_FIELDS = (
    "condition",
    "recipe",
    "outer_fold",
    "source_id",
    "sample_id",
    "label",
    "label_id",
    "prediction",
    "probability_ai_edited",
    "logit_ai_edited",
)
REFERENCE_METRIC_KEYS = (
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "brier_score",
    "ece",
    "fpr",
    "fnr",
    "tn",
    "fp",
    "fn",
    "tp",
)


class MissingArtifactsError(FileNotFoundError):
    """Raised with the exact list of fitted artifacts that are absent."""

    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__(
            "Required frozen artifacts are missing (nothing will be refit):\n  - "
            + "\n  - ".join(self.missing)
        )


class GateError(RuntimeError):
    """Raised when the original-reproduction gate does not allow the requested work."""


# --------------------------------------------------------------------------- protocol


def load_protocol(path: Path | str) -> dict[str, Any]:
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Protocol experiment_id must be {EXPERIMENT_ID!r}")
    if protocol.get("status") != PROTOCOL_STATUS:
        raise ValueError(f"Protocol is not in {PROTOCOL_STATUS} state")
    if tuple(r["id"] for r in protocol["recipes"]) != RECIPE_IDS:
        raise ValueError(f"Protocol recipes must be exactly {RECIPE_IDS}")
    conditions = {c["id"]: tuple(dict(op) for op in c["operations"]) for c in protocol["conditions"]}
    if list(conditions) != list(CONDITION_IDS) or conditions != CANONICAL_CONDITIONS:
        raise ValueError("Protocol conditions differ from the implemented canonical conditions")
    if protocol["frozen_models"].get("fitting_allowed") is not False:
        raise ValueError("Protocol must forbid fitting")
    if "locked_test" not in protocol["data_scope"]["forbidden_partitions"]:
        raise ValueError("Protocol must forbid the locked_test partition")
    primary = protocol["endpoints"]["primary"]
    if (primary["recipe"], primary["comparison"], primary["metric"]) != (
        "late_fusion_stacked",
        "jpeg_q75_minus_original",
        "macro_f1",
    ):
        raise ValueError("Primary endpoint differs from the locked definition")
    return protocol


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


# --------------------------------------------------------------------------- image transforms


def round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def resized_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round_half_up(width * scale)), max(1, round_half_up(height * scale))


def encode_jpeg(rgb: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(quality), subsampling=2, optimize=False, progressive=False)
    return buffer.getvalue()


def apply_operations(rgb: Image.Image, operations: Sequence[dict[str, Any]]) -> tuple[Image.Image, bytes | None]:
    """Apply locked operations in order to an RGB image.

    Returns the final RGB image and, when the last operation is JPEG, its exact
    encoded bytes (so a saved copy is the file the model actually saw).
    """
    if rgb.mode != "RGB":
        raise ValueError("Transforms operate on RGB images only")
    image = rgb
    encoded: bytes | None = None
    for operation in operations:
        if operation["op"] == "resize":
            size = resized_size(image.width, image.height, float(operation["scale"]))
            image = image.resize(size, resample=Image.Resampling.BICUBIC, reducing_gap=None)
            encoded = None
        elif operation["op"] == "jpeg":
            encoded = encode_jpeg(image, int(operation["quality"]))
            with Image.open(io.BytesIO(encoded)) as decoded:
                image = decoded.convert("RGB")
        else:
            raise ValueError(f"Unknown operation {operation!r}")
    return image, encoded


def pixel_digest(image: Image.Image) -> str:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    header = json.dumps({"shape": list(array.shape)}).encode("utf-8")
    return sha256_bytes(header + b"\n" + np.ascontiguousarray(array).tobytes())


def load_condition_image(path: Path, condition: str) -> tuple[Image.Image, bytes | None]:
    """Decode one file and apply one condition. `original` is the untouched 4C.3B path."""
    operations = CANONICAL_CONDITIONS[condition]
    with Image.open(path) as handle:
        handle.load()
        if not operations:
            return handle.copy(), None
        rgb = handle.convert("RGB")
    return apply_operations(rgb, operations)


def image_library_versions() -> dict[str, str | None]:
    import PIL
    from PIL import features

    return {
        "pillow": PIL.__version__,
        "libjpeg": features.version("jpg"),
        "zlib": features.version("zlib"),
    }


# --------------------------------------------------------------------------- feature extraction


class FeatureExtractor(Protocol):
    is_reference_pipeline: bool

    def extract(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]: ...

    def describe(self) -> dict[str, Any]: ...


class ReferenceFeatureExtractor:
    """The exact 4C.3B feature path: frozen MobileNetV3-Small pooled features + 16 DSP features."""

    is_reference_pipeline = True

    def __init__(self, weights_path: Path, device: str = "cpu", expected_weights_sha256: str | None = None):
        import torch
        from ml.training.mobilenetv3_forensics import MobileNetV3Forensics
        from ml.training.phase_4c2h_development import (
            apply_frozen_backbone_policy,
            build_canonical_transform,
            set_deterministic_seed,
        )

        if not weights_path.is_file():
            raise MissingArtifactsError([f"pretrained weights: {weights_path}"])
        actual = sha256_file(weights_path)
        if expected_weights_sha256 and actual != expected_weights_sha256:
            raise ValueError(f"Weights SHA mismatch: {actual} != {expected_weights_sha256}")
        self.torch = torch
        self.device = torch.device(device)
        set_deterministic_seed(0)
        self.model = MobileNetV3Forensics(
            num_classes=2, pretrained=False, weights_path=str(weights_path), freeze_backbone=True
        ).to(self.device)
        self.model.eval()
        apply_frozen_backbone_policy(self.model)
        self.transform = build_canonical_transform()
        self.weights_sha256 = actual

    def extract(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        from ml.training.dsp_features import extract_dsp_features

        torch = self.torch
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        with torch.no_grad():
            visual = torch.flatten(self.model.avgpool(self.model.features(batch)), 1).detach().cpu().numpy()
        dsp = np.stack([extract_dsp_features(image) for image in images]).astype(np.float32)
        return visual.astype(np.float32), dsp

    def describe(self) -> dict[str, Any]:
        return {
            "extractor": "reference_mobilenet_v3_small_plus_dsp16",
            "device": str(self.device),
            "torch": self.torch.__version__,
            "weights_sha256": self.weights_sha256,
        }


# --------------------------------------------------------------------------- frozen models


def _scaler(mean: Sequence[float], scale: Sequence[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    scaler.scale_ = np.asarray(scale, dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = scaler.mean_.size
    scaler.n_samples_seen_ = np.int64(0)
    return scaler


def _logistic(coef: Any, intercept: Any, dtype: Any) -> LogisticRegression:
    model = LogisticRegression()
    model.coef_ = np.asarray(coef, dtype=dtype).reshape(1, -1)
    model.intercept_ = np.asarray(intercept, dtype=dtype).reshape(1)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = model.coef_.shape[1]
    return model


@dataclass(frozen=True)
class FrozenFoldModels:
    outer_fold: int
    visual_scaler: StandardScaler
    visual_model: LogisticRegression
    dsp_scaler: StandardScaler
    dsp_model: LogisticRegression
    temperatures: dict[str, float]
    stacker: LogisticRegression
    early_scaler: StandardScaler
    early_model: LogisticRegression
    parameter_digest: str


def frozen_from_artifacts(outer_fold: int, late_model: dict[str, Any], early_model: dict[str, Any]) -> FrozenFoldModels:
    """Rebuild estimators exactly as they were fitted (base LR float32, stacker float64)."""
    base = late_model["base"]
    digest_payload = json.dumps(
        {
            "late": late_model,
            "early": {k: np.asarray(v).tolist() for k, v in early_model.items() if k != "classes"},
        },
        sort_keys=True,
    ).encode("utf-8")
    return FrozenFoldModels(
        outer_fold=outer_fold,
        visual_scaler=_scaler(base["visual"]["scaler_mean"], base["visual"]["scaler_scale"]),
        visual_model=_logistic(base["visual"]["coef"], base["visual"]["intercept"], np.float32),
        dsp_scaler=_scaler(base["dsp"]["scaler_mean"], base["dsp"]["scaler_scale"]),
        dsp_model=_logistic(base["dsp"]["coef"], base["dsp"]["intercept"], np.float32),
        temperatures={m: float(late_model["temperatures"][m]) for m in ("visual", "dsp")},
        stacker=_logistic(late_model["stacker"]["coef"], late_model["stacker"]["intercept"], np.float64),
        early_scaler=_scaler(early_model["scaler_mean"], early_model["scaler_scale"]),
        early_model=_logistic(
            early_model["coef"], early_model["intercept"], np.asarray(early_model["coef"]).dtype
        ),
        parameter_digest=sha256_bytes(digest_payload),
    )


def score_with_frozen_models(
    models: FrozenFoldModels, visual: np.ndarray, dsp: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {recipe: (logits, probabilities)}; each row uses only that image's features."""
    visual = np.asarray(visual, dtype=np.float32)
    dsp = np.asarray(dsp, dtype=np.float32)
    if visual.ndim != 2 or dsp.ndim != 2 or visual.shape[0] != dsp.shape[0]:
        raise ValueError("Expected aligned 2D per-image feature matrices")
    z_visual = models.visual_model.decision_function(models.visual_scaler.transform(visual)).astype(np.float64)
    z_dsp = models.dsp_model.decision_function(models.dsp_scaler.transform(dsp)).astype(np.float64)
    calibrated_visual = z_visual / models.temperatures["visual"]
    calibrated_dsp = z_dsp / models.temperatures["dsp"]
    stacked = models.stacker.decision_function(np.column_stack([calibrated_visual, calibrated_dsp])).astype(
        np.float64
    )
    early_scaled = models.early_scaler.transform(np.concatenate([visual, dsp], axis=1))
    early_logits = models.early_model.decision_function(early_scaled).astype(np.float64)
    early_probabilities = models.early_model.predict_proba(early_scaled)[:, 1].astype(np.float64)
    return {
        "visual_calibrated": (calibrated_visual, sigmoid(calibrated_visual)),
        "early_fusion": (early_logits, early_probabilities),
        "late_fusion_stacked": (stacked, sigmoid(stacked)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_torch_artifact(path: Path) -> dict[str, Any]:
    import torch

    # The ablation stored numpy arrays with torch.save; its SHA-256 was verified
    # against the ablation fit receipt before this call.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {key: np.asarray(value) for key, value in payload.items()}


def load_frozen_artifacts(
    *,
    protocol: dict[str, Any],
    late_fusion_dir: Path,
    ablation_dir: Path,
) -> dict[str, Any]:
    """Verify and load every fitted artifact; raise MissingArtifactsError listing absent files."""
    outer_folds = int(protocol["grouped_nested_cv"]["outer_folds"])
    late_spec = protocol["frozen_models"]["late_fusion"]
    early_spec = protocol["frozen_models"]["early_fusion"]
    required = [late_fusion_dir / RUN_MANIFEST_NAME]
    for k in range(outer_folds):
        late_dir = fold_directory(late_fusion_dir, k)
        required += [late_dir / FOLD_RECEIPT_NAME, late_dir / PREDICTIONS_NAME, late_dir / MODEL_NAME]
        early_dir = ablation_dir / "fits" / early_spec["recipe"] / f"outer_{k}"
        required += [early_dir / early_spec["receipt_file"], early_dir / early_spec["model_file"], early_dir / "predictions.csv"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise MissingArtifactsError(missing)

    manifest_path = late_fusion_dir / RUN_MANIFEST_NAME
    if sha256_text_file(manifest_path) != late_spec["run_manifest_sha256"]:
        raise ValueError("Late-fusion run_manifest.json does not match the protocol binding")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = manifest["bindings"]
    if bindings["protocol_sha256"] != late_spec["protocol_sha256"]:
        raise ValueError("Late-fusion run was produced under a different protocol")

    late_models: list[dict[str, Any]] = []
    early_models: list[dict[str, Any]] = []
    reference_rows: list[dict[str, str]] = []
    early_rows: list[dict[str, str]] = []
    artifact_hashes: dict[str, Any] = {"late_fusion": [], "early_fusion": []}
    for k in range(outer_folds):
        receipt = verify_fold_artifacts(late_fusion_dir, k, bindings)
        if receipt is None:
            raise MissingArtifactsError([str(fold_directory(late_fusion_dir, k) / FOLD_RECEIPT_NAME)])
        if receipt["predictions_sha256"] != late_spec["fold_predictions_sha256"][k]:
            raise ValueError(f"Late-fusion fold {k} predictions differ from the protocol binding")
        late_dir = fold_directory(late_fusion_dir, k)
        late_models.append(json.loads((late_dir / MODEL_NAME).read_text(encoding="utf-8")))
        reference_rows += _read_csv(late_dir / PREDICTIONS_NAME)
        artifact_hashes["late_fusion"].append(
            {"outer_fold": k, "model_sha256": receipt["model_sha256"], "predictions_sha256": receipt["predictions_sha256"]}
        )

        early_dir = ablation_dir / "fits" / early_spec["recipe"] / f"outer_{k}"
        early_receipt = json.loads((early_dir / early_spec["receipt_file"]).read_text(encoding="utf-8"))
        problems = []
        if early_receipt.get("status") != "COMPLETED" or early_receipt.get("outer_fold") != k:
            problems.append("status/outer_fold")
        if early_receipt.get("recipe_id") != early_spec["recipe"]:
            problems.append("recipe_id")
        if sha256_file(early_dir / early_spec["model_file"]) != early_receipt.get("model_sha256"):
            problems.append("model hash")
        if sha256_file(early_dir / "predictions.csv") != early_receipt.get("predictions_sha256"):
            problems.append("predictions hash")
        if problems:
            raise ValueError(f"Ablation early-fusion fold {k} fails verification: {', '.join(problems)}")
        early_models.append(_load_torch_artifact(early_dir / early_spec["model_file"]))
        early_rows += _read_csv(early_dir / "predictions.csv")
        artifact_hashes["early_fusion"].append(
            {
                "outer_fold": k,
                "model_sha256": early_receipt["model_sha256"],
                "predictions_sha256": early_receipt["predictions_sha256"],
            }
        )

    frozen = [frozen_from_artifacts(k, late_models[k], early_models[k]) for k in range(outer_folds)]
    reference: dict[tuple[str, str], dict[str, Any]] = {}
    for row in reference_rows:
        if row["recipe"] in ("visual_calibrated", "late_fusion_stacked"):
            reference[(row["recipe"], row["sample_id"])] = row
    for row in early_rows:
        reference[("early_fusion", row["sample_id"])] = {**row, "recipe": "early_fusion"}
    return {
        "frozen": frozen,
        "reference": reference,
        "late_fusion_bindings": bindings,
        "artifact_hashes": artifact_hashes,
    }


# --------------------------------------------------------------------------- folds and scoring


def verify_fold_binding(
    *, folds: Sequence[Any], bindings: dict[str, Any], reference: dict[tuple[str, str], dict[str, Any]]
) -> dict[int, int]:
    """Check folds equal the 4C.3B fold lock and reference rows; return source -> outer fold."""
    from ml.training.phase_4c2h_development import public_fold_lock

    if public_fold_lock(folds) != bindings["fold_lock"]:
        raise ValueError("Rebuilt outer folds differ from the 4C.3B fold lock")
    fold_of_source: dict[str, int] = {}
    for fold in folds:
        for source in fold.outer_test:
            if source in fold_of_source:
                raise ValueError(f"Source {source} appears in two outer-test folds")
            fold_of_source[source] = fold.outer_fold
    for (recipe, sample_id), row in reference.items():
        source = row["source_id"]
        if fold_of_source.get(source) != int(row["outer_fold"]):
            raise ValueError(f"Reference {recipe} row for {sample_id} has a different outer fold")
    return fold_of_source


@dataclass(frozen=True)
class SampleRef:
    source_id: str
    label_id: int
    path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.source_id}:{LABEL_NAMES[self.label_id]}"


def scope_samples(pairs: Sequence[Any], folds: Sequence[Any], scope: str, pilot_fold: int = 0) -> list[SampleRef]:
    """Samples ordered by source_id then (authentic, ai_edited), like the 4C.3B caches."""
    allowed = set(folds[pilot_fold].outer_test) if scope == "pilot" else None
    samples: list[SampleRef] = []
    for pair in pairs:
        if allowed is not None and pair.source_id not in allowed:
            continue
        samples.append(SampleRef(pair.source_id, 0, Path(pair.authentic.absolute_path)))
        samples.append(SampleRef(pair.source_id, 1, Path(pair.edited.absolute_path)))
    return samples


def score_samples(
    *,
    condition: str,
    samples: Sequence[SampleRef],
    visual: np.ndarray,
    dsp: np.ndarray,
    folds: Sequence[Any],
    frozen: Sequence[FrozenFoldModels],
) -> list[dict[str, Any]]:
    """Score every sample with the models of its own outer fold.

    Rows are grouped per fold in the 4C.3B order (fold.outer_test order, authentic
    then ai_edited) so the original condition is numerically identical to 4C.3B.
    """
    index = {(s.source_id, s.label_id): i for i, s in enumerate(samples)}
    rows_by_recipe: dict[str, list[dict[str, Any]]] = {recipe: [] for recipe in RECIPE_IDS}
    for fold in folds:
        keys = [(source, label) for source in fold.outer_test for label in (0, 1) if (source, label) in index]
        if not keys:
            continue
        positions = [index[key] for key in keys]
        scored = score_with_frozen_models(frozen[fold.outer_fold], visual[positions], dsp[positions])
        for recipe in RECIPE_IDS:
            logits, probabilities = scored[recipe]
            for (source, label), logit, probability in zip(keys, logits, probabilities):
                rows_by_recipe[recipe].append(
                    {
                        "condition": condition,
                        "recipe": recipe,
                        "outer_fold": fold.outer_fold,
                        "source_id": source,
                        "sample_id": f"{source}:{LABEL_NAMES[label]}",
                        "label": LABEL_NAMES[label],
                        "label_id": label,
                        "prediction": int(probability >= 0.5),
                        "probability_ai_edited": float(probability),
                        "logit_ai_edited": float(logit),
                    }
                )
    rows = [row for recipe in RECIPE_IDS for row in rows_by_recipe[recipe]]
    if len(rows) != len(RECIPE_IDS) * len(samples):
        raise AssertionError("Every scoped sample must be scored exactly once per recipe")
    return rows


def render_rows_csv(rows: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(PREDICTION_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (repr(row[k]) if isinstance(row[k], float) else row[k]) for k in PREDICTION_FIELDS})
    return buffer.getvalue()


def compare_to_reference(
    rows: Sequence[dict[str, Any]], reference: dict[tuple[str, str], dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    differences: dict[str, float] = {}
    decision_mismatches: dict[str, int] = {}
    tolerated_flips = 0
    missing = 0
    for recipe in RECIPE_IDS:
        worst = 0.0
        mismatches = 0
        for row in (r for r in rows if r["recipe"] == recipe):
            ref = reference.get((recipe, row["sample_id"]))
            if ref is None:
                missing += 1
                continue
            ref_probability = float(ref["probability_ai_edited"])
            worst = max(worst, abs(ref_probability - row["probability_ai_edited"]))
            if int(ref["prediction"]) != row["prediction"]:
                if abs(ref_probability - 0.5) <= tolerance:
                    tolerated_flips += 1
                else:
                    mismatches += 1
        differences[recipe] = worst
        decision_mismatches[recipe] = mismatches
    passed = missing == 0 and all(d <= tolerance for d in differences.values()) and not any(
        decision_mismatches.values()
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "tolerance": tolerance,
        "max_abs_probability_difference": differences,
        "decision_mismatches": decision_mismatches,
        "tolerated_threshold_flips": tolerated_flips,
        "missing_reference_rows": missing,
    }


def reference_metric_check(
    reference: dict[tuple[str, str], dict[str, Any]], summary_path: Path, expected_sha256: str, tolerance: float = 1e-12
) -> dict[str, Any]:
    """Recompute metrics of the stored reference predictions and compare with the committed summary."""
    from scripts.research.analyze_calibrated_late_fusion import point_metrics

    if not summary_path.is_file() or sha256_text_file(summary_path) != expected_sha256:
        return {"status": "FAIL", "reason": "reference summary missing or SHA mismatch"}
    committed = json.loads(summary_path.read_text(encoding="utf-8"))["metrics_by_recipe"]
    worst = 0.0
    for recipe in RECIPE_IDS:
        rows = sorted(
            (row for (r, _), row in reference.items() if r == recipe),
            key=lambda row: (row["source_id"], int(row["label_id"])),
        )
        labels = np.array([int(r["label_id"]) for r in rows])
        probabilities = np.array([float(r["probability_ai_edited"]) for r in rows])
        ours = point_metrics(labels, probabilities, 0.65)
        worst = max(worst, *(abs(ours[k] - float(committed[recipe][k])) for k in REFERENCE_METRIC_KEYS))
    return {"status": "PASS" if worst <= tolerance else "FAIL", "max_abs_difference": worst, "tolerance": tolerance}


def metrics_for_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    from scripts.research.analyze_calibrated_late_fusion import point_metrics

    out = {}
    for recipe in RECIPE_IDS:
        selected = sorted((r for r in rows if r["recipe"] == recipe), key=lambda r: (r["source_id"], r["label_id"]))
        labels = np.array([int(r["label_id"]) for r in selected])
        probabilities = np.array([float(r["probability_ai_edited"]) for r in selected])
        out[recipe] = point_metrics(labels, probabilities, 0.65)
    return out


# --------------------------------------------------------------------------- condition execution


def environment_record(extractor: FeatureExtractor) -> dict[str, Any]:
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        **image_library_versions(),
        **extractor.describe(),
    }


def extract_condition_features(
    *,
    condition: str,
    samples: Sequence[SampleRef],
    extractor: FeatureExtractor,
    batch_size: int,
    save_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    visual_parts: list[np.ndarray] = []
    dsp_parts: list[np.ndarray] = []
    image_records: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = []
        for sample in chunk:
            image, encoded = load_condition_image(sample.path, condition)
            record = {"sample_id": sample.sample_id, "pixel_sha256": pixel_digest(image), "size": list(image.size)}
            if save_dir is not None and condition != "original":
                save_dir.mkdir(parents=True, exist_ok=True)
                name = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()[:24]
                if encoded is not None:
                    target, payload = save_dir / f"{name}.jpg", encoded
                else:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG", compress_level=6)
                    target, payload = save_dir / f"{name}.png", buffer.getvalue()
                atomic_write_bytes(target, payload)
                record["saved_file"] = target.name
                record["saved_sha256"] = sha256_bytes(payload)
            images.append(image)
            image_records.append(record)
        visual, dsp = extractor.extract(images)
        visual_parts.append(np.asarray(visual, dtype=np.float32))
        dsp_parts.append(np.asarray(dsp, dtype=np.float32))
    visual_all = np.concatenate(visual_parts)
    dsp_all = np.concatenate(dsp_parts)
    if not (np.isfinite(visual_all).all() and np.isfinite(dsp_all).all()):
        raise ValueError(f"Non-finite features under condition {condition}")
    return visual_all, dsp_all, image_records


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _npz_bytes(visual: np.ndarray, dsp: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, visual=visual, dsp=dsp)
    return buffer.getvalue()


def condition_directory(scope_dir: Path, condition: str) -> Path:
    return scope_dir / "conditions" / condition


def verify_condition(scope_dir: Path, condition: str, run_bindings: dict[str, Any]) -> dict[str, Any] | None:
    directory = condition_directory(scope_dir, condition)
    receipt_path = directory / "condition_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    problems = []
    if receipt.get("status") != "COMPLETED" or receipt.get("condition") != condition:
        problems.append("status/condition")
    if receipt.get("run_bindings") != run_bindings:
        problems.append("run bindings")
    for name, field in (("features.npz", "features_sha256"), ("predictions.csv", "predictions_sha256"), ("images.json", "images_sha256")):
        path = directory / name
        if not path.is_file() or sha256_file(path) != receipt.get(field):
            problems.append(f"hash:{name}")
    if problems:
        raise ValueError(
            f"Existing condition {condition!r} in {scope_dir} fails verification ({', '.join(problems)}); "
            "refusing to overwrite. Move it aside to rerun."
        )
    return receipt


def run_condition(
    *,
    scope_dir: Path,
    condition: str,
    samples: Sequence[SampleRef],
    folds: Sequence[Any],
    frozen: Sequence[FrozenFoldModels],
    extractor: FeatureExtractor,
    run_bindings: dict[str, Any],
    gate_sha256: str | None,
    batch_size: int = 32,
    save_images: bool = True,
) -> tuple[dict[str, Any], bool]:
    existing = verify_condition(scope_dir, condition, run_bindings)
    if existing is not None:
        if condition != "original" and existing.get("original_gate_sha256") != gate_sha256:
            raise GateError(f"Condition {condition!r} was produced under a different original gate")
        return existing, True
    if condition != "original" and gate_sha256 is None:
        raise GateError(f"Condition {condition!r} requires a PASS original-reproduction gate first")
    started = time.perf_counter()
    directory = condition_directory(scope_dir, condition)
    visual, dsp, image_records = extract_condition_features(
        condition=condition,
        samples=samples,
        extractor=extractor,
        batch_size=batch_size,
        save_dir=(directory / "images") if save_images else None,
    )
    rows = score_samples(condition=condition, samples=samples, visual=visual, dsp=dsp, folds=folds, frozen=frozen)
    atomic_write_bytes(directory / "features.npz", _npz_bytes(visual, dsp))
    atomic_write_text(directory / "predictions.csv", render_rows_csv(rows))
    atomic_write_json(directory / "images.json", image_records)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED",
        "condition": condition,
        "operations": list(CANONICAL_CONDITIONS[condition]),
        "samples": len(samples),
        "prediction_rows": len(rows),
        "run_bindings": run_bindings,
        "original_gate_sha256": gate_sha256 if condition != "original" else None,
        "visual_feature_commitment": feature_array_commitment(visual),
        "dsp_feature_commitment": feature_array_commitment(dsp),
        "features_sha256": sha256_file(directory / "features.npz"),
        "predictions_sha256": sha256_file(directory / "predictions.csv"),
        "images_sha256": sha256_file(directory / "images.json"),
        "fits": 0,
        "wall_seconds": time.perf_counter() - started,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(directory / "condition_receipt.json", receipt)
    return receipt, False


def read_condition_rows(scope_dir: Path, condition: str) -> list[dict[str, Any]]:
    rows = []
    for row in _read_csv(condition_directory(scope_dir, condition) / "predictions.csv"):
        rows.append(
            {
                **row,
                "outer_fold": int(row["outer_fold"]),
                "label_id": int(row["label_id"]),
                "prediction": int(row["prediction"]),
                "probability_ai_edited": float(row["probability_ai_edited"]),
                "logit_ai_edited": float(row["logit_ai_edited"]),
            }
        )
    return rows


def evaluate_original_gate(
    *,
    scope: str,
    scope_dir: Path,
    protocol: dict[str, Any],
    reference: dict[tuple[str, str], dict[str, Any]],
    run_bindings: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Compare the recomputed original condition with 4C.3B; write original_gate.json."""
    gate_config = protocol["original_reproduction_gate"]["recomputed_original_check"]
    rows = read_condition_rows(scope_dir, "original")
    comparison = compare_to_reference(rows, reference, float(gate_config["max_abs_probability_difference"]))
    gate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "run_bindings": run_bindings,
        "original_predictions_sha256": sha256_file(condition_directory(scope_dir, "original") / "predictions.csv"),
        "prediction_comparison": comparison,
    }
    status = comparison["status"]
    if scope == "full":
        reference_rows = [
            {**row, "label_id": int(row["label_id"]), "probability_ai_edited": float(row["probability_ai_edited"])}
            for row in reference.values()
        ]
        ours, theirs = metrics_for_rows(rows), metrics_for_rows(reference_rows)
        tolerance = float(gate_config["full_metric_tolerance"])
        worst = max(abs(ours[r][k] - theirs[r][k]) for r in RECIPE_IDS for k in REFERENCE_METRIC_KEYS if k not in ("tn", "fp", "fn", "tp"))
        gate["metric_comparison"] = {"max_abs_difference": worst, "tolerance": tolerance}
        if worst > tolerance:
            status = "FAIL"
    gate["status"] = status
    path = scope_dir / "original_gate.json"
    atomic_write_json(path, gate)
    return gate, (sha256_file(path) if status == "PASS" else None)


def run_scope(
    *,
    scope: str,
    output_dir: Path,
    samples: Sequence[SampleRef],
    folds: Sequence[Any],
    frozen: Sequence[FrozenFoldModels],
    reference: dict[tuple[str, str], dict[str, Any]],
    extractor: FeatureExtractor,
    protocol: dict[str, Any],
    run_bindings: dict[str, Any],
    conditions: Sequence[str] = CONDITION_IDS,
    save_images: bool = True,
    batch_size: int = 32,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope!r}")
    if not conditions or conditions[0] != "original":
        raise GateError("The original condition must run first")
    scope_dir = output_dir / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = scope_dir / "scope_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_bindings") != run_bindings:
            raise ValueError(f"{manifest_path} has different bindings; refusing to mix runs")
    else:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "scope": scope,
                "samples": len(samples),
                "sources": len({s.source_id for s in samples}),
                "run_bindings": run_bindings,
                "environment": environment_record(extractor),
                "created_at_utc": utc_now(),
            },
        )

    receipts: dict[str, Any] = {}
    receipt, reused = run_condition(
        scope_dir=scope_dir, condition="original", samples=samples, folds=folds, frozen=frozen,
        extractor=extractor, run_bindings=run_bindings, gate_sha256=None, save_images=save_images,
        batch_size=batch_size,
    )
    receipts["original"] = receipt
    gate, gate_sha = evaluate_original_gate(
        scope=scope, scope_dir=scope_dir, protocol=protocol, reference=reference, run_bindings=run_bindings
    )
    log(f"[{utc_now()}] {scope}: original {'RESUMED' if reused else 'COMPLETED'}; gate {gate['status']}")
    if gate_sha is None:
        raise GateError(
            f"Original-reproduction gate FAILED for scope {scope}; transformed conditions are blocked. "
            f"See {scope_dir / 'original_gate.json'}"
        )
    for condition in conditions[1:]:
        receipt, reused = run_condition(
            scope_dir=scope_dir, condition=condition, samples=samples, folds=folds, frozen=frozen,
            extractor=extractor, run_bindings=run_bindings, gate_sha256=gate_sha, save_images=save_images,
            batch_size=batch_size,
        )
        receipts[condition] = receipt
        log(f"[{utc_now()}] {scope}: {condition} {'RESUMED' if reused else 'COMPLETED'}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "conditions": list(receipts),
        "samples": len(samples),
        "fits": 0,
        "original_gate": gate,
        "original_gate_sha256": gate_sha,
        "receipts": receipts,
    }
    atomic_write_json(scope_dir / "scope_summary.json", summary)
    return summary
