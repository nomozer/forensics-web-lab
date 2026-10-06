"""Candidate Model Bindings Loader and Deterministic Scorers for Phase 4C.7A.

Strictly enforces:
1. Loading the 5 outer-fold models from Phase 4C.6B/4C.3B with exact SHA-256 verification.
2. Refusing execution if any model file is missing or tampered with.
3. Providing pure algebraic scoring on feature vectors without refitting or recalibration.
4. Preserving the exact feature dimensions: Visual (576-d) and DSP (16-d).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINDINGS_PATH = REPO_ROOT / "research/evidence/phase-4c.7a/candidate_model_bindings.json"


class ModelIntegrityError(ValueError):
    """Raised when a candidate model file is missing or its SHA-256 hash does not match bindings."""


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


@dataclass(frozen=True)
class LinearScorer:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    temperature: float

    def raw_logit(self, x: np.ndarray) -> np.ndarray:
        x_scaled = (x - self.scaler_mean) / self.scaler_scale
        return np.dot(x_scaled, self.coef) + self.intercept

    def calibrated_logit(self, x: np.ndarray) -> np.ndarray:
        return self.raw_logit(x) / self.temperature

    def probability(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.calibrated_logit(x))


@dataclass(frozen=True)
class StackerScorer:
    coef: np.ndarray  # shape (2,)
    intercept: float

    def fusion_logit(self, visual_cal_logit: np.ndarray, dsp_cal_logit: np.ndarray) -> np.ndarray:
        return visual_cal_logit * self.coef[0] + dsp_cal_logit * self.coef[1] + self.intercept

    def probability(self, visual_cal_logit: np.ndarray, dsp_cal_logit: np.ndarray) -> np.ndarray:
        return _sigmoid(self.fusion_logit(visual_cal_logit, dsp_cal_logit))


@dataclass(frozen=True)
class FoldCandidateModel:
    outer_fold: int
    visual_scorer: LinearScorer
    dsp_augmented_scorer: LinearScorer
    stacker: StackerScorer

    def score(
        self,
        visual_features: np.ndarray,
        dsp_features: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Score a batch of samples for both recipes. Pure function."""
        # Visual Calibrated Recipe
        z_v_raw = self.visual_scorer.raw_logit(visual_features)
        z_v_cal = z_v_raw / self.visual_scorer.temperature
        p_v = _sigmoid(z_v_cal)

        # Augmented DSP Branch
        z_d_raw = self.dsp_augmented_scorer.raw_logit(dsp_features)
        z_d_cal = z_d_raw / self.dsp_augmented_scorer.temperature

        # Late Fusion Augmented Recipe
        z_fusion = self.stacker.fusion_logit(z_v_cal, z_d_cal)
        p_fusion = _sigmoid(z_fusion)

        return {
            "visual_calibrated": {
                "raw_logit": z_v_raw,
                "calibrated_logit": z_v_cal,
                "probability": p_v,
                "prediction": (p_v >= 0.5).astype(int),
            },
            "late_fusion_dsp_augmented": {
                "raw_logit": z_fusion,
                "calibrated_logit": z_fusion,  # fusion logits are already calibrated
                "probability": p_fusion,
                "prediction": (p_fusion >= 0.5).astype(int),
            },
        }


def load_candidate_models(
    bindings_path: Path | str | None = None,
    repo_root: Path | None = None,
) -> list[FoldCandidateModel]:
    """Load and verify all 5 outer-fold candidate models from bindings."""
    root = repo_root or REPO_ROOT
    b_path = Path(bindings_path) if bindings_path is not None else DEFAULT_BINDINGS_PATH
    if not b_path.is_file():
        raise FileNotFoundError(f"Bindings file not found at {b_path}")

    bindings = json.loads(b_path.read_text(encoding="utf-8"))

    # Verify DSP feature contract against canonical extractor if specified
    from ml.training.dsp_features import DSP_FEATURE_NAMES

    feature_contracts = bindings.get("feature_contracts")
    if feature_contracts is not None:
        dsp_contract = feature_contracts.get("dsp")
        if dsp_contract is not None and "feature_order" in dsp_contract:
            bound_order = tuple(dsp_contract.get("feature_order", []))
            if bound_order != DSP_FEATURE_NAMES:
                raise ModelIntegrityError(
                    f"DSP feature order mismatch in bindings! Expected {DSP_FEATURE_NAMES}, got {bound_order}"
                )

    folds_spec = bindings.get("outer_folds", [])
    if len(folds_spec) != 5:
        raise ModelIntegrityError(f"Expected exactly 5 outer folds in bindings, found {len(folds_spec)}")

    models: list[FoldCandidateModel] = []
    for spec in folds_spec:
        fold_idx = int(spec["outer_fold"])
        model_path = root / spec["model_path"]
        expected_sha = spec["model_sha256"]

        if not model_path.is_file():
            raise ModelIntegrityError(f"Fold {fold_idx} model file missing: {model_path}")

        actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_sha.lower() != expected_sha.lower():
            raise ModelIntegrityError(
                f"Fold {fold_idx} model SHA-256 mismatch! Expected {expected_sha}, got {actual_sha}"
            )

        data = json.loads(model_path.read_text(encoding="utf-8"))

        # Visual Scorer (from baseline.base.visual and augmented.temperatures.visual)
        base_v = data["baseline"]["base"]["visual"]
        t_v = float(data["augmented"]["temperatures"]["visual"])
        visual_scorer = LinearScorer(
            scaler_mean=np.array(base_v["scaler_mean"], dtype=np.float64),
            scaler_scale=np.array(base_v["scaler_scale"], dtype=np.float64),
            coef=np.array(base_v["coef"], dtype=np.float64),
            intercept=float(base_v["intercept"]),
            temperature=t_v,
        )

        # Augmented DSP Scorer (from augmented.dsp and augmented.temperatures.dsp)
        aug_d = data["augmented"]["dsp"]
        t_d = float(data["augmented"]["temperatures"]["dsp"])
        dsp_scorer = LinearScorer(
            scaler_mean=np.array(aug_d["scaler_mean"], dtype=np.float64),
            scaler_scale=np.array(aug_d["scaler_scale"], dtype=np.float64),
            coef=np.array(aug_d["coef"], dtype=np.float64),
            intercept=float(aug_d["intercept"]),
            temperature=t_d,
        )

        # Stacker (from augmented.stacker)
        stk = data["augmented"]["stacker"]
        stacker = StackerScorer(
            coef=np.array(stk["coef"], dtype=np.float64),
            intercept=float(stk["intercept"]),
        )

        models.append(
            FoldCandidateModel(
                outer_fold=fold_idx,
                visual_scorer=visual_scorer,
                dsp_augmented_scorer=dsp_scorer,
                stacker=stacker,
            )
        )

    return sorted(models, key=lambda m: m.outer_fold)


def create_mock_candidate_models(num_models: int = 5, seed: int = 42) -> list[FoldCandidateModel]:
    """Create in-memory deterministic candidate models for hermetic unit testing and CI gates."""
    rng = np.random.Generator(np.random.PCG64(seed))
    models: list[FoldCandidateModel] = []
    for k in range(num_models):
        vis = LinearScorer(
            scaler_mean=np.zeros(576, dtype=np.float64),
            scaler_scale=np.ones(576, dtype=np.float64),
            coef=rng.normal(0.0, 0.1, size=576).astype(np.float64),
            intercept=float(rng.normal(0.0, 0.1)),
            temperature=1.0,
        )
        dsp = LinearScorer(
            scaler_mean=np.zeros(16, dtype=np.float64),
            scaler_scale=np.ones(16, dtype=np.float64),
            coef=rng.normal(0.0, 0.1, size=16).astype(np.float64),
            intercept=float(rng.normal(0.0, 0.1)),
            temperature=1.0,
        )
        stk = StackerScorer(
            coef=np.array([1.0, 0.5], dtype=np.float64),
            intercept=0.0,
        )
        models.append(
            FoldCandidateModel(
                outer_fold=k,
                visual_scorer=vis,
                dsp_augmented_scorer=dsp,
                stacker=stk,
            )
        )
    return models
