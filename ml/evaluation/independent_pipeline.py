"""End-to-End Evaluation Pipeline for Independent Validation (Phase 4C.7A).

Verifies the complete execution chain:
image decode -> transformations (6 conditions) -> feature extraction (Visual 576-d, DSP 16-d)
-> frozen scaler / classifiers -> temperatures / stacker -> per-model predictions
-> arithmetic mean-per-model Macro-F1 -> paired source-cluster bootstrap (10,000 replicates).

Strict constraints:
- Individual Image Isolation: each image is scored purely from its own transformed pixels.
  Zero access to paired counterpart, source_id, file paths, or ground-truth labels during extraction/scoring.
- Exact Transform Lock: reuses canonical Pillow RGB transforms, bicubic resize, and JPEG encode settings.
- Model Freeze: exactly 5 outer-fold models per recipe, 0 refit, 0 recalibration, 0 best-fold selection.
- Arithmetic Estimand: mean of 5 fold Macro-F1 scores, NO probability averaging before metric calculation.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from ml.evaluation.independent_cohort import CohortPair
from ml.evaluation.independent_evaluator import (
    CONDITIONS,
    PRIMARY_CONDITION,
    RECIPES,
    aggregate_per_model_metrics,
    derive_independent_verdict,
    run_paired_source_cluster_bootstrap,
)
from ml.evaluation.independent_model_bindings import FoldCandidateModel, load_candidate_models
from ml.training.development_robustness import (
    CANONICAL_CONDITIONS,
    apply_operations,
)
from ml.training.dsp_features import DSP_FEATURE_DIM, extract_dsp_features


class IndependentEvaluationPipeline:
    """Manages end-to-end evaluation of image cohorts across 6 conditions and 2 recipes."""

    def __init__(
        self,
        candidate_models: list[FoldCandidateModel] | None = None,
        visual_extractor: Callable[[Image.Image], np.ndarray] | None = None,
    ):
        if candidate_models is not None:
            self.models = candidate_models
        else:
            self.models = load_candidate_models()
        assert len(self.models) == 5, f"Expected 5 candidate models, got {len(self.models)}"

        self.visual_extractor = visual_extractor

    def _extract_visual_features(self, image: Image.Image) -> np.ndarray:
        """Extract 576-dim visual feature vector from PIL Image."""
        if self.visual_extractor is not None:
            feat = self.visual_extractor(image)
            assert feat.shape == (576,), f"Expected visual feature shape (576,), got {feat.shape}"
            return feat.astype(np.float64)

        # Fallback to deterministic pseudo-extractor for tests if no neural extractor injected
        # Generates deterministic 576-d array based on image hash to ensure reproducibility in tests
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        rng = np.random.Generator(np.random.PCG64(int(mean_val * 1000 + std_val * 100)))
        return rng.normal(0.0, 1.0, size=576).astype(np.float64)

    def _extract_dsp_features(self, image: Image.Image) -> np.ndarray:
        """Extract 16-dim forensic DSP feature vector from PIL Image using canonical extractor."""
        dsp_feat = extract_dsp_features(image)
        assert dsp_feat.shape == (DSP_FEATURE_DIM,), f"Expected shape ({DSP_FEATURE_DIM},), got {dsp_feat.shape}"
        return dsp_feat.astype(np.float64)

    def evaluate_cohort(
        self,
        pairs: Sequence[CohortPair],
        data_root: Path | str | None = None,
        image_resolver: Callable[[str], Image.Image] | None = None,
        bootstrap_replicates: int = 10000,
        seed: int = 20261007,
        is_synthetic: bool = True,
    ) -> dict[str, Any]:
        """Execute end-to-end pipeline across all 6 conditions and 2 candidate recipes.

        Parameters:
            pairs: List of verified CohortPair objects.
            data_root: Root directory to resolve image_relpath (if image_resolver not provided).
            image_resolver: Optional custom loader function (relpath -> PIL.Image).
            bootstrap_replicates: Number of bootstrap replicates (default 10,000).
            seed: PRNG seed for bootstrap.
            is_synthetic: Whether the evaluation is synthetic preflight or real data.
        """
        root = Path(data_root) if data_root is not None else Path(".")

        # Assemble individual images list: strictly isolated, order deterministic
        # Each source contributes 2 samples: authentic (label 0) and ai_edited (label 1)
        samples_meta: list[dict[str, Any]] = []
        for p in pairs:
            samples_meta.append({
                "source_id": p.source_id,
                "label": "authentic",
                "label_id": 0,
                "relpath": p.authentic.image_relpath,
            })
            samples_meta.append({
                "source_id": p.source_id,
                "label": "ai_edited",
                "label_id": 1,
                "relpath": p.ai_edited.image_relpath,
            })

        n_samples = len(samples_meta)
        source_ids = [s["source_id"] for s in samples_meta]
        labels = np.array([s["label_id"] for s in samples_meta], dtype=int)

        # 1. Load base RGB images
        raw_images: list[Image.Image] = []
        for s in samples_meta:
            if image_resolver is not None:
                img = image_resolver(s["relpath"]).convert("RGB")
            else:
                img_path = root / s["relpath"]
                with Image.open(img_path) as handle:
                    img = handle.convert("RGB")
            raw_images.append(img)

        condition_results: dict[str, Any] = {}
        vis_q75_probs: list[np.ndarray] = []
        aug_q75_probs: list[np.ndarray] = []

        # 2. Iterate through all 6 conditions
        for cond in CONDITIONS:
            operations = CANONICAL_CONDITIONS[cond]

            # A. Transform images individually
            transformed_images: list[Image.Image] = []
            for img in raw_images:
                t_img, _ = apply_operations(img, operations)
                transformed_images.append(t_img)

            # B. Extract features individually (zero peer context)
            visual_feats = np.stack([self._extract_visual_features(img) for img in transformed_images])
            dsp_feats = np.stack([self._extract_dsp_features(img) for img in transformed_images])

            # C. Score across 5 candidate models for both recipes
            recipe_metrics: dict[str, Any] = {}
            for recipe in RECIPES:
                fold_predictions: list[dict[str, np.ndarray]] = []
                for m in self.models:
                    scores = m.score(visual_feats, dsp_feats)
                    fold_predictions.append(scores[recipe])

                # D. Arithmetic mean-per-model aggregation
                agg = aggregate_per_model_metrics(fold_predictions, labels)
                recipe_metrics[recipe] = agg

                if cond == PRIMARY_CONDITION:
                    probs_list = [fp["probability"] for fp in fold_predictions]
                    if recipe == "visual_calibrated":
                        vis_q75_probs = probs_list
                    elif recipe == "late_fusion_dsp_augmented":
                        aug_q75_probs = probs_list

            condition_results[cond] = recipe_metrics

        # 3. Paired source-cluster bootstrap on primary condition (jpeg_q75)
        bootstrap_result = run_paired_source_cluster_bootstrap(
            source_ids=source_ids,
            labels=labels,
            visual_model_probs=vis_q75_probs,
            augmented_model_probs=aug_q75_probs,
            replicates=bootstrap_replicates,
            seed=seed,
        )

        # 4. Prespecified verdict derivation
        verdict = derive_independent_verdict(
            ci_lower=bootstrap_result["ci_lower_95"],
            ci_upper=bootstrap_result["ci_upper_95"],
            is_synthetic=is_synthetic,
        )

        return {
            "status": "EVALUATION_SUCCESS",
            "is_synthetic": is_synthetic,
            "verdict": verdict,
            "primary_condition": PRIMARY_CONDITION,
            "conditions_evaluated": list(CONDITIONS),
            "num_pairs": len(pairs),
            "num_samples": n_samples,
            "condition_results": condition_results,
            "bootstrap": bootstrap_result,
        }
