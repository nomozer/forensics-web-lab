"""
Pilot Experiment Configuration Validator and Label-Semantics Guard.

Validates machine-readable pilot configurations against scientific standards:
- Strictly enforces 3-class label taxonomy (authentic, fully_generated, ai_edited).
- Prohibits TGIF conditional regeneration (fr) components from being mapped to fully_generated.
- Requires group_key to be source_id to prevent parent-child data leakage.
- Ensures required metadata, metrics, and shortcut checks are declared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

VALID_CLASSIFICATION_LABELS = {"authentic", "fully_generated", "ai_edited"}
VALID_COMPONENT_LABELS = VALID_CLASSIFICATION_LABELS | {"ground_truth_mask"}
VALID_TASKS = {"classification", "classification_and_localization", "localization"}
VALID_LICENSE_TRACKS = {"research-only", "product-eligible", "fixture-only"}
VALID_SCIENTIFIC_STATUSES = {
    "exploratory_pilot",
    "confirmatory_benchmark",
    "pipeline_smoke",
    "preregistered_protocol",
}

MANDATORY_BASELINES = {
    "stratified_dummy",
    "metadata_only",
    "dsp_only",
    "frozen_visual_linear",
    "finetuned_visual",
    "multimodal_fusion",
}


def validate_learning_curve_config_dict(config: Dict[str, Any], config_source: str = "config") -> List[str]:
    """Validates small-data learning curve protocol configuration against scientific standards."""
    errors: List[str] = []

    # 1. Schema and metadata
    if config.get("schema_version") != "1.0.0":
        errors.append(f"[{config_source}] Invalid or missing schema_version: '{config.get('schema_version')}'")

    if not config.get("config_id"):
        errors.append(f"[{config_source}] Missing config_id.")

    status = config.get("scientific_status")
    if status not in VALID_SCIENTIFIC_STATUSES:
        errors.append(f"[{config_source}] Invalid scientific_status: '{status}'. Expected one of: {sorted(VALID_SCIENTIFIC_STATUSES)}")

    track = config.get("license_track")
    if track not in VALID_LICENSE_TRACKS:
        errors.append(f"[{config_source}] Invalid license_track: '{track}'. Expected one of: {sorted(VALID_LICENSE_TRACKS)}")

    # 2. Independent statistical unit declaration
    unit = config.get("independent_statistical_unit")
    if unit != "source_id":
        errors.append(
            f"[{config_source}] INDEPENDENT-UNIT VIOLATION: independent_statistical_unit must be 'source_id', got '{unit}'."
        )

    if not config.get("enforce_source_isolation", False):
        errors.append(f"[{config_source}] enforce_source_isolation must be True.")

    if not config.get("forbid_masks_as_independent_samples", False):
        errors.append(f"[{config_source}] forbid_masks_as_independent_samples must be True.")

    if not config.get("no_fr_to_fully_generated", False):
        errors.append(f"[{config_source}] no_fr_to_fully_generated must be True.")

    # 3. Data scope
    data_scope = config.get("data_scope", {})
    if not isinstance(data_scope, dict):
        errors.append(f"[{config_source}] Missing data_scope section.")
    else:
        labels = data_scope.get("labels", [])
        if "fully_generated" in labels and data_scope.get("target_task") == "authentic_vs_ai_edited":
            errors.append(f"[{config_source}] UNSUPPORTED THREE-CLASS ACTIVATION: Pilot A only accepts authentic and ai_edited.")

        train_pool = data_scope.get("train_pool_sources", 0)

        # 4. Learning curve sample sizes and over-capacity guard
        lc = config.get("learning_curve", {})
        sample_sizes = lc.get("sample_sizes", [])
        if not isinstance(sample_sizes, list) or len(sample_sizes) == 0:
            errors.append(f"[{config_source}] learning_curve.sample_sizes must be a non-empty list.")
        else:
            for s in sample_sizes:
                n = s.get("n_sources")
                s_status = s.get("status")
                if not isinstance(n, int) or n <= 0:
                    errors.append(f"[{config_source}] Invalid n_sources: {n}")
                if s_status not in ("runnable", "not_runnable"):
                    errors.append(f"[{config_source}] Invalid status for level N={n}: '{s_status}'")
                if n > train_pool and s_status == "runnable":
                    errors.append(
                        f"[{config_source}] CAPACITY OVERFLOW: Sample size N={n} exceeds training pool "
                        f"({train_pool} sources) but status is 'runnable'. Must be 'not_runnable'."
                    )

    # 5. Experimental protocol (seeds, fixed test evaluation)
    proto = config.get("experimental_protocol", {})
    if not isinstance(proto, dict):
        errors.append(f"[{config_source}] Missing experimental_protocol section.")
    else:
        exp_seeds = proto.get("exploratory_seeds", [])
        if not isinstance(exp_seeds, list) or len(exp_seeds) < 3:
            errors.append(f"[{config_source}] exploratory_seeds must contain at least 3 seeds.")

        fin_seeds = proto.get("final_evaluation_seeds", [])
        if not isinstance(fin_seeds, list) or len(fin_seeds) < 5:
            errors.append(f"[{config_source}] final_evaluation_seeds must contain at least 5 seeds.")

        if not proto.get("fixed_test_evaluation", False):
            errors.append(f"[{config_source}] fixed_test_evaluation must be True.")

    # 6. Mandatory baselines
    baselines = config.get("baselines", [])
    if not isinstance(baselines, list):
        errors.append(f"[{config_source}] baselines must be a list.")
    else:
        b_ids = {b.get("id") for b in baselines if isinstance(b, dict)}
        missing_baselines = MANDATORY_BASELINES - b_ids
        if missing_baselines:
            errors.append(
                f"[{config_source}] Missing mandatory baselines: {sorted(missing_baselines)}"
            )

    return errors


def validate_pilot_config_dict(config: Dict[str, Any], config_source: str = "config") -> List[str]:
    """Validates an in-memory pilot configuration dictionary against scientific gates."""
    # Check if this is a learning curve protocol config
    if config.get("config_id") == "pilot_a_learning_curve" or "learning_curve" in config:
        return validate_learning_curve_config_dict(config, config_source=config_source)

    errors: List[str] = []

    # 1. Schema version
    if config.get("schema_version") != "1.0.0":
        errors.append(f"[{config_source}] Invalid or missing schema_version: '{config.get('schema_version')}'")

    # 2. Pilot ID and metadata
    if not config.get("pilot_id"):
        errors.append(f"[{config_source}] Missing pilot_id.")

    # 3. Scientific status and track
    status = config.get("scientific_status")
    if status not in VALID_SCIENTIFIC_STATUSES:
        errors.append(f"[{config_source}] Invalid scientific_status: '{status}'. Expected one of: {sorted(VALID_SCIENTIFIC_STATUSES)}")

    track = config.get("license_track")
    if track not in VALID_LICENSE_TRACKS:
        errors.append(f"[{config_source}] Invalid license_track: '{track}'. Expected one of: {sorted(VALID_LICENSE_TRACKS)}")

    # 4. Task
    task = config.get("task")
    if task not in VALID_TASKS:
        errors.append(f"[{config_source}] Invalid task: '{task}'. Expected one of: {sorted(VALID_TASKS)}")

    # 5. Accepted labels taxonomy check
    accepted_labels = config.get("accepted_labels", [])
    if not isinstance(accepted_labels, list) or len(accepted_labels) < 2:
        errors.append(f"[{config_source}] accepted_labels must be a list containing at least 2 labels.")
    else:
        for lbl in accepted_labels:
            if lbl not in VALID_CLASSIFICATION_LABELS:
                errors.append(
                    f"[{config_source}] Invalid classification label '{lbl}'. "
                    f"Must strictly be one of: {sorted(VALID_CLASSIFICATION_LABELS)}"
                )

    # 6. Dataset components and TGIF fr semantics check
    components = config.get("dataset_components", [])
    if not isinstance(components, list) or len(components) == 0:
        errors.append(f"[{config_source}] dataset_components must be a non-empty list.")
    else:
        for i, comp in enumerate(components):
            c_name = comp.get("component_id", f"comp_{i}")
            c_label = comp.get("assigned_label")
            remote_folder = comp.get("remote_folder", "").lower()

            if not c_label:
                errors.append(f"[{config_source}:{c_name}] Missing assigned_label.")
            elif c_label not in VALID_COMPONENT_LABELS:
                errors.append(
                    f"[{config_source}:{c_name}] Assigned label '{c_label}' is not valid. "
                    f"Must be one of: {sorted(VALID_COMPONENT_LABELS)}"
                )

            # CRITICAL RULE: TGIF fr components (sd2-fr, sdxl-fr, flux*-fr) CANNOT be fully_generated!
            if any(fr_kw in remote_folder for fr_kw in ["-fr", "fr/"]):
                if c_label == "fully_generated":
                    errors.append(
                        f"[{config_source}:{c_name}] SEMANTICS VIOLATION: TGIF component '{remote_folder}' "
                        f"is a fully-regenerated canvas conditioned on authentic MS-COCO images. "
                        f"It is strictly PROHIBITED from being assigned to 'fully_generated'. "
                        f"It must be classified as 'ai_edited' or quarantined."
                    )

            # SP components must be ai_edited
            if any(sp_kw in remote_folder for sp_kw in ["-sp", "sp/"]):
                if c_label not in ("ai_edited", "ground_truth_mask"):
                    errors.append(
                        f"[{config_source}:{c_name}] SEMANTICS VIOLATION: Spliced component '{remote_folder}' "
                        f"must be assigned to 'ai_edited', not '{c_label}'."
                    )

    # 7. Split configuration and group_key leakage guard
    split_cfg = config.get("split", {})
    if not isinstance(split_cfg, dict):
        errors.append(f"[{config_source}] Missing or invalid 'split' section.")
    else:
        group_key = split_cfg.get("group_key")
        if group_key != "source_id":
            errors.append(
                f"[{config_source}] ANTI-LEAKAGE VIOLATION: split.group_key must be 'source_id' "
                f"to prevent parent-child image leakage across splits. Found: '{group_key}'."
            )
        train_ratio = split_cfg.get("train_ratio", 0)
        val_ratio = split_cfg.get("val_ratio", 0)
        test_ratio = split_cfg.get("test_ratio", 0)
        if round(train_ratio + val_ratio + test_ratio, 2) != 1.0:
            errors.append(
                f"[{config_source}] Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio})."
            )

    # 8. Preprocessing
    preproc = config.get("preprocessing", {})
    if not isinstance(preproc, dict) or not preproc.get("target_size"):
        errors.append(f"[{config_source}] Missing or invalid preprocessing.target_size.")

    # 9. Evaluation metrics
    eval_cfg = config.get("evaluation", {})
    if not isinstance(eval_cfg, dict):
        errors.append(f"[{config_source}] Missing evaluation section.")
    else:
        primary = eval_cfg.get("primary_classification_metrics", [])
        if not isinstance(primary, list) or len(primary) == 0:
            errors.append(f"[{config_source}] Missing primary_classification_metrics in evaluation section.")

    # 10. Shortcut checks
    shortcuts = config.get("shortcut_checks", [])
    if not isinstance(shortcuts, list) or len(shortcuts) == 0:
        errors.append(f"[{config_source}] shortcut_checks must be a non-empty list of verification protocols.")

    return errors


def validate_pilot_config_file(config_path: Path) -> List[str]:
    """Loads and validates a YAML pilot config file."""
    if not config_path.exists():
        return [f"Config file not found: {config_path}"]

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return [f"Failed to parse YAML from {config_path}: {e}"]

    if not isinstance(data, dict):
        return [f"Config root must be a mapping: {config_path}"]

    return validate_pilot_config_dict(data, config_source=config_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot Experiment Config Validator & Label Semantics Gate")
    parser.add_argument("--config", type=str, help="Path to pilot config YAML file")
    parser.add_argument("--validate-all", action="store_true", help="Validate all configs in ml/configs/pilot_*.yaml")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    if args.validate_all:
        config_dir = repo_root / "ml" / "configs"
        pilot_files = sorted(config_dir.glob("pilot_*.yaml"))
        if not pilot_files:
            print(f"[ERROR] No pilot_*.yaml files found in {config_dir}", file=sys.stderr)
            sys.exit(1)

        total_errors = 0
        print(f"Validating {len(pilot_files)} pilot configurations...")
        for p in pilot_files:
            errs = validate_pilot_config_file(p)
            if errs:
                print(f"[FAIL] {p.name}:", file=sys.stderr)
                for e in errs:
                    print(f"  - {e}", file=sys.stderr)
                total_errors += len(errs)
            else:
                print(f"[PASS] {p.name} is valid and meets label-semantics gate.")

        if total_errors > 0:
            print(f"\nTotal validation errors: {total_errors}", file=sys.stderr)
            sys.exit(1)
        print("\nAll pilot configurations validated successfully.")
        sys.exit(0)

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = repo_root / cfg_path
        errs = validate_pilot_config_file(cfg_path)
        if errs:
            print(f"[FAIL] Config validation failed for {cfg_path.name}:", file=sys.stderr)
            for e in errs:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[PASS] {cfg_path.name} is valid and meets label-semantics gate.")
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
