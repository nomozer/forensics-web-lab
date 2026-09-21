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
VALID_SCIENTIFIC_STATUSES = {"exploratory_pilot", "confirmatory_benchmark", "pipeline_smoke"}


def validate_pilot_config_dict(config: Dict[str, Any], config_source: str = "config") -> List[str]:
    """Validates an in-memory pilot configuration dictionary against scientific gates."""
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
