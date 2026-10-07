"""Regression gates for functional repository path names."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING = REPO_ROOT / "docs" / "REPOSITORY_NAMING_MAP.md"
SEALED_PHASE_PATHS = {
    "ml/configs/phase_4c1_learning_curve.yaml",
    "ml/configs/phase_4c2_stage2_finetuning.yaml",
    "ml/configs/phase_4c2h_development_nested_cv.yaml",
    "ml/evaluation/phase_4c2g_dataset.py",
    "ml/evaluation/phase_4c2g_io.py",
    "ml/evaluation/phase_4c2g_model.py",
    "ml/evaluation/phase_4c2g_windows.py",
    "ml/evaluation/run_phase_4c2f_evaluator.py",
    "ml/evaluation/run_phase_4c2g_confirmatory.py",
    "notebooks/phase_4c1_learning_curve_colab.ipynb",
    "notebooks/phase_4c2_finetuning_colab.ipynb",
    "notebooks/phase_4c2h_development_colab.ipynb",
    "scripts/phase_4c2_execute_all.sh",
    "scripts/research/build_phase4c2g_execution_package.py",
    "scripts/research/build_phase4c2g_manifest_custodian_package.py",
    "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1",
    "scripts/research/RUN_PHASE4C2G_AUTOMATED_ISOLATION.ps1",
    "scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1",
    "scripts/research/RUN_PHASE4C2G_OFFLINE_VERIFIER.ps1",
    "scripts/research/verify_phase_4c2g_offline_runtime.py",
}


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line]


def mapping_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    pattern = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \|")
    for line in MAPPING.read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line):
            rows.append((match.group(1), match.group(2)))
    return rows


def test_naming_map_covers_all_renames() -> None:
    rows = mapping_rows()
    assert len(rows) == 47
    for previous, current in rows:
        assert not (REPO_ROOT / previous).exists(), previous
        assert (REPO_ROOT / current).is_file(), current


def test_only_sealed_tracked_paths_keep_phase_led_names() -> None:
    phase_pattern = re.compile(r"phase[_-]?[0-9]", re.IGNORECASE)
    remaining = {
        path
        for path in tracked_paths()
        if phase_pattern.search(path) and not path.startswith("research/evidence/")
    }
    assert remaining == SEALED_PHASE_PATHS


def test_current_continuity_uses_functional_paths() -> None:
    current_docs = (
        REPO_ROOT / "docs" / "continuity" / "CURRENT_STATE.md",
        REPO_ROOT / "docs" / "continuity" / "CODE_INDEX.md",
        REPO_ROOT / "docs" / "RESEARCH_WORKSTREAM_INDEX.md",
    )
    for previous, _current in mapping_rows():
        for document in current_docs:
            assert previous not in document.read_text(encoding="utf-8"), (
                f"stale current path {previous} in {document.relative_to(REPO_ROOT)}"
            )
