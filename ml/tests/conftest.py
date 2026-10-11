from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat

import pytest

pytest_plugins = ["ml.tests.fixtures.cohort_catalog_fixture"]


_TEMP_MARKER = ".forensics_pytest_temp"
_MANAGED_PREFIXES = ("pytest-", "pilot-dossier-pytest-")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_ARTIFACTS_ROOT = _REPO_ROOT / "data" / "research" / "local-artifacts"


def _is_managed_workspace_basetemp(path: Path, artifacts_root: Path) -> bool:
    candidate = path.resolve(strict=False)
    root = artifacts_root.resolve(strict=False)
    return (
        candidate.parent == root
        and candidate.name.startswith(_MANAGED_PREFIXES)
        and not candidate.is_symlink()
    )


def _remove_readonly(func, path: str, _exc) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    func(path)


def _cleanup_workspace_basetemp(path: Path, artifacts_root: Path = _LOCAL_ARTIFACTS_ROOT) -> bool:
    """Delete one explicitly marked pytest root without widening cleanup scope."""
    candidate = path.resolve(strict=False)
    if not _is_managed_workspace_basetemp(candidate, artifacts_root):
        return False
    marker = candidate / _TEMP_MARKER
    if not marker.is_file() or marker.read_text(encoding="utf-8") != "test-only\n":
        return False
    shutil.rmtree(candidate, onexc=_remove_readonly)
    return True


@pytest.fixture(scope="session", autouse=True)
def cleanup_managed_workspace_basetemp(tmp_path_factory: pytest.TempPathFactory):
    """Clean workspace --basetemp output even when a test fails normally."""
    base_temp = tmp_path_factory.getbasetemp().resolve(strict=False)
    managed = _is_managed_workspace_basetemp(base_temp, _LOCAL_ARTIFACTS_ROOT)
    if managed:
        (base_temp / _TEMP_MARKER).write_text("test-only\n", encoding="utf-8")
    try:
        yield
    finally:
        if managed and base_temp.exists() and not _cleanup_workspace_basetemp(base_temp):
            raise RuntimeError(f"Refused to clean unmarked managed pytest directory: {base_temp}")
