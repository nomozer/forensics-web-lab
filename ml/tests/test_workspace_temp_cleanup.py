from __future__ import annotations

from pathlib import Path

from ml.tests.conftest import _cleanup_workspace_basetemp


def test_cleanup_workspace_basetemp_requires_marker_and_locked_location(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "local-artifacts"
    candidate = artifacts_root / "pytest-cleanup-regression"
    candidate.mkdir(parents=True)
    payload = candidate / "nested" / "fixture.txt"
    payload.parent.mkdir()
    payload.write_text("fixture", encoding="utf-8")

    assert _cleanup_workspace_basetemp(candidate, artifacts_root) is False
    assert payload.is_file()

    (candidate / ".forensics_pytest_temp").write_text("test-only\n", encoding="utf-8")
    assert _cleanup_workspace_basetemp(candidate, artifacts_root) is True
    assert not candidate.exists()


def test_cleanup_workspace_basetemp_refuses_unlocked_name(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "local-artifacts"
    candidate = artifacts_root / "pilot-20261007T132003Z"
    candidate.mkdir(parents=True)
    marker = candidate / ".forensics_pytest_temp"
    marker.write_text("test-only\n", encoding="utf-8")

    assert _cleanup_workspace_basetemp(candidate, artifacts_root) is False
    assert marker.is_file()
