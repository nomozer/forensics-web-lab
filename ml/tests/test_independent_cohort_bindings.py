"""Phase 4C.7B regression tests: notebook commit binding, fail-closed Colab cells,
candidate eligibility/provenance, namespaced disjoint guard, run binding and audit.

All catalogs used here are SYNTHETIC fixtures (ml/tests/fixtures/cohort_catalog_fixture.py);
nothing in this file is evidence about real photographs, real inpainting or a GPU pilot.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

from unittest.mock import MagicMock

from PIL import Image
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.independent_cohort import SourceOverlapError
from ml.evaluation.independent_cohort_acquisition import (
    CandidateEligibilityError,
    CandidateSpec,
    DiffusersInpaintingEngine,
    GenerationContractError,
    INPAINTING_MODEL_REGISTRY,
    MockInpaintingEngine,
    ModelPreflightError,
    RunAuditError,
    RunBindingMismatchError,
    STRATA_KEYS,
    audit_acquisition_run,
    check_gpu_policy,
    download_authentic_image,
    evaluate_candidate_eligibility,
    execute_cohort_acquisition,
    generate_canonical_candidate_plan,
    generate_synthetic_fixture_plan,
    load_historical_source_keys,
    verify_model_access_preflight,
)
from ml.tests.fixtures.cohort_catalog_fixture import (  # noqa: F401 (autouse fixture)
    build_synthetic_catalog,
    clean_detector_modules_isolation,
    synthetic_coco_entry,
    write_synthetic_catalog,
)

NOTEBOOK = REPO_ROOT / "notebooks/independent_cohort_acquisition_colab.ipynb"
LEGACY_CATALOG = REPO_ROOT / "research/evidence/phase-4c.7b/verified_candidate_catalog.json"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def _helpers() -> dict:
    """Exec notebook Cell 1 (pure helper definitions, no side effects)."""
    ns: dict = {}
    exec(_cells()[1], ns)
    return ns


def _binding(run_id: str = "run-A", commit: str = "a" * 40) -> dict:
    return {"run_id": run_id, "git_commit": commit, "protocol_sha256": "p", "catalog_sha256": "c",
            "plan_sha256": "q", "mode": "pilot", "target_per_stratum": 2}


# ---------------------------------------------------------------- notebook binding


@pytest.fixture()
def origin_with_two_commits(tmp_path: Path) -> tuple[Path, str, str]:
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    (origin / "f.txt").write_text("old\n")
    _git(origin, "add", ".")
    _git(origin, "commit", "-q", "-m", "old")
    old = _git(origin, "rev-parse", "HEAD")
    (origin / "f.txt").write_text("new\n")
    _git(origin, "commit", "-q", "-am", "new")
    new = _git(origin, "rev-parse", "HEAD")
    return origin, old, new


def test_existing_clone_at_old_commit_is_checked_out_at_expected_sha(origin_with_two_commits, tmp_path):
    origin, old, new = origin_with_two_commits
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", "--detach", old)

    _helpers()["checkout_pinned"](clone, str(origin), new)

    assert _git(clone, "rev-parse", "HEAD") == new
    assert (clone / "f.txt").read_text() == "new\n"


def test_fresh_clone_is_checked_out_at_expected_sha_not_branch_head(origin_with_two_commits, tmp_path):
    origin, old, _new = origin_with_two_commits
    clone = tmp_path / "fresh"
    _helpers()["checkout_pinned"](clone, str(origin), old)
    assert _git(clone, "rev-parse", "HEAD") == old


def test_dirty_clone_stops_without_touching_changes(origin_with_two_commits, tmp_path):
    origin, old, new = origin_with_two_commits
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "checkout", "-q", "--detach", old)
    (clone / "f.txt").write_text("local edit\n")
    (clone / "untracked.py").write_text("x = 1\n")

    with pytest.raises(RuntimeError, match="uncommitted"):
        _helpers()["checkout_pinned"](clone, str(origin), new)

    assert _git(clone, "rev-parse", "HEAD") == old
    assert (clone / "f.txt").read_text() == "local edit\n"
    assert (clone / "untracked.py").is_file()


def test_unknown_sha_checkout_fails(origin_with_two_commits, tmp_path):
    origin, _old, _new = origin_with_two_commits
    with pytest.raises(subprocess.CalledProcessError):
        _helpers()["checkout_pinned"](tmp_path / "c", str(origin), "0" * 40)


def test_run_helper_raises_on_nonzero_exit():
    with pytest.raises(subprocess.CalledProcessError):
        _helpers()["run"]([sys.executable, "-c", "raise SystemExit(3)"])


def test_failed_pilot_cli_blocks_audit_and_packaging(tmp_path):
    cells = _cells()
    ns = _helpers()

    def failing_run(cmd, cwd=None):
        raise subprocess.CalledProcessError(2, cmd)

    ns.update(run=failing_run, REPO_DIR=tmp_path, EXPECTED_COMMIT="a" * 40, RUN_ID="r1",
              RUNS_ROOT=tmp_path / "runs", RESUME_RUN_ID=None, EDIT_PLAN_PATH=tmp_path / "plan.json")
    with pytest.raises(subprocess.CalledProcessError):
        exec(cells[3], ns)
    assert "PILOT_RUN_COMPLETED" not in ns

    with pytest.raises(RuntimeError, match="did not complete"):
        exec(cells[4], ns)
    assert not list(tmp_path.rglob("*.zip"))


def test_drive_mount_failure_stops_and_creates_no_scratch_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def broken_mount(mount_point, **kwargs):
        raise OSError("mount failed")

    with pytest.raises(RuntimeError, match="Drive"):
        _helpers()["mount_drive_or_fail"](broken_mount, tmp_path / "drive")
    assert list(tmp_path.iterdir()) == []

    # A mount call that "returns" but leaves no MyDrive must also stop.
    with pytest.raises(RuntimeError, match="MyDrive"):
        _helpers()["mount_drive_or_fail"](lambda mp, **kw: None, tmp_path / "drive2")
    assert list(tmp_path.iterdir()) == []


def test_notebook_has_no_shell_magics_or_scratch_fallback():
    code = "\n".join(_cells()[1:])
    assert not re.search(r"^\s*!", code, re.M), "shell magics ignore exit status"
    assert "/content/phase_4c7b" not in code
    assert "DRIVE_AVAILABLE = False" not in code
    assert '"full"' not in code and "'full'" not in code, "notebook only launches the pilot"
    assert '"independent_cohort_acquisition" / "runs"' in code
    assert '"--edit-plan-path", EDIT_PLAN_PATH' in code


def test_notebook_pin_is_full_sha_in_history_and_covers_functional_code():
    m = re.search(r'EXPECTED_COMMIT\s*=\s*"([0-9a-f]+)"', _cells()[2])
    assert m and len(m.group(1)) == 40
    sha = m.group(1)
    assert "checkout_pinned(" in _cells()[2] and "EXPECTED_COMMIT" in _cells()[2]
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=REPO_ROOT, check=True)
    changed = _git(REPO_ROOT, "diff", "--name-only", sha, "HEAD", "--", "ml", "scripts")
    assert changed == "", f"functional code changed after the notebook pin: {changed}"


# ---------------------------------------------------------------- eligibility


def test_placeholder_creator_is_rejected():
    e = synthetic_coco_entry(1, creator_name="flickr_contributor_466319")
    res = evaluate_candidate_eligibility(e, set())
    assert res["status"] == "EXCLUDED"
    assert "CREATOR_PLACEHOLDER" in res["reasons"]


def test_unverified_creator_and_missing_provenance_are_rejected():
    e = synthetic_coco_entry(1)
    del e["creator_verification"]
    del e["provenance_checked_at_utc"]
    res = evaluate_candidate_eligibility(e, set())
    assert res["status"] == "EXCLUDED"
    assert "CREATOR_UNVERIFIED" in res["reasons"]
    assert "PROVENANCE_FIELD_MISSING:provenance_checked_at_utc" in res["reasons"]


@pytest.mark.parametrize("override", [{"source_keys": ["coco:"]}, {"source_keys": "coco:1"}, {"license_name": ""},
                                      {"creator_name": ""}])
def test_empty_or_malformed_required_fields_are_rejected(override):
    assert evaluate_candidate_eligibility(synthetic_coco_entry(1, **override), set())["status"] == "EXCLUDED"


def test_nonempty_but_unverified_fields_are_not_verified():
    e = synthetic_coco_entry(1, creator_verification="CLAIMED", creator_evidence_url="")
    assert evaluate_candidate_eligibility(e, set())["status"] == "EXCLUDED"


@pytest.mark.parametrize("code", ["by-nd", "by-nc-nd"])
def test_nd_candidates_not_eligible_for_edited_export_by_default(code):
    res = evaluate_candidate_eligibility(synthetic_coco_entry(1, license_code=code), set())
    assert res["status"] == "EXCLUDED"
    assert "LICENSE_ND_NO_EDITED_EXPORT" in res["reasons"]

    granted = synthetic_coco_entry(1, license_code=code, special_permission_evidence="https://x.invalid/grant")
    assert evaluate_candidate_eligibility(granted, set())["status"] != "EXCLUDED"


@pytest.mark.parametrize("code,needle", [("by-nc", "NonCommercial"), ("by-sa", "ShareAlike"), ("by-nc-sa", "ShareAlike")])
def test_nc_sa_candidates_carry_usage_policy(code, needle):
    res = evaluate_candidate_eligibility(synthetic_coco_entry(1, license_code=code), set())
    assert res["status"] == "ELIGIBLE_WITH_POLICY"
    assert needle in res["usage_policy"] and "Attribution" in res["usage_policy"]


def test_unsplash_lite_dataset_channel_is_not_eligible_for_edited_export():
    e = synthetic_coco_entry(1, acquisition_channel="unsplash_lite_dataset")
    res = evaluate_candidate_eligibility(e, set())
    assert res["status"] == "EXCLUDED"
    assert "CHANNEL_TERMS_DO_NOT_COVER_EDITED_EXPORT" in res["reasons"]


def test_historical_keys_are_namespaced_coco_ids_from_committed_map():
    keys = load_historical_source_keys()
    assert len(keys) == 684
    assert all(k.startswith("coco:") for k in keys)
    assert "coco:2261" in keys


def test_overlap_detected_by_namespaced_source_id_not_prefix():
    e = synthetic_coco_entry(1, source_keys=["coco:2261", "flickr:1"])
    res = evaluate_candidate_eligibility(e, load_historical_source_keys())
    assert "HISTORICAL_OPTION_P_OVERLAP" in res["reasons"]


def test_downloader_blocks_historical_namespaced_overlap(tmp_path):
    # Bypass plan-time filtering to prove the downloader guard is independent.
    from ml.evaluation.independent_cohort_acquisition import CandidateSpec
    spec = CandidateSpec(
        candidate_id="X", stratum_id="coco_sd2", source_origin="coco_2017", tool_key="t",
        modification_type="object_replacement", mask_area_class="small_under_10pct",
        origin_id="coco:2261", author="SYNTHETIC", origin_url="https://x.invalid", download_url="https://x.invalid/a.jpg",
        license_name="l", license_evidence_source="e", published_date="", prompt="p", generation_seed=1,
        pool_index=0, source_keys=("coco:2261",),
    )
    with pytest.raises(SourceOverlapError):
        download_authentic_image(spec, hist_keys=load_historical_source_keys())


def test_legacy_catalog_has_zero_eligible_candidates_and_plan_fails_closed():
    legacy = json.loads(LEGACY_CATALOG.read_text(encoding="utf-8"))
    hist = load_historical_source_keys()
    results = [evaluate_candidate_eligibility(e, hist) for e in legacy["coco_candidates"] + legacy["unsplash_candidates"]]
    assert len(results) == 440
    assert all(r["status"] == "EXCLUDED" for r in results)
    with pytest.raises(CandidateEligibilityError):
        generate_canonical_candidate_plan(catalog_path=LEGACY_CATALOG)


def test_new_catalog_keeps_quota_and_skips_overlapping_and_nd_entries(tmp_path):
    cat = build_synthetic_catalog(per_source=225)
    cat["coco_candidates"][0]["source_keys"] = ["coco:2261"]  # historical overlap
    cat["coco_candidates"][1] = synthetic_coco_entry(1, license_code="by-nd")
    cat["unsplash_candidates"][3]["creator_name"] = "photographer_abc"
    path = write_synthetic_catalog(tmp_path / "cat.json", cat)

    plan = generate_canonical_candidate_plan(catalog_path=path)

    assert len(plan) == 440
    keys = {k for c in plan for k in c.source_keys}
    assert "coco:2261" not in keys
    assert not any("by-nd" in c.license_url for c in plan)
    for s in STRATA_KEYS:
        sc = [c for c in plan if c.stratum_id == s]
        assert len(sc) == 110
        assert sum(c.modification_type == "object_replacement" for c in sc) == 44
        assert sum(c.mask_area_class == "medium_10_to_30pct" for c in sc) == 44
    assert all(c.is_synthetic for c in plan)  # synthetic catalog stays synthetic


def test_catalog_with_too_few_eligible_candidates_fails_closed(tmp_path):
    cat = build_synthetic_catalog(per_source=220)
    cat["unsplash_candidates"][5]["license_url"] = "https://creativecommons.org/licenses/by-nd/2.0/"
    with pytest.raises(CandidateEligibilityError, match="219/220"):
        generate_canonical_candidate_plan(catalog_path=write_synthetic_catalog(tmp_path / "c.json", cat))


# ---------------------------------------------------------------- run binding / audit


@pytest.fixture()
def synthetic_plan(tmp_path):
    return generate_canonical_candidate_plan(catalog_path=write_synthetic_catalog(tmp_path / "cat.json"))


def _run(out: Path, plan, binding: dict, **kw):
    return execute_cohort_acquisition(
        output_dir=out, candidate_specs=plan, engine_provider=lambda _: MockInpaintingEngine(),
        target_per_stratum=2, mode="fixture_test", allow_synthetic=True, run_binding=binding, **kw,
    )


def test_audit_passes_for_matching_binding(tmp_path, synthetic_plan):
    out = tmp_path / "run"
    _run(out, synthetic_plan, _binding())
    summary = audit_acquisition_run(out, _binding())
    assert summary["status"] == "PASS"
    assert summary["pairs"] == 8


@pytest.mark.parametrize("field,value", [("run_id", "run-B"), ("git_commit", "b" * 40), ("catalog_sha256", "other"),
                                         ("plan_sha256", "other"), ("protocol_sha256", "other")])
def test_stale_or_foreign_receipt_cannot_pass_new_session(tmp_path, synthetic_plan, field, value):
    out = tmp_path / "run"
    _run(out, synthetic_plan, _binding())
    expected = _binding() | {field: value}
    with pytest.raises(RunAuditError, match=field):
        audit_acquisition_run(out, expected)


def test_audit_rejects_unbound_legacy_output(tmp_path, synthetic_plan):
    out = tmp_path / "legacy"
    _run(out, synthetic_plan, _binding())
    (out / "run_receipt.json").unlink()
    with pytest.raises(RunAuditError, match="run_receipt.json"):
        audit_acquisition_run(out, _binding())


def test_audit_rejects_ledger_rows_from_another_run(tmp_path, synthetic_plan):
    out = tmp_path / "run"
    _run(out, synthetic_plan, _binding())
    ledger = out / "provenance_ledger.jsonl"
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    rows[0]["run_id"] = "run-OLD"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RunAuditError, match="run_id"):
        audit_acquisition_run(out, _binding())


def test_audit_rejects_tampered_image(tmp_path, synthetic_plan):
    out = tmp_path / "run"
    _run(out, synthetic_plan, _binding())
    img = next((out / "images").glob("*_edit.png"))
    img.write_bytes(img.read_bytes() + b"x")
    with pytest.raises(RunAuditError, match="sha256"):
        audit_acquisition_run(out, _binding())


def test_resume_refuses_artifacts_bound_to_another_run(tmp_path, synthetic_plan):
    out = tmp_path / "run"
    _run(out, synthetic_plan, _binding("run-A"))
    with pytest.raises(RunBindingMismatchError):
        _run(out, synthetic_plan, _binding("run-B"))


def test_run_receipt_records_generation_and_timing(tmp_path, synthetic_plan):
    out = tmp_path / "run"
    _run(out, synthetic_plan, _binding())
    receipt = json.loads((out / "run_receipt.json").read_text())
    assert receipt["binding"] == _binding()
    assert set(receipt["strata"]) == set(STRATA_KEYS)
    for s in receipt["strata"].values():
        assert s["elapsed_seconds"] >= 0
        assert s["peak_vram_bytes"] == "NOT_MEASURED"  # mock engine has no GPU
        assert s["engine"]["engine"] == "MockInpaintingEngine"
    attempts = [json.loads(x) for x in (out / "attempt_ledger.jsonl").read_text().splitlines()]
    assert all("elapsed_seconds" in a and a["run_id"] == "run-A" for a in attempts)


def test_full_mode_quota_preservation_counts_new_acceptances(tmp_path, synthetic_plan):
    """Regression: per-stratum quota counters were only updated on resume, and the plan layout
    could not replace a failed core slot. One core candidate per stratum fails QC here; the
    buffer must replace it while keeping every quota exact."""
    import numpy as np
    from PIL import Image

    failing = {c.candidate_id for c in synthetic_plan if c.pool_index == 7}

    def provider(spec):
        if spec.candidate_id in failing:
            return Image.new("RGB", (512, 512), (0, 0, 0))  # solid image -> QC failure
        rng = np.random.default_rng(spec.generation_seed)
        return Image.fromarray(rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8), mode="RGB")

    res = execute_cohort_acquisition(
        output_dir=tmp_path / "full", candidate_specs=synthetic_plan, authentic_image_provider=provider,
        engine_provider=lambda _: MockInpaintingEngine(), mode="fixture_test", allow_synthetic=True,
        run_binding=_binding() | {"mode": "full", "target_per_stratum": 100},
    )
    assert len(failing) == 4
    assert res["total_valid_pairs"] == 400
    assert res["modification_type_counts"] == {"object_replacement": 160, "object_removal_and_infill": 120,
                                               "object_insertion": 120}
    assert res["mask_area_counts"] == {"small_under_10pct": 120, "medium_10_to_30pct": 160, "large_over_30pct": 120}


# ---------------------------------------------------------------- execution policy


def test_gpu_policy_requires_cuda_and_vram():
    with pytest.raises(RuntimeError, match="CUDA"):
        check_gpu_policy(False, 0, "")
    with pytest.raises(RuntimeError, match="VRAM"):
        check_gpu_policy(True, 8 * 1024**3, "Tesla P4")
    check_gpu_policy(True, 15_843_721_216, "Tesla T4")


def test_cli_refuses_mock_engine_for_real_acquisition():
    from scripts.research.run_cohort_acquisition import main
    with pytest.raises(SystemExit):
        main(["--mode", "pilot", "--engine", "mock", "--run-id", "r", "--expected-commit", "a" * 40,
              "--output-root", "/nonexistent"])


def test_cli_refuses_checkout_not_at_expected_commit(tmp_path):
    from scripts.research.run_cohort_acquisition import verify_checkout
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a").write_text("1")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "a")
    head = _git(tmp_path, "rev-parse", "HEAD")
    assert verify_checkout(tmp_path, head) == head
    with pytest.raises(SystemExit, match="HEAD"):
        verify_checkout(tmp_path, "b" * 40)
    (tmp_path / "a").write_text("2")
    with pytest.raises(SystemExit, match="clean"):
        verify_checkout(tmp_path, head)


# ---------------------------------------------------------------- catalog v2 builder (SYNTHETIC responses)

_SYNTH_IMG = {"id": 900000001, "license": 4, "file_name": "x.jpg", "width": 640, "height": 480,
              "coco_url": "http://images.cocodataset.org/test2017/x.jpg", "date_captured": "2013-01-01 00:00:00",
              "flickr_url": "http://farm1.staticflickr.com/1/12345678901_abcdef0123_z.jpg"}
_SYNTH_LICENSES = {4: {"id": 4, "name": "Attribution License", "url": "http://creativecommons.org/licenses/by/2.0/"},
                   6: {"id": 6, "name": "Attribution-NoDerivs License", "url": "http://creativecommons.org/licenses/by-nd/2.0/"}}
_SYNTH_OEMBED = {"author_name": "SYNTHETIC Person", "author_url": "https://www.flickr.com/photos/synthetic/",
                 "web_page": "https://www.flickr.com/photos/synthetic/12345678901/",
                 "license_url": "https://creativecommons.org/licenses/by/2.0/"}


def _builder():
    from scripts.research import build_verified_candidate_catalog as b
    return b


def test_builder_entry_verified_only_with_source_response():
    b = _builder()
    e = b.coco_entry(_SYNTH_IMG, _SYNTH_LICENSES, _SYNTH_OEMBED, "https://x.invalid/oembed", "f" * 64, "2026-01-01T00:00:00+00:00")
    assert e["flickr_photo_id"] == "12345678901"
    assert e["source_keys"] == ["coco:900000001", "flickr:12345678901"]
    assert e["creator_name"] == "SYNTHETIC Person" and e["creator_verification"] == "VERIFIED_FROM_SOURCE"
    assert e["license_version"] == "2.0" and e["published_date"] is None
    assert "size-'z'" in e["download_rendition"] and "not the original upload" in e["download_rendition"]
    assert evaluate_candidate_eligibility(e, set())["status"] == "ELIGIBLE"

    unverified = b.coco_entry(_SYNTH_IMG, _SYNTH_LICENSES, None, "", "", "2026-01-01T00:00:00+00:00")
    assert unverified["creator_name"] == ""
    assert "CREATOR_UNVERIFIED" in evaluate_candidate_eligibility(unverified, set())["reasons"]


def test_builder_rejects_license_conflict_and_nd():
    b = _builder()
    conflict = b.coco_entry(_SYNTH_IMG, _SYNTH_LICENSES, _SYNTH_OEMBED | {"license_url": "https://creativecommons.org/licenses/by-nc/2.0/"},
                            "https://x.invalid/o", "f" * 64, "2026-01-01T00:00:00+00:00")
    assert conflict["license_conflict"]
    assert evaluate_candidate_eligibility(conflict, set())["status"] == "EXCLUDED"
    no_cc_now = b.coco_entry(_SYNTH_IMG, _SYNTH_LICENSES, {k: v for k, v in _SYNTH_OEMBED.items() if k != "license_url"},
                             "https://x.invalid/o", "f" * 64, "2026-01-01T00:00:00+00:00")
    assert evaluate_candidate_eligibility(no_cc_now, set())["status"] == "EXCLUDED"
    nd = b.coco_entry(_SYNTH_IMG | {"license": 6}, _SYNTH_LICENSES, _SYNTH_OEMBED | {"license_url": "https://creativecommons.org/licenses/by-nd/2.0/"},
                      "https://x.invalid/o", "f" * 64, "2026-01-01T00:00:00+00:00")
    assert "LICENSE_ND_NO_EDITED_EXPORT" in evaluate_candidate_eligibility(nd, set())["reasons"]


def test_flickr_short_url_is_base58_of_photo_id():
    b = _builder()
    code = b.flickr_short_url("12345678901").rsplit("/", 1)[1]
    n = 0
    for ch in code:
        n = n * 58 + b.BASE58.index(ch)
    assert n == 12345678901


# ---------------------------------------------------------------- model preflight & mirror tests


class _MockHTTPResponse:
    def __init__(self, status: int, data: bytes, headers: dict | None = None) -> None:
        self.status = status
        self._data = data
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _MockOpener:
    def __init__(self, handler) -> None:
        self.handler = handler

    def open(self, req, timeout=15.0):
        return self.handler(req)


def test_model_preflight_fails_on_401_deprecated_repo():
    import io
    import urllib.error

    def mock_401(req):
        fp = io.BytesIO(b'{"error":"Invalid username or password."}')
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, fp)

    with pytest.raises(ModelPreflightError) as exc_info:
        verify_model_access_preflight(
            ["stable_diffusion_2_inpainting"],
            http_opener=_MockOpener(mock_401),
        )
    err = exc_info.value
    assert err.status_code == 401
    assert err.step == "metadata_api"
    assert err.error_category == "repository_unavailable"
    assert "sd2-community/stable-diffusion-2-inpainting" in str(err)
    assert "deprecated" in str(err).lower() or "unavailable" in str(err).lower()


def test_model_preflight_fails_on_404_not_found():
    import io
    import urllib.error

    def mock_404(req):
        fp = io.BytesIO(b'{"error":"Repository not found"}')
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, fp)

    with pytest.raises(ModelPreflightError) as exc_info:
        verify_model_access_preflight(
            ["stable_diffusion_2_inpainting"],
            http_opener=_MockOpener(mock_404),
        )
    err = exc_info.value
    assert err.status_code == 404
    assert err.step == "metadata_api"
    assert err.error_category == "repository_not_found"


def test_model_preflight_fails_on_network_error():
    import urllib.error

    def mock_net_err(req):
        raise urllib.error.URLError("Connection refused")

    with pytest.raises(ModelPreflightError) as exc_info:
        verify_model_access_preflight(
            ["stable_diffusion_2_inpainting"],
            http_opener=_MockOpener(mock_net_err),
        )
    err = exc_info.value
    assert err.step == "metadata_api"
    assert err.error_category == "network_error"


def test_model_preflight_fails_on_gated_or_private_repo():
    def mock_gated(req):
        return _MockHTTPResponse(200, json.dumps({"gated": True, "sha": "a" * 40}).encode("utf-8"))

    with pytest.raises(ModelPreflightError) as exc_info:
        verify_model_access_preflight(
            ["stable_diffusion_2_inpainting"],
            http_opener=_MockOpener(mock_gated),
        )
    assert exc_info.value.error_category == "gated_repository"


def test_model_preflight_live_succeeds_for_mirror_and_sdxl():
    try:
        results = verify_model_access_preflight()
    except ModelPreflightError as e:
        if "WinError 10013" in str(e) or "access permissions" in str(e):
            pytest.skip(f"Transient Windows socket permission error during test suite: {e}")
        raise
    assert "stable_diffusion_2_inpainting" in results
    assert "sdxl_inpainting" in results

    sd2 = results["stable_diffusion_2_inpainting"]
    assert sd2["checkpoint"] == "sd2-community/stable-diffusion-2-inpainting"
    assert sd2["checkpoint_source_type"] == "community_mirror"
    assert sd2["upstream_original_checkpoint"] == "stabilityai/stable-diffusion-2-inpainting"
    assert len(sd2["resolved_revision"]) == 40
    assert sd2["resolved_revision"] == "5f74973cbb64c8568780732c17f43eb269d63a0d"
    assert sd2["num_inference_steps"] == 50
    assert sd2["guidance_scale"] == 7.5
    assert "unet/config.json" in sd2["verified_configs"]

    sdxl = results["sdxl_inpainting"]
    assert sdxl["checkpoint"] == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    assert sdxl["checkpoint_source_type"] == "official_diffusers"
    assert len(sdxl["resolved_revision"]) == 40
    assert sdxl["resolved_revision"] == "115134f363124c53c7d878647567d04daf26e41e"
    assert sdxl["num_inference_steps"] == 30
    assert sdxl["guidance_scale"] == 7.5


def test_diffusers_engine_binds_resolved_revision_and_metadata():
    reg = INPAINTING_MODEL_REGISTRY["stable_diffusion_2_inpainting"]
    assert reg["checkpoint"] == "sd2-community/stable-diffusion-2-inpainting"
    assert reg["checkpoint_source_type"] == "community_mirror"
    assert reg["upstream_original_checkpoint"] == "stabilityai/stable-diffusion-2-inpainting"
    assert reg["pinned_revision"] == "5f74973cbb64c8568780732c17f43eb269d63a0d"


def test_preflight_failure_does_not_create_run_directory(tmp_path, monkeypatch):
    from scripts.research.run_cohort_acquisition import main
    import subprocess

    def failing_preflight(*args, **kwargs):
        raise ModelPreflightError("Simulated preflight failure", repo_id="broken-model", step="metadata_api")

    monkeypatch.setattr(
        "scripts.research.run_cohort_acquisition.verify_model_access_preflight",
        failing_preflight,
    )
    monkeypatch.setattr(
        "scripts.research.run_cohort_acquisition.verify_checkout",
        lambda repo_root, expected: expected,
    )
    monkeypatch.setattr(
        "scripts.research.run_cohort_acquisition.load_content_grounded_edit_plan",
        lambda candidates, *args, **kwargs: candidates,
    )

    out_root = tmp_path / "runs"
    run_id = "test-fail-preflight-run"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()

    with pytest.raises(SystemExit) as exc:
        main([
            "--mode", "pilot",
            "--engine", "diffusers",
            "--run-id", run_id,
            "--expected-commit", head,
            "--output-root", str(out_root),
        ])

    assert exc.value.code == 1
    # Fail-closed verification: output directory must NOT be created!
    assert not (out_root / run_id).exists()


def test_cli_check_models_flag_executes_successfully():
    from scripts.research.run_cohort_acquisition import main

    try:
        main(["--check-models"])
    except ModelPreflightError as e:
        if "WinError 10013" in str(e) or "access permissions" in str(e):
            pytest.skip(f"Transient Windows socket permission error during test suite: {e}")
        raise


def test_diffusers_engine_passes_explicit_height_width_and_validates_dimensions():
    """Verify Diffusers inpainting engine passes explicit 512x512 height/width and fails on mismatch."""
    engine = object.__new__(DiffusersInpaintingEngine)
    engine.tool_key = "sdxl_inpainting"
    engine.device = "cpu"
    engine.num_steps = 30
    engine.guidance_scale = 7.5
    engine.torch = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.return_value.images = [Image.new("RGB", (512, 512))]
    engine.pipeline = mock_pipeline

    auth = Image.new("RGB", (512, 512))
    mask = Image.new("L", (512, 512))

    res = engine.inpaint(auth, mask, prompt="test prompt", seed=42)
    assert res.size == (512, 512)
    assert mock_pipeline.call_count == 1
    call_kwargs = mock_pipeline.call_args[1]
    assert call_kwargs["height"] == 512
    assert call_kwargs["width"] == 512
    assert call_kwargs["prompt"] == "test prompt"

    # Input authentic size mismatch raises GenerationContractError
    bad_auth = Image.new("RGB", (256, 256))
    with pytest.raises(GenerationContractError, match="Input authentic image size/mode invalid"):
        engine.inpaint(bad_auth, mask, prompt="prompt", seed=42)

    # Input mask mode mismatch raises GenerationContractError
    bad_mask = Image.new("RGB", (512, 512))
    with pytest.raises(GenerationContractError, match="Input mask image size/mode invalid"):
        engine.inpaint(auth, bad_mask, prompt="prompt", seed=42)

    # Pipeline output 1024x1024 raises GenerationContractError
    mock_pipeline.return_value.images = [Image.new("RGB", (1024, 1024))]
    with pytest.raises(GenerationContractError, match="produced invalid output size/mode"):
        engine.inpaint(auth, mask, prompt="prompt", seed=42)


def test_output_1024_halts_immediately_and_does_not_try_next_candidate(tmp_path):
    """Verify that 1024x1024 output raises GenerationContractError, writes failure receipt, and halts immediately without trying next candidate."""
    plan = generate_synthetic_fixture_plan()
    run_dir = tmp_path / "run_fail_1024"
    binding = _binding("run-fail-1024")

    calls = 0

    class Failing1024Engine:
        def inpaint(self, auth, mask, prompt, seed):
            nonlocal calls
            calls += 1
            return Image.new("RGB", (1024, 1024))

    with pytest.raises(GenerationContractError):
        execute_cohort_acquisition(
            output_dir=run_dir,
            candidate_specs=plan,
            engine_provider=(lambda _: Failing1024Engine()),
            target_per_stratum=2,
            mode="fixture_test",
            allow_synthetic=True,
            run_binding=binding,
        )

    # Must halt immediately on candidate 1, never trying the subsequent candidates
    assert calls == 1

    # Verify attempt ledger recorded the contract error
    attempt_file = run_dir / "attempt_ledger.jsonl"
    assert attempt_file.is_file()
    records = [json.loads(line) for line in attempt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["status"] == "GENERATION_CONTRACT_ERROR"

    # Verify failure receipt was written
    fail_receipt_file = run_dir / "failure_receipt.json"
    assert fail_receipt_file.is_file()
    fail_data = json.loads(fail_receipt_file.read_text(encoding="utf-8"))
    assert fail_data["status"] == "FAILED"
    assert fail_data["failure_type"] == "GENERATION_CONTRACT_ERROR"
    assert "1024" in fail_data["error_message"]

    # Completion receipt must NOT exist
    assert not (run_dir / "run_receipt.json").exists()


def test_output_correct_size_proceeds_through_qc_and_handles_technical_qc_rejection(tmp_path):
    """Verify that 512x512 images proceed and candidate-level Technical QC failures trigger replacement."""
    plan = generate_synthetic_fixture_plan()
    run_dir = tmp_path / "run_qc_replace"
    binding = _binding("run-qc-replace")

    candidate_call_count = 0

    def corrupting_first_candidate_provider(spec: CandidateSpec):
        nonlocal candidate_call_count
        candidate_call_count += 1
        if candidate_call_count == 1:
            # Degenerate solid black image (Technical QC failure, not a generation-contract violation)
            img = Image.new("RGB", (512, 512), color=(0, 0, 0))
            return img, b"solid", "sha_solid"
        rng = np.random.default_rng(spec.generation_seed)
        arr = rng.integers(30, 220, size=(512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        return img, b"valid", f"sha_{candidate_call_count}"

    result = execute_cohort_acquisition(
        output_dir=run_dir,
        candidate_specs=plan,
        engine_provider=(lambda _: MockInpaintingEngine()),
        authentic_image_provider=corrupting_first_candidate_provider,
        target_per_stratum=2,
        mode="fixture_test",
        allow_synthetic=True,
        run_binding=binding,
    )

    assert result["total_valid_pairs"] == 8
    attempt_file = run_dir / "attempt_ledger.jsonl"
    records = [json.loads(line) for line in attempt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    # At least one QC_FAILED record
    qc_failed = [r for r in records if r["status"] == "QC_FAILED"]
    assert len(qc_failed) >= 1
    assert "Degenerate solid" in qc_failed[0]["reason"]
    # And successful ACCEPTED records
    accepted = [r for r in records if r["status"] == "ACCEPTED"]
    assert len(accepted) == 8


def test_generation_contract_failure_prevents_completion_receipt_and_blocks_audit(tmp_path):
    """Verify that a run with a failure receipt cannot pass run audit and cannot be packaged."""
    run_dir = tmp_path / "failed_run"
    run_dir.mkdir(parents=True)
    binding = _binding("failed-run-id")
    (run_dir / "run_binding.json").write_text(json.dumps(binding), encoding="utf-8")
    fail_receipt = {
        "run_id": "failed-run-id",
        "status": "FAILED",
        "failure_type": "GENERATION_CONTRACT_ERROR",
        "error_message": "Output size invalid: (1024, 1024)",
    }
    (run_dir / "failure_receipt.json").write_text(json.dumps(fail_receipt), encoding="utf-8")

    # audit_acquisition_run must fail closed on failure_receipt.json
    with pytest.raises(RunAuditError, match="Run terminated with failure receipt"):
        audit_acquisition_run(run_dir, binding)

    # Notebook cell 4 check: without completion flag in session globals, packaging is refused
    cells = _cells()
    ns = _helpers()
    ns.update(REPO_DIR=tmp_path, EXPECTED_COMMIT="a" * 40, RUN_ID="failed-run-id", RUNS_ROOT=tmp_path,
              EDIT_PLAN_PATH=tmp_path / "plan.json")
    with pytest.raises(RuntimeError, match="did not complete"):
        exec(cells[4], ns)
    assert not list(tmp_path.rglob("*.zip"))
