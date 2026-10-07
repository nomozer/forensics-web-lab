"""
Phase 4C.2B Notebook Verification Test Suite.
Tests canonical 5-cell structure, EXECUTE=False preflight logic, and deduplication.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
CANONICAL_NOTEBOOK = NOTEBOOK_DIR / "partial_finetuning_colab.ipynb"


def test_29_notebook_5_cell_structure():
    """29. Notebook must have exactly 5 cells (1 markdown + 4 code) and 0 execution counts in git."""
    assert CANONICAL_NOTEBOOK.exists(), f"Notebook missing at {CANONICAL_NOTEBOOK}"

    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    assert len(cells) == 5, f"Expected exactly 5 cells, got {len(cells)}"

    # Cell 0: markdown
    assert cells[0]["cell_type"] == "markdown"
    title = "".join(cells[0]["source"])
    assert "Partial Fine-Tuning Colab Launcher" in title
    assert "phase trace 4C.2" in title

    # Cells 1-4: code
    for idx, c in enumerate(cells[1:], start=1):
        assert c["cell_type"] == "code", f"Cell {idx} must be code cell"
        assert c.get("execution_count") is None, f"Cell {idx} has execution_count in git!"
        assert c.get("outputs") == [], f"Cell {idx} has non-empty outputs in git!"


def test_30_notebook_execute_false_invokes_preflight_only():
    """30. Notebook sets EXECUTE=False by default and invokes --preflight-only conditionally."""
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_1_src = "".join(nb["cells"][1]["source"])
    assert "EXECUTE = False" in cell_1_src, "Default must be EXECUTE = False"

    cell_3_src = "".join(nb["cells"][3]["source"])
    assert "--preflight-only" in cell_3_src
    assert "--execute" in cell_3_src
    assert "if EXECUTE:" in cell_3_src or "if not EXECUTE:" in cell_3_src


def test_31_no_duplicate_notebook():
    """31. Exactly one Stage 2 notebook exists; no duplicate _v2, (1), etc."""
    nb_files = list(NOTEBOOK_DIR.glob("*partial_finetuning*colab*.ipynb"))
    assert len(nb_files) == 1, f"Found multiple Stage 2 notebooks: {[f.name for f in nb_files]}"
    assert nb_files[0].name == "partial_finetuning_colab.ipynb"


def test_33_notebook_operator_verification_and_restaging():
    """33. Notebook verifies operator SHA/bytes, checks Python runtime capability, and enforces atomic restaging."""
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_1_src = "".join(nb["cells"][1]["source"])
    assert "CANONICAL_OPERATOR_SHA = " in cell_1_src
    assert "CANONICAL_OPERATOR_BYTES = " in cell_1_src

    cell_2_src = "".join(nb["cells"][2]["source"])
    assert "assert sys.version_info >= (3, 10)" in cell_2_src
    assert "CANONICAL_OPERATOR_BYTES" in cell_2_src
    assert "CANONICAL_OPERATOR_SHA" in cell_2_src
    assert "sha256_file(operator_script)" in cell_2_src
    assert "os.replace(part_dst, dst)" in cell_2_src
    assert "needs_copy = True" in cell_2_src


def test_34_notebook_exact_archive_binding():
    """34. Notebook binds exact code archive filename, bytes, streaming sha, and rejects glob."""
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_1_src = "".join(nb["cells"][1]["source"])
    assert 'CANONICAL_CODE_ARCHIVE_NAME = "phase_4c2_code_9ee7fdb.tar.gz"' in cell_1_src
    assert "CANONICAL_CODE_ARCHIVE_BYTES = 10478136" in cell_1_src
    assert 'CANONICAL_CODE_ARCHIVE_SHA256 = "951e9089582eb60cf3d293c37982a8f3c3b6a3e05fb3ef45f23c666d41bc7d89"' in cell_1_src

    cell_2_src = "".join(nb["cells"][2]["source"])
    assert 'code_archive = DRIVE_INPUT_DIR / CANONICAL_CODE_ARCHIVE_NAME' in cell_2_src
    assert 'glob("phase_4c2_code_*.tar.gz")' not in cell_2_src
    assert 'code_archives[-1]' not in cell_2_src
    assert 'assert code_archive.stat().st_size == CANONICAL_CODE_ARCHIVE_BYTES' in cell_2_src
    assert 'drive_code_sha = sha256_file(code_archive)' in cell_2_src
    assert 'assert drive_code_sha == CANONICAL_CODE_ARCHIVE_SHA256' in cell_2_src
    assert 'os.replace(part_dst, dst)' in cell_2_src


def test_35_notebook_exact_execution_directory():
    """35. Notebook audit and error reporting use exact execution_9ee7fdb directory without rglob/sorted."""
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_1_src = "".join(nb["cells"][1]["source"])
    assert 'CANONICAL_EXECUTION_SHORT_SHA = "9ee7fdb"' in cell_1_src

    cell_3_src = "".join(nb["cells"][3]["source"])
    assert 'exact_execution_dir = DRIVE_OUTPUT_DIR / f"execution_{CANONICAL_EXECUTION_SHORT_SHA}"' in cell_3_src
    assert 'failure_report = exact_execution_dir / "OPERATOR_FAILURE.json"' in cell_3_src
    assert 'console_log = exact_execution_dir / "logs" / "operator_console.log"' in cell_3_src
    assert 'rglob("OPERATOR_FAILURE.json")' not in cell_3_src
    assert 'rglob("operator_console.log")' not in cell_3_src

    cell_4_src = "".join(nb["cells"][4]["source"])
    assert 'final_exec_dir = DRIVE_OUTPUT_DIR / f"execution_{CANONICAL_EXECUTION_SHORT_SHA}"' in cell_4_src
    assert 'glob("execution_*")' not in cell_4_src
# ------------------------------------------------------------------------------
# Phase 4C.2B.3.2 Post-Execution Audit & Archive Parity Tests
# ------------------------------------------------------------------------------

OPERATOR_SCRIPT = REPO_ROOT / "scripts" / "run_partial_finetuning_all.sh"


def extract_operator_package_and_hash_names() -> list[str]:
    """Extract exact archive names passed to package_and_hash() in the operator."""
    import re
    assert OPERATOR_SCRIPT.exists(), f"Operator missing at {OPERATOR_SCRIPT}"
    op_text = OPERATOR_SCRIPT.read_text(encoding="utf-8")
    names = re.findall(r'package_and_hash\("([^"]+)"', op_text)
    return names


def extract_notebook_required_archives(nb_json: dict) -> list[str]:
    """Extract REQUIRED_ARCHIVES list defined in cell 4 of canonical notebook."""
    import re
    cell_4_src = "".join(nb_json["cells"][4]["source"])
    match = re.search(r'REQUIRED_ARCHIVES\s*=\s*\[(.*?)\]', cell_4_src, re.DOTALL)
    assert match, "REQUIRED_ARCHIVES not found in notebook cell 4"
    return re.findall(r'"([^"]+\.tar\.gz)"', match.group(1))


def create_valid_execution_fixture(base_dir: Path, short_sha: str = "9ee7fdb") -> Path:
    """Create a fully compliant mock execution output directory with 15 runs and 5 archives."""
    import hashlib
    exec_dir = base_dir / f"execution_{short_sha}"
    exec_dir.mkdir(parents=True, exist_ok=True)

    # 1. OPERATOR_STATUS.json
    status = {
        "status": "completed",
        "mode": "execute",
        "stage2_invocations": 1,
        "training_runs_completed": 15,
        "execution_short_sha": short_sha,
        "timestamp_utc": "2026-10-01T10:48:24Z",
        "verdict": "READY_FOR_USER_COLAB_PREFLIGHT",
    }
    (exec_dir / "OPERATOR_STATUS.json").write_text(json.dumps(status), encoding="utf-8")

    # 2. download dir with 5 canonical archives & sidecars
    archive_names = [
        "execution_9ee7fdb_run_receipts_metrics.tar.gz",
        "execution_9ee7fdb_run_predictions.tar.gz",
        "execution_9ee7fdb_run_histories.tar.gz",
        "execution_9ee7fdb_run_checkpoints.tar.gz",
        "execution_9ee7fdb_environment_checksums.tar.gz",
    ]
    dl_dir = exec_dir / "download"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for name in archive_names:
        arch_file = dl_dir / name
        arch_file.write_bytes(b"dummy archive payload for " + name.encode("utf-8"))
        sha = hashlib.sha256(arch_file.read_bytes()).hexdigest()
        (dl_dir / f"{name}.sha256").write_text(f"{sha}  {name}\n", encoding="utf-8")

    # 3. Exactly 15 runs
    sample_sizes = (50, 100, 250)
    seeds = (42, 1337, 2025, 3407, 9001)
    for size in sample_sizes:
        for seed in seeds:
            run_dir = exec_dir / f"n{size}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            receipt = {
                "status": "completed",
                "stage": "partial_finetune",
                "sample_size": size,
                "seed": seed,
                "treatment_designation": "pre-registered partial fine-tuning protocol",
                "trainable_parameters_count": 204674,
                "validation_source_count": 91,
                "locked_test_access": 0,
                "stage1_output_writes": 0,
            }
            (run_dir / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    return exec_dir


def execute_notebook_cell_4(drive_output_dir: Path, short_sha: str = "9ee7fdb"):
    """Execute cell 4 audit logic in a sandboxed namespace with EXECUTE=True."""
    import hashlib
    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell_4_code = "".join(nb["cells"][4]["source"])

    def sha256_file(path, chunk_size=1024 * 1024):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # Capture/suppress print to avoid Windows charmap encoding issues
    captured_logs = []

    def safe_print(*args, **kwargs):
        captured_logs.append(" ".join(str(a) for a in args))

    global_ns = {
        "EXECUTE": True,
        "DRIVE_OUTPUT_DIR": drive_output_dir,
        "CANONICAL_EXECUTION_SHORT_SHA": short_sha,
        "json": json,
        "sha256_file": sha256_file,
        "print": safe_print,
    }
    exec(cell_4_code, global_ns)
    return captured_logs


def test_61_archive_names_operator_notebook_parity():
    """61. REQUIRED_ARCHIVES in notebook matches exact 5 names passed to package_and_hash in operator."""
    op_names = extract_operator_package_and_hash_names()
    assert len(op_names) == 5, f"Operator must define exactly 5 archives, got {len(op_names)}: {op_names}"

    with open(CANONICAL_NOTEBOOK, "r", encoding="utf-8") as f:
        nb = json.load(f)

    nb_names = extract_notebook_required_archives(nb)
    assert len(nb_names) == 5, f"Notebook must define exactly 5 REQUIRED_ARCHIVES, got {len(nb_names)}: {nb_names}"
    assert op_names == nb_names, f"Operator and notebook archive names mismatch: {op_names} != {nb_names}"


def test_62_post_execution_audit_valid_fixture_passes(tmp_path: Path):
    """62. Valid execution fixture with 15 runs and 5 archives passes notebook audit without errors."""
    create_valid_execution_fixture(tmp_path)
    logs = execute_notebook_cell_4(tmp_path)
    assert any("[PASS]" in log for log in logs), "Audit did not emit [PASS] log"


def test_63_post_execution_audit_fault_old_archives_fails(tmp_path: Path):
    """63. Fault injection: old archive names (n50_stage2_results, etc.) must fail closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    dl = exec_dir / "download"
    for f in dl.glob("*"):
        f.unlink()

    old_names = [
        "n50_stage2_results.tar.gz",
        "n100_stage2_results.tar.gz",
        "n250_stage2_results.tar.gz",
        "phase_4c2_execution_logs.tar.gz",
        "phase_4c2_all_15_runs_results.tar.gz",
    ]
    for name in old_names:
        (dl / name).write_bytes(b"old")
        (dl / f"{name}.sha256").write_text(f"dummy {name}\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="Thiếu archive"):
        execute_notebook_cell_4(tmp_path)


def test_64_post_execution_audit_fault_missing_archive_fails(tmp_path: Path):
    """64. Fault injection: missing one of the 5 canonical archives fails closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    (exec_dir / "download" / "execution_9ee7fdb_run_receipts_metrics.tar.gz").unlink()

    with pytest.raises(AssertionError, match="Thiếu archive"):
        execute_notebook_cell_4(tmp_path)


def test_65_post_execution_audit_fault_missing_sidecar_fails(tmp_path: Path):
    """65. Fault injection: missing sidecar sha256 file fails closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    (exec_dir / "download" / "execution_9ee7fdb_run_predictions.tar.gz.sha256").unlink()

    with pytest.raises(AssertionError, match="Thiếu sidecar"):
        execute_notebook_cell_4(tmp_path)


def test_66_post_execution_audit_fault_sha_mismatch_fails(tmp_path: Path):
    """66. Fault injection: corrupted sidecar checksum fails closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    bad_sidecar = exec_dir / "download" / "execution_9ee7fdb_run_histories.tar.gz.sha256"
    bad_sidecar.write_text("0000000000000000000000000000000000000000000000000000000000000000  bad\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="Sai SHA-256"):
        execute_notebook_cell_4(tmp_path)


def test_67_post_execution_audit_fault_status_not_completed_fails(tmp_path: Path):
    """67. Fault injection: OPERATOR_STATUS status != 'completed' fails closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    st_file = exec_dir / "OPERATOR_STATUS.json"
    st = json.loads(st_file.read_text(encoding="utf-8"))
    st["status"] = "in_progress"
    st_file.write_text(json.dumps(st), encoding="utf-8")

    with pytest.raises(AssertionError, match="không phải completed"):
        execute_notebook_cell_4(tmp_path)


def test_68_post_execution_audit_fault_incomplete_runs_fails(tmp_path: Path):
    """68. Fault injection: fewer than 15 run directories fails closed."""
    import shutil
    exec_dir = create_valid_execution_fixture(tmp_path)
    shutil.rmtree(exec_dir / "n250_seed_9001")

    with pytest.raises(AssertionError, match="Kỳ vọng đúng 15 run directories"):
        execute_notebook_cell_4(tmp_path)


def test_69_post_execution_audit_fault_locked_test_access_fails(tmp_path: Path):
    """69. Fault injection: locked_test_access != 0 in any receipt fails closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    rc_file = exec_dir / "n50_seed_42" / "run_receipt.json"
    rc = json.loads(rc_file.read_text(encoding="utf-8"))
    rc["locked_test_access"] = 1
    rc_file.write_text(json.dumps(rc), encoding="utf-8")

    with pytest.raises(AssertionError, match="locked_test_access khác 0"):
        execute_notebook_cell_4(tmp_path)


def test_70_post_execution_audit_fault_stage1_writes_fails(tmp_path: Path):
    """70. Fault injection: stage1_output_writes != 0 in any receipt fails closed."""
    exec_dir = create_valid_execution_fixture(tmp_path)
    rc_file = exec_dir / "n100_seed_1337" / "run_receipt.json"
    rc = json.loads(rc_file.read_text(encoding="utf-8"))
    rc["stage1_output_writes"] = 2
    rc_file.write_text(json.dumps(rc), encoding="utf-8")

    with pytest.raises(AssertionError, match="stage1_output_writes khác 0"):
        execute_notebook_cell_4(tmp_path)
