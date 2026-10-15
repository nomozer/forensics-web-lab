#!/usr/bin/env python3
"""Builds the short AnyText Colab launcher notebook.

The notebook embeds no code payloads. It fetches this repository at a full commit SHA (never a branch head),
verifies the SHA-256 of every pinned source file, and calls scripts/research/anytext_colab_setup.py.

Usage:  python scripts/research/build_anytext_colab_notebook.py --source-commit <40-hex SHA>

The builder refuses a commit whose pinned files differ from the working tree, so the notebook can never
point at stale sources. Rebuild (and re-verify) after the last change to any pinned file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "independent_anytext_feasibility_colab.ipynb"
PLAN_REL = "research/evidence/document_pilot/anytext_feasibility_plan.json"
PINNED_FILES: List[str] = [
    "scripts/research/run_anytext_colab.py",
    "scripts/research/anytext_colab_setup.py",
    "scripts/research/anytext_requirements.txt",
    "ml/evaluation/anytext_document_adapter.py",
    "ml/evaluation/document_feasibility_geometry.py",
    PLAN_REL,
]
SPARSE_DIRS = ["scripts/research", "ml/evaluation", "research/evidence/document_pilot"]
ARCHIVE_NAME = "document_feasibility_results.zip"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned_hashes(source_commit: str, repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """SHA-256 of each pinned file at `source_commit`; raises if it differs from the working tree."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise SystemExit(f"--source-commit must be a full 40-hex SHA, got {source_commit!r}")
    hashes: Dict[str, str] = {}
    for rel in PINNED_FILES:
        blob = subprocess.run(
            ["git", "show", f"{source_commit}:{rel}"], cwd=repo_root, capture_output=True, check=False
        )
        if blob.returncode != 0:
            raise SystemExit(f"{rel} does not exist at {source_commit}")
        working = (repo_root / rel).read_bytes()
        if _sha256(blob.stdout) != _sha256(working):
            raise SystemExit(f"{rel} differs between {source_commit[:12]} and the working tree; commit it and pass that SHA")
        hashes[rel] = _sha256(blob.stdout)
    return hashes


def _code(title: str, lines: List[str]) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [f"# {title}\n"] + [ln + "\n" for ln in lines[:-1]] + [lines[-1]],
    }


def build_cells(source_commit: str, hashes: Dict[str, str]) -> List[dict]:
    plan = json.loads((REPO_ROOT / PLAN_REL).read_text(encoding="utf-8"))
    archive_sha = plan["input_source_specification"]["archive_sha256"]
    md = [
        "# AnyText Document Feasibility (Colab GPU) - launcher\n",
        "\n",
        "Chạy lần lượt 4 bước. Không cần dán thêm cell vá; mã nguồn được tải từ commit đã khóa và kiểm tra SHA-256.\n",
        "\n",
        "1. **Chuẩn bị và kiểm tra môi trường** (Python 3.10 cô lập, `pip check`, smoke test GPU fp16).\n",
        "2. **Nhận archive đầu vào và preflight** (tải lên `document_feasibility_results.zip`).\n",
        "3. **Chạy hai attempts** (A: CORD row 22, B: CORD row 61; hiển thị tiến độ, log đầy đủ lưu riêng).\n",
        "4. **Xem kết quả và tải archive** của đúng lần chạy hiện tại.\n",
        "\n",
        f"Nguồn: `nomozer/forensics-web-lab` @ `{source_commit}`. Cần runtime GPU (T4 trở lên), không cần token/đăng nhập.\n",
        "Human Content QC luôn ở trạng thái `PENDING_HUMAN_REVIEW`.",
    ]
    step1 = _code("Bước 1: Chuẩn bị và kiểm tra môi trường", [
        "import hashlib, pathlib, subprocess, sys",
        f"COMMIT = '{source_commit}'",
        f"PINNED = {json.dumps(hashes, indent=4)}",
        "REPO = pathlib.Path('/content/forensics-web-lab')",
        "SETUP = REPO / 'scripts/research/anytext_colab_setup.py'",
        "",
        "def sh(cmd, cwd=None):",
        "    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
        "    for line in p.stdout:",
        "        print(line, end='')",
        "    rc = p.wait()",
        "    assert rc == 0, f'{cmd[:3]} failed with exit code {rc}'",
        "",
        "REPO.mkdir(parents=True, exist_ok=True)",
        "if not (REPO / '.git').is_dir():",
        "    sh(['git', 'init', '-q'], cwd=REPO)",
        "    sh(['git', 'remote', 'add', 'origin', 'https://github.com/nomozer/forensics-web-lab.git'], cwd=REPO)",
        "sh(['git', 'fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', COMMIT], cwd=REPO)",
        f"sh(['git', 'sparse-checkout', 'set', '--cone', {', '.join(repr(d) for d in SPARSE_DIRS)}], cwd=REPO)",
        "sh(['git', 'checkout', '-q', '-f', COMMIT], cwd=REPO)",
        "head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()",
        "assert head == COMMIT, f'commit mismatch: {head}'",
        "for rel, sha in PINNED.items():",
        "    assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == sha, f'SHA-256 mismatch: {rel}'",
        "print(f'[PASS] {len(PINNED)} source files verified at {COMMIT[:12]}')",
        "sh([sys.executable, '-u', str(SETUP), 'env', '--repo-root', str(REPO)])",
    ])
    step2 = _code("Bước 2: Nhận archive đầu vào và preflight", [
        "import hashlib, pathlib",
        "from google.colab import files",
        f"ARCHIVE = pathlib.Path('/content/{ARCHIVE_NAME}')",
        f"ARCHIVE_SHA256 = '{archive_sha}'",
        "if not ARCHIVE.is_file():",
        f"    print('Tải lên {ARCHIVE_NAME}')",
        "    for name, data in files.upload().items():",
        f"        if name.endswith('.zip'):",
        "            ARCHIVE.write_bytes(data)",
        "assert ARCHIVE.is_file() and hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA256, 'archive SHA-256 mismatch'",
        "sh([sys.executable, '-u', str(SETUP), 'run', '--archive', str(ARCHIVE), '--preflight-only'])",
    ])
    step3 = _code("Bước 3: Chạy hai attempts (A và B)", [
        "import time",
        "RUN_STARTED = time.time()",
        "sh([sys.executable, '-u', str(SETUP), 'run', '--archive', str(ARCHIVE), '--label', 'repair_fp16_xformers'])",
    ])
    step4 = _code("Bước 4: Xem kết quả và tải archive của lần chạy hiện tại", [
        "import datetime, hashlib, json, pathlib",
        "from IPython.display import Image, display",
        "from google.colab import files",
        "out = pathlib.Path('/content/outputs')",
        "ptr = json.loads((out / 'latest_run.json').read_text(encoding='utf-8'))",
        "assert datetime.datetime.fromisoformat(ptr['timestamp']).timestamp() >= RUN_STARTED, 'latest_run.json belongs to an older run'",
        "sheet = out / ptr['run_dir'] / 'anytext_contact_sheet.png'",
        "if sheet.is_file():",
        "    display(Image(str(sheet)))",
        "zip_path = out / ptr['archive']",
        "assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == ptr['archive_sha256'], 'archive hash mismatch'",
        "print(f\"Run {ptr['run_id']}: {ptr['status']} -> {zip_path.name} ({zip_path.stat().st_size / 2**20:.1f} MiB)\")",
        "files.download(str(zip_path))",
    ])
    return [{"cell_type": "markdown", "metadata": {}, "source": md}, step1, step2, step3, step4]


def build_notebook(source_commit: str) -> None:
    hashes = pinned_hashes(source_commit)
    nb = {
        "cells": build_cells(source_commit, hashes),
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Launcher notebook written: {NOTEBOOK_PATH}")
    print(f"     Source commit: {source_commit}")
    print(f"     SHA-256: {_sha256(NOTEBOOK_PATH.read_bytes())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-commit", required=True)
    build_notebook(ap.parse_args().source_commit)
