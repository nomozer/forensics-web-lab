#!/usr/bin/env python3
"""Build the deterministic self-contained Phase 4C.2H Colab code snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


SOURCE_MEMBERS = (
    "ml/evaluation/__init__.py",
    "ml/evaluation/calibration.py",
    "ml/evaluation/metrics.py",
    "ml/training/__init__.py",
    "ml/training/loss.py",
    "ml/training/mobilenetv3_forensics.py",
    "ml/training/phase_4c2h_development.py",
    "ml/training/run_phase_4c2h.py",
    "ml/configs/phase_4c2h_development_nested_cv.yaml",
    "ml/requirements.txt",
)
WEIGHTS_MEMBER = "models/research/pretrained/mobilenet_v3_small-047dcff4.pth"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(repo: Path, commit: str, member: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{member}"],
        cwd=repo,
        stderr=subprocess.STDOUT,
    )


def normalized_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mode = 0o644
    return info


def build_snapshot(
    *, repo: Path, source_commit: str, weights_path: Path, output: Path
) -> dict[str, object]:
    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", f"{source_commit}^{{commit}}"],
        cwd=repo,
        text=True,
    ).strip()
    if len(resolved_commit) != 40:
        raise ValueError("Source commit did not resolve to a full commit SHA")
    payloads = {member: git_blob(repo, resolved_commit, member) for member in SOURCE_MEMBERS}
    weights = weights_path.resolve(strict=True)
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("Weights must be a regular non-symlink file")
    payloads[WEIGHTS_MEMBER] = weights.read_bytes()
    members = [
        {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    ]
    manifest = {
        "schema_version": "1.0.0",
        "phase": "4C.2H",
        "source_commit": resolved_commit,
        "members": members,
    }
    payloads["SNAPSHOT_MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, payload in sorted(payloads.items()):
                    archive.addfile(normalized_info(name, len(payload)), io.BytesIO(payload))
    temporary.replace(output)
    return {
        "schema_version": "1.0.0",
        "phase": "4C.2H",
        "source_commit": resolved_commit,
        "archive": output.name,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "snapshot_manifest_sha256": sha256_bytes(payloads["SNAPSHOT_MANIFEST.json"]),
        "member_count_excluding_manifest": len(members),
        "weights": next(row for row in members if row["path"] == WEIGHTS_MEMBER),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_snapshot(
        repo=args.repo.resolve(strict=True),
        source_commit=args.source_commit,
        weights_path=args.weights,
        output=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
