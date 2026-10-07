#!/usr/bin/env python3
"""Build the deterministic self-contained Visual / DSP Ablation Colab package."""

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
    "ml/training/dsp_features.py",
    "ml/training/mobilenetv3_forensics.py",
    "ml/training/nested_cv_development.py",
    "ml/training/visual_dsp_ablation.py",
    "ml/training/run_visual_dsp_ablation.py",
    "scripts/research/analyze_visual_dsp_ablation.py",
    "ml/configs/visual_dsp_ablation_protocol.yaml",
    "ml/requirements.txt",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_package(
    *,
    repo: Path,
    output: Path,
    weights_path: Path | None = None,
    source_commit: str = "HEAD",
) -> dict[str, object]:
    try:
        resolved_commit = subprocess.check_output(
            ["git", "rev-parse", f"{source_commit}^{{commit}}"],
            cwd=repo,
            text=True,
        ).strip()
    except Exception:
        resolved_commit = "uncommitted_local_build"

    payloads: dict[str, bytes] = {}
    for member in SOURCE_MEMBERS:
        file_path = repo / member
        if file_path.is_file():
            payloads[member] = file_path.read_bytes()
        else:
            raise FileNotFoundError(f"Source member missing: {member}")

    if weights_path and weights_path.is_file():
        weights_bytes = weights_path.read_bytes()
        target_name = f"models/research/pretrained/{weights_path.name}"
        payloads[target_name] = weights_bytes

    members = [
        {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    ]

    manifest = {
        "schema_version": "1.0.0",
        "experiment": "visual_dsp_ablation",
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
        "output_path": str(output),
        "source_commit": resolved_commit,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "member_count": len(payloads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Visual / DSP Ablation Code Package")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/local-artifacts/phase_visual_dsp_ablation_code.tar.gz"),
    )
    parser.add_argument("--weights-path", type=Path, default=None)
    args = parser.parse_args()

    result = build_package(
        repo=args.repo_root,
        output=args.output,
        weights_path=args.weights_path,
    )
    print(f"Package built successfully: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
