#!/usr/bin/env python3
"""Build the deterministic code package for a development experiment.

Experiments: calibrated_late_fusion (Phase 4C.3A) and controlled_dsp_augmentation
(Phase 4C.6A); each has its own fixed member list.

Members are read from Git objects at one exact commit (not the working tree), so
the archive bytes are a pure function of that commit (plus optional weights).
The package stages into a directory that runs `python -m
ml.training.run_calibrated_late_fusion` and the analyzer without the repository.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from collections.abc import Sequence
from pathlib import Path

SOURCE_MEMBERS = (
    "ml/configs/calibrated_late_fusion_protocol.yaml",
    "ml/evaluation/__init__.py",
    "ml/evaluation/calibration.py",
    "ml/evaluation/metrics.py",
    "ml/requirements.txt",
    "ml/training/__init__.py",
    "ml/training/calibrated_late_fusion.py",
    "ml/training/dsp_features.py",
    "ml/training/loss.py",
    "ml/training/mobilenetv3_forensics.py",
    "ml/training/phase_4c2h_development.py",
    "ml/training/run_calibrated_late_fusion.py",
    "ml/training/visual_dsp_ablation.py",
    "research/evidence/visual_dsp_ablation/analysis_summary.json",
    "scripts/research/analyze_calibrated_late_fusion.py",
)
DSP_AUGMENTATION_MEMBERS = (
    "ml/configs/calibrated_late_fusion_protocol.yaml",
    "ml/configs/development_robustness_protocol.yaml",
    "ml/configs/dsp_augmentation_protocol.yaml",
    "ml/requirements.txt",
    "ml/training/__init__.py",
    "ml/training/calibrated_late_fusion.py",
    "ml/training/development_robustness.py",
    "ml/training/dsp_augmentation.py",
    "ml/training/dsp_features.py",
    "ml/training/loss.py",
    "ml/training/mobilenetv3_forensics.py",
    "ml/training/phase_4c2h_development.py",
    "ml/training/run_development_robustness.py",
    "ml/training/run_dsp_augmentation.py",
    "ml/training/visual_dsp_ablation.py",
    "scripts/research/analyze_calibrated_late_fusion.py",
    "scripts/research/analyze_development_robustness.py",
    "scripts/research/analyze_dsp_augmentation.py",
    "scripts/research/diagnose_fusion_shift.py",
)
EXPERIMENT_MEMBERS = {
    "calibrated_late_fusion": SOURCE_MEMBERS,
    "controlled_dsp_augmentation": DSP_AUGMENTATION_MEMBERS,
}
WEIGHTS_TARGET_DIR = "models/research/pretrained"


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


def resolve_commit(repo: Path, revision: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=repo, text=True
    ).strip()


def read_git_blob(repo: Path, commit: str, member: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{member}"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"{member} is not present in commit {commit}") from exc


def build_package(
    *,
    repo: Path,
    output: Path,
    source_commit: str = "HEAD",
    weights_path: Path | None = None,
    expected_weights_sha256: str | None = None,
    members: Sequence[str] | None = None,
    experiment: str = "calibrated_late_fusion",
) -> dict[str, object]:
    members = EXPERIMENT_MEMBERS[experiment] if members is None else members
    commit = resolve_commit(repo, source_commit)
    payloads = {member: read_git_blob(repo, commit, member) for member in members}

    if weights_path is not None:
        weights = weights_path.read_bytes()
        if expected_weights_sha256 and sha256_bytes(weights) != expected_weights_sha256:
            raise ValueError("Pretrained weights SHA-256 does not match the protocol binding")
        payloads[f"{WEIGHTS_TARGET_DIR}/{weights_path.name}"] = weights

    manifest = {
        "schema_version": "1.0.0",
        "experiment": experiment,
        "source_commit": commit,
        "members": [
            {"path": name, "bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in sorted(payloads.items())
        ],
    }
    payloads["SNAPSHOT_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.part")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for name, payload in sorted(payloads.items()):
            archive.addfile(normalized_info(name, len(payload)), io.BytesIO(payload))
    temporary.replace(output)

    return {
        "schema_version": "1.0.0",
        "experiment": experiment,
        "source_commit": commit,
        "archive_filename": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "member_count": len(payloads),
        "includes_weights": weights_path is not None,
        "snapshot_manifest_sha256": sha256_bytes(payloads["SNAPSHOT_MANIFEST.json"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic development-experiment code package")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_MEMBERS), default="calibrated_late_fusion")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-commit", default="HEAD")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/research/local-artifacts/phase_4c3_calibrated_late_fusion_code.tar.gz"),
    )
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument(
        "--expected-weights-sha256",
        default="047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
    )
    parser.add_argument("--receipt", type=Path, default=None, help="Optional path for a JSON receipt")
    args = parser.parse_args(argv)
    result = build_package(
        repo=args.repo_root,
        output=args.output,
        source_commit=args.source_commit,
        weights_path=args.weights_path,
        expected_weights_sha256=args.expected_weights_sha256,
        experiment=args.experiment,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
