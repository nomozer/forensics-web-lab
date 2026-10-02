#!/usr/bin/env python3
"""Build and audit the complete external Phase 4C.2G execution package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


PACKAGE_SOURCE_MEMBERS = (
    "ml/requirements.txt",
    "ml/evaluation/__init__.py",
    "ml/evaluation/calibration.py",
    "ml/evaluation/confirmatory_metrics.py",
    "ml/evaluation/locked_test_evaluator.py",
    "ml/evaluation/metrics.py",
    "ml/evaluation/phase_4c2g_dataset.py",
    "ml/evaluation/phase_4c2g_io.py",
    "ml/evaluation/phase_4c2g_model.py",
    "ml/evaluation/phase_4c2g_windows.py",
    "ml/evaluation/run_phase_4c2g_confirmatory.py",
    "ml/training/__init__.py",
    "ml/training/loss.py",
    "ml/training/mobilenetv3_forensics.py",
    "docs/schemas/human-unsealing-authorization.v2.schema.json",
    "docs/schemas/locked-test-manifest.v1.schema.json",
    "scripts/research/RUN_PHASE4C2G_AUTHORIZED_SESSION.ps1",
    "research/evidence/phase-4c.2e/LOCKED_TEST_PREREGISTRATION.json",
    "research/evidence/phase-4c.2e/candidate_checkpoint_binding.json",
    "research/evidence/phase-4c.2e/confirmatory_metrics_plan.json",
    "research/evidence/phase-4c.2e/unsealing_protocol.json",
    "research/evidence/phase-4b.2/split-lock.json",
)

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".npy",
    ".npz",
}
FORBIDDEN_NAME_PATTERN = re.compile(
    r"(^|/)(human_unsealing_authorization\.json|\.env|credentials?[^/]*)$",
    re.IGNORECASE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_name(name: str) -> None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "" in pure.parts:
        raise ValueError(f"Unsafe package member path: {name!r}")
    if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Dataset/model binary forbidden in package: {name}")
    if FORBIDDEN_NAME_PATTERN.search(name):
        raise ValueError(f"Authorization/credential artifact forbidden in package: {name}")


@dataclass(frozen=True)
class PackageAudit:
    archive_path: Path
    bytes: int
    sha256: str
    member_count: int
    members: tuple[str, ...]


def audit_package(archive_path: Path | str) -> PackageAudit:
    archive = Path(archive_path).resolve(strict=True)
    names: list[str] = []
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            _safe_member_name(member.name)
            if member.name in seen:
                raise ValueError(f"Duplicate TAR member: {member.name}")
            if not member.isfile():
                raise ValueError(
                    f"Non-regular TAR member forbidden: {member.name} type={member.type!r}"
                )
            seen.add(member.name)
            names.append(member.name)
    return PackageAudit(
        archive_path=archive,
        bytes=archive.stat().st_size,
        sha256=sha256_file(archive),
        member_count=len(names),
        members=tuple(names),
    )


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def build_package(
    *,
    repo_root: Path | str,
    output_path: Path | str,
    effective_execution_commit: str,
    execution_package_commit: str,
    source_binding_path: Path | str,
) -> PackageAudit:
    root = Path(repo_root).resolve(strict=True)
    output = Path(output_path).resolve(strict=False)
    if output.exists() or output.with_name(output.name + ".part").exists():
        raise FileExistsError(f"Refusing to overwrite package or partial package: {output}")
    binding_path = Path(source_binding_path).resolve(strict=True)
    binding_bytes = binding_path.read_bytes()
    source_binding = json.loads(binding_bytes.decode("utf-8"))
    if source_binding.get("effective_evaluator_commit") != effective_execution_commit:
        raise ValueError("Source binding effective commit does not match builder input.")

    source_entries: list[tuple[str, bytes]] = []
    for name in PACKAGE_SOURCE_MEMBERS:
        _safe_member_name(name)
        path = (root / name).resolve(strict=True)
        path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Package source must be a regular non-symlink file: {name}")
        source_entries.append((name, path.read_bytes()))
    source_entries.append(("runtime_source_binding.json", binding_bytes))

    provenance = {
        "schema_version": "1.0.0",
        "effective_execution_commit": effective_execution_commit,
        "execution_package_commit": execution_package_commit,
        "source_binding_sha256": sha256_file(binding_path),
        "source_binding": {
            key: value
            for key, value in source_binding.items()
            if key not in {"sealed_package_sha256", "sealed_package_bytes"}
        },
    }
    provenance_bytes = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    source_entries.append(("source_provenance_manifest.json", provenance_bytes))

    member_manifest = {
        name: {"bytes": len(data), "sha256": sha256_bytes(data)}
        for name, data in sorted(source_entries)
    }
    package_manifest = {
        "schema_version": "1.0.0",
        "effective_execution_commit": effective_execution_commit,
        "execution_package_commit": execution_package_commit,
        "contains_dataset": False,
        "contains_checkpoints": False,
        "contains_authorization_artifact": False,
        "members": member_manifest,
    }
    manifest_bytes = (
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    source_entries.append(("PACKAGE_MANIFEST.json", manifest_bytes))

    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    try:
        with part.open("xb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for name, data in sorted(source_entries):
                        _safe_member_name(name)
                        tar.addfile(_tar_info(name, len(data)), io.BytesIO(data))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(part, output)
    except Exception:
        raise
    return audit_package(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-execution-commit", required=True)
    parser.add_argument("--execution-package-commit", required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_package(
        repo_root=args.repo_root,
        output_path=args.output,
        effective_execution_commit=args.effective_execution_commit,
        execution_package_commit=args.execution_package_commit,
        source_binding_path=args.source_binding,
    )
    print(
        json.dumps(
            {
                "archive_path": str(audit.archive_path),
                "bytes": audit.bytes,
                "sha256": audit.sha256,
                "member_count": audit.member_count,
                "members": audit.members,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
