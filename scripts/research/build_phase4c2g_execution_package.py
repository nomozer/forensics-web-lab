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
import subprocess
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


def _read_git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Package source is absent at effective commit {commit}: {relative_path}"
        )
    return completed.stdout


def audit_git_commit_components(
    *,
    repo_root: Path | str,
    commit: str,
    components: dict[str, dict[str, Any]],
    require_worktree_match: bool = False,
) -> dict[str, dict[str, Any]]:
    """Verify component byte counts and SHA-256 against exact Git blob bytes."""
    root = Path(repo_root).resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Default commit must be an exact 40-hex object id: {commit!r}")
    audit: dict[str, dict[str, Any]] = {}
    for name, metadata in components.items():
        relative_path = str(metadata["relative_path"]).replace("\\", "/")
        component_commit = str(metadata.get("git_commit", commit))
        if not re.fullmatch(r"[0-9a-f]{40}", component_commit):
            raise ValueError(
                f"Component {name!r} git_commit must be exact 40-hex: "
                f"{component_commit!r}"
            )
        _safe_member_name(relative_path)
        try:
            payload = _read_git_blob(root, component_commit, relative_path)
        except ValueError as exc:
            raise ValueError(
                f"Runtime component {name!r} is absent at bound commit: "
                f"{relative_path}"
            ) from exc
        actual_sha256 = sha256_bytes(payload)
        bytes_match = len(payload) == int(metadata["bytes"])
        sha256_match = actual_sha256 == str(metadata["sha256"])
        if not bytes_match or not sha256_match:
            raise ValueError(
                f"Runtime component {name!r} differs at bound commit: "
                f"bytes={len(payload)}, sha256={actual_sha256}"
            )
        if require_worktree_match:
            worktree_check = subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    "--no-ext-diff",
                    component_commit,
                    "--",
                    relative_path,
                ],
                cwd=root,
                check=False,
            )
            if worktree_check.returncode == 1:
                raise ValueError(
                    f"Runtime component {name!r} changed after effective commit: "
                    f"{relative_path}"
                )
            if worktree_check.returncode != 0:
                raise RuntimeError(
                    f"Git worktree comparison failed for runtime component {name!r}."
                )
        audit[name] = {
            "relative_path": relative_path,
            "git_commit": component_commit,
            "exists_at_commit": True,
            "bytes": len(payload),
            "sha256": actual_sha256,
            "bytes_match": True,
            "sha256_match": True,
        }
    return audit


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
    source_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_bytes = (
        json.dumps(source_binding, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    evaluator_commit = str(source_binding.get("effective_evaluator_commit", ""))
    bound_execution_commit = str(
        source_binding.get("effective_execution_commit", evaluator_commit)
    )
    if bound_execution_commit != effective_execution_commit:
        raise ValueError(
            "Source binding effective execution commit does not match builder input."
        )
    if source_binding.get("execution_package_commit") != execution_package_commit:
        raise ValueError(
            "Source binding execution package commit does not match builder input."
        )

    commit_components = dict(source_binding.get("components", {}))
    commit_components["authorization_schema"] = source_binding.get(
        "authorization_schema", {}
    )
    audit_git_commit_components(
        repo_root=root,
        commit=evaluator_commit,
        components=commit_components,
        require_worktree_match=False,
    )
    bound_paths = {
        str(metadata["relative_path"]).replace("\\", "/")
        for metadata in commit_components.values()
    }
    if bound_paths != set(PACKAGE_SOURCE_MEMBERS):
        missing = sorted(set(PACKAGE_SOURCE_MEMBERS) - bound_paths)
        extra = sorted(bound_paths - set(PACKAGE_SOURCE_MEMBERS))
        raise ValueError(
            f"Package/source-binding member mismatch: missing={missing}, extra={extra}"
        )

    component_by_path = {
        str(metadata["relative_path"]).replace("\\", "/"): metadata
        for metadata in commit_components.values()
    }
    if evaluator_commit != effective_execution_commit:
        missing_commits = sorted(
            path
            for path, metadata in component_by_path.items()
            if "git_commit" not in metadata
        )
        if missing_commits:
            raise ValueError(
                "Mixed evaluator/execution package requires explicit git_commit "
                f"on every component: {missing_commits}"
            )

    source_entries: list[tuple[str, bytes]] = []
    for name in PACKAGE_SOURCE_MEMBERS:
        _safe_member_name(name)
        metadata = component_by_path[name]
        component_commit = str(metadata.get("git_commit", evaluator_commit))
        source_entries.append(
            (name, _read_git_blob(root, component_commit, name))
        )
    source_entries.append(("runtime_source_binding.json", binding_bytes))

    provenance = {
        "schema_version": "1.0.0",
        "evaluator_effective_commit": evaluator_commit,
        "effective_execution_commit": effective_execution_commit,
        "execution_package_commit": execution_package_commit,
        "source_binding_sha256": sha256_bytes(binding_bytes),
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
        "evaluator_effective_commit": evaluator_commit,
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
