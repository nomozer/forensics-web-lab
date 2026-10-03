#!/usr/bin/env python3
"""Build the deterministic, data-free manifest custodian package from Git blobs."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Sequence

try:
    from scripts.research.build_phase4c2g_execution_package import (
        PackageAudit,
        _read_git_blob,
        audit_git_commit_components,
        audit_package,
        sha256_bytes,
    )
except ModuleNotFoundError:  # Direct ``python scripts/research/<builder>.py`` entry.
    from build_phase4c2g_execution_package import (  # type: ignore[no-redef]
        PackageAudit,
        _read_git_blob,
        audit_git_commit_components,
        audit_package,
        sha256_bytes,
    )


CUSTODIAN_PACKAGE_MEMBERS = (
    "scripts/research/seal_locked_test_manifest.py",
    "scripts/research/RUN_PHASE4C2G_MANIFEST_CUSTODIAN_SESSION.ps1",
    "docs/schemas/human-manifest-custodian-authorization.v1.schema.json",
    "docs/schemas/locked-test-manifest.v1.schema.json",
    "research/evidence/phase-4b.2/split-lock.json",
)


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    return info


def build_custodian_package(
    *,
    repo_root: Path | str,
    output_path: Path | str,
    effective_custodian_commit: str,
    custodian_package_commit: str,
    source_binding_path: Path | str,
) -> PackageAudit:
    root = Path(repo_root).resolve(strict=True)
    output = Path(output_path).resolve(strict=False)
    if output.exists() or output.with_name(output.name + ".part").exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")
    binding = json.loads(Path(source_binding_path).read_text(encoding="utf-8"))
    if binding.get("effective_custodian_commit") != effective_custodian_commit:
        raise ValueError("Custodian source binding effective commit mismatch.")
    if binding.get("custodian_package_commit") != custodian_package_commit:
        raise ValueError("Custodian source binding package commit mismatch.")
    components = binding.get("components")
    if not isinstance(components, dict):
        raise ValueError("Custodian source binding components are absent.")
    audit_git_commit_components(
        repo_root=root,
        commit=effective_custodian_commit,
        components=components,
        require_worktree_match=True,
    )
    paths = {str(item["relative_path"]).replace("\\", "/") for item in components.values()}
    if paths != set(CUSTODIAN_PACKAGE_MEMBERS):
        raise ValueError("Custodian source binding/package allowlist mismatch.")
    binding_bytes = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    entries = [
        (name, _read_git_blob(root, effective_custodian_commit, name))
        for name in CUSTODIAN_PACKAGE_MEMBERS
    ]
    entries.append(("custodian_source_binding.json", binding_bytes))
    manifest = {
        "schema_version": "1.0.0",
        "effective_custodian_commit": effective_custodian_commit,
        "custodian_package_commit": custodian_package_commit,
        "contains_dataset": False,
        "contains_model": False,
        "contains_authorization": False,
        "members": {
            name: {"bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(entries)
        },
    }
    entries.append(
        (
            "CUSTODIAN_PACKAGE_MANIFEST.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    with part.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name, data in sorted(entries):
                    tar.addfile(_tar_info(name, len(data)), io.BytesIO(data))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(part, output)
    return audit_package(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-custodian-commit", required=True)
    parser.add_argument("--custodian-package-commit", required=True)
    parser.add_argument("--source-binding", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_custodian_package(
        repo_root=args.repo_root,
        output_path=args.output,
        effective_custodian_commit=args.effective_custodian_commit,
        custodian_package_commit=args.custodian_package_commit,
        source_binding_path=args.source_binding,
    )
    print(json.dumps({
        "archive_path": str(audit.archive_path),
        "sha256": audit.sha256,
        "bytes": audit.bytes,
        "member_count": audit.member_count,
        "members": audit.members,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
