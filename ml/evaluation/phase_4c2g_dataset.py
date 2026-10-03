"""Fail-closed locked-test manifest and dataset contract for Phase 4C.2G.

The module contains no default machine path and performs no work at import time.
It is invoked only inside a future authorized, isolated, read-only session.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Sequence

import jsonschema


EXPECTED_SOURCE_COUNT = 343
EXPECTED_SAMPLE_COUNT = 686
LABEL_MAP = {"authentic": 0, "ai_edited": 1}
CANONICAL_LOCKED_SPLIT_SEAL = (
    "519e7a0e6815e781d1cefa95971e5221ac1f25656837374d8dc4ba41401fded9"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LockedTestSample:
    sample_id: str
    unique_source_id: str
    relative_path: str
    absolute_path: Path
    sha256: str
    label: str
    label_id: int


@dataclass(frozen=True)
class LockedTestManifest:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    schema_sha256: str
    split_seal: str
    samples: tuple[LockedTestSample, ...]
    source_ids: tuple[str, ...]
    label_map: dict[str, int]

    def iter_samples(self) -> Iterator[LockedTestSample]:
        return iter(self.samples)


def validate_manifest_relative_path(relative_path: str) -> PurePosixPath:
    """Return a canonical POSIX relative path or reject cross-platform escapes."""
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or posix == PurePosixPath(".")
    ):
        raise ValueError(f"Unsafe manifest relative_path: {relative_path!r}")
    return posix


def _resolve_member(root: Path, relative_path: str) -> Path:
    rel = validate_manifest_relative_path(relative_path)
    candidate = (root / rel).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path escapes locked-test root: {relative_path!r}"
        ) from exc
    return candidate


def _source_split_seal(source_ids: Sequence[str]) -> str:
    payload = json.dumps(sorted(source_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_locked_test_manifest(
    *,
    locked_test_root: Path | str,
    manifest_path: Path | str,
    manifest_schema_path: Path | str,
    expected_manifest_sha256: str,
    expected_schema_sha256: str,
    expected_split_seal: str = CANONICAL_LOCKED_SPLIT_SEAL,
) -> LockedTestManifest:
    """Validate and bind the complete locked-test inventory without writing to it."""
    root = Path(locked_test_root).resolve(strict=True)
    manifest_file = Path(manifest_path).resolve(strict=True)
    schema_file = Path(manifest_schema_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Locked-test root is not a directory: {root}")
    if root.is_symlink() or manifest_file.is_symlink() or schema_file.is_symlink():
        raise ValueError("Symlinks are forbidden for locked-test contract inputs.")

    actual_schema_sha = sha256_file(schema_file)
    if actual_schema_sha != expected_schema_sha256:
        raise ValueError(
            "Locked-test manifest schema SHA-256 mismatch: "
            f"expected {expected_schema_sha256}, got {actual_schema_sha}"
        )
    actual_manifest_sha = sha256_file(manifest_file)
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError(
            "Locked-test manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha}"
        )

    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(manifest_file.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(manifest)

    rows = manifest["samples"]
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Locked-test sample count {len(rows)} != {EXPECTED_SAMPLE_COUNT}"
        )

    seen_sample_ids: set[str] = set()
    seen_relative_paths: set[str] = set()
    seen_source_label_pairs: set[tuple[str, int]] = set()
    labels_by_source: dict[str, set[int]] = defaultdict(set)
    loaded: list[LockedTestSample] = []
    expected_files: set[Path] = set()

    for row in rows:
        sample_id = str(row["sample_id"])
        relative_path = str(row["relative_path"])
        label = str(row["label"])
        label_id = int(row["label_id"])
        source_id = str(row["unique_source_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if relative_path in seen_relative_paths:
            raise ValueError(f"Duplicate relative_path: {relative_path}")
        if LABEL_MAP.get(label) != label_id:
            raise ValueError(
                f"Label mapping mismatch for {sample_id}: {label!r} != {label_id}"
            )
        source_label_pair = (source_id, label_id)
        if source_label_pair in seen_source_label_pairs:
            raise ValueError(
                f"Duplicate source/label pair: source={source_id!r}, label_id={label_id}"
            )

        sample_path = _resolve_member(root, relative_path)
        if not sample_path.is_file() or sample_path.is_symlink():
            raise FileNotFoundError(f"Missing or non-regular sample: {relative_path}")
        actual_sha = sha256_file(sample_path)
        if actual_sha != row["sha256"]:
            raise ValueError(
                f"Sample SHA-256 mismatch for {sample_id}: "
                f"expected {row['sha256']}, got {actual_sha}"
            )

        seen_sample_ids.add(sample_id)
        seen_relative_paths.add(relative_path)
        seen_source_label_pairs.add(source_label_pair)
        labels_by_source[source_id].add(label_id)
        expected_files.add(sample_path)
        loaded.append(
            LockedTestSample(
                sample_id=sample_id,
                unique_source_id=source_id,
                relative_path=relative_path,
                absolute_path=sample_path,
                sha256=actual_sha,
                label=label,
                label_id=label_id,
            )
        )

    if len(labels_by_source) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"Locked-test source count {len(labels_by_source)} != {EXPECTED_SOURCE_COUNT}"
        )
    invalid_pairs = {
        source_id: sorted(labels)
        for source_id, labels in labels_by_source.items()
        if labels != {0, 1}
    }
    if invalid_pairs:
        raise ValueError(f"Paired-label integrity violation: {invalid_pairs}")

    source_ids = tuple(sorted(labels_by_source))
    actual_split_seal = _source_split_seal(source_ids)
    if actual_split_seal != expected_split_seal:
        raise ValueError(
            "Locked-test source split seal mismatch: "
            f"expected {expected_split_seal}, got {actual_split_seal}"
        )

    actual_files = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
    }
    try:
        manifest_file.relative_to(root)
    except ValueError:
        pass
    else:
        actual_files.discard(manifest_file)
    missing = expected_files - actual_files
    extra = actual_files - expected_files
    if missing or extra:
        raise ValueError(
            "Locked-test file inventory mismatch: "
            f"missing={sorted(str(p.relative_to(root)) for p in missing)}, "
            f"extra={sorted(str(p.relative_to(root)) for p in extra)}"
        )

    loaded.sort(key=lambda sample: (sample.unique_source_id, sample.label_id))
    return LockedTestManifest(
        root=root,
        manifest_path=manifest_file,
        manifest_sha256=actual_manifest_sha,
        schema_sha256=actual_schema_sha,
        split_seal=actual_split_seal,
        samples=tuple(loaded),
        source_ids=source_ids,
        label_map=dict(LABEL_MAP),
    )
