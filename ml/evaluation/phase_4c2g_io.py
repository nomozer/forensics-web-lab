"""Atomic output primitives for the sealed Phase 4C.2G evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    """Write JSON as `.part`, flush/fsync, then atomically publish with replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite sealed output: {target}")
    if part.exists():
        raise RuntimeError(
            f"Interrupted atomic output detected: {part}. Human adjudication required."
        )
    try:
        with part.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, target)
    except Exception:
        # Preserve any post-write partial file as crash evidence.
        raise
    return target
