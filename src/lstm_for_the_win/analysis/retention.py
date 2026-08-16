"""Retention policy for versioned experiment output directories."""

from __future__ import annotations

import shutil
from pathlib import Path


def prune_output_runs(output_root: str | Path, keep: str) -> list[str]:
    """Delete every experiment directory except ``keep``."""
    root = Path(output_root)
    kept = root / keep
    if not kept.is_dir():
        raise FileNotFoundError(f"Run to keep does not exist: {kept}")

    removed: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name in {keep, ".gitkeep", "latest.json"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
            removed.append(child.name)
    return removed
