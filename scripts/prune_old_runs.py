#!/usr/bin/env python3
"""Keep only the current versioned experiment directory under data/output."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def prune(output_root: Path, keep: str) -> list[str]:
    kept = output_root / keep
    if not kept.is_dir():
        raise FileNotFoundError(f"Run to keep does not exist: {kept}")

    removed: list[str] = []
    for child in sorted(output_root.iterdir()):
        if child.name in {keep, ".gitkeep", "latest.json"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
            removed.append(child.name)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all prior experiment directories and retain one run.")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument("--keep", required=True, help="Experiment directory name to retain.")
    args = parser.parse_args()

    removed = prune(Path(args.output_root), args.keep)
    print(json.dumps({"status": "ok", "kept": args.keep, "removed": removed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
