#!/usr/bin/env python3
"""Keep only the current versioned experiment directory under data/output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lstm_for_the_win.analysis.retention import prune_output_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete all prior experiment directories and retain one run.")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument("--keep", required=True, help="Experiment directory name to retain.")
    args = parser.parse_args()

    removed = prune_output_runs(args.output_root, args.keep)
    print(json.dumps({"status": "ok", "kept": args.keep, "removed": removed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
