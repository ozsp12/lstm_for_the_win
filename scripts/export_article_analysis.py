#!/usr/bin/env python3
"""Export all manuscript/dashboard analyses for one pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lstm_for_the_win.analysis import export_article_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export versioned manuscript and dashboard analyses to article_analysis.csv."
    )
    parser.add_argument("run_path", help="Run directory containing metrics.json and predictions.csv.")
    parser.add_argument("--output", help="Optional output CSV path. Defaults to <run_path>/article_analysis.csv.")
    args = parser.parse_args()

    output = export_article_analysis(args.run_path, args.output)
    print(json.dumps({"status": "ok", "article_analysis": str(output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
