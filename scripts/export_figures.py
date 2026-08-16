#!/usr/bin/env python3
"""Export article and website figures for one pipeline run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lstm_for_the_win.analysis.dashboard_figures import export_dashboard_figures
from lstm_for_the_win.analysis.figures import export_figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Export all versioned manuscript/site figures for one run.")
    parser.add_argument("run_path", help="Run directory containing article_analysis.csv.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to <run_path>/figures.")
    args = parser.parse_args()

    manifest = export_figures(args.run_path, args.output_dir)
    if args.output_dir is None:
        manifest = export_dashboard_figures(args.run_path)
    print(json.dumps({"status": "ok", "figures_manifest": str(manifest.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
