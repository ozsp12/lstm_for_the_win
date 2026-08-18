"""Fail CI when package, pipeline, project, and citation versions diverge."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _match(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Could not read version from {path}")
    return match.group(1)


def main() -> int:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    package = _match(ROOT / "src/lstm_for_the_win/__init__.py", r'__version__\s*=\s*"([^"]+)"')
    pipeline = _match(ROOT / "src/lstm_for_the_win/experiment.py", r'PIPELINE_VERSION\s*=\s*"([^"]+)"')
    citation = _match(ROOT / "CITATION.cff", r"(?m)^version:\s*\"?([^\"\n]+)\"?$").strip()
    versions = {"pyproject": project, "package": package, "pipeline": pipeline, "citation": citation}
    if len(set(versions.values())) != 1:
        raise SystemExit(f"Version metadata diverged: {versions}")
    print(versions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
