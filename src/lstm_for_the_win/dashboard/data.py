"""Artifact discovery and loading for the Streamlit application."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunBundle:
    """All lightweight artifacts required by the dashboard."""

    path: Path
    manifest: dict[str, Any]
    results: dict[str, Any]
    inference_predictions: list[dict[str, Any]]
    evaluation_predictions: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _read_inference_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["sentiment_confidence"] = float(row["sentiment_confidence"])
        row["topic_confidence"] = float(row["topic_confidence"])
        row["sentiment_correct"] = _as_bool(row["sentiment_correct"])
        row["topic_correct"] = _as_bool(row["topic_correct"])
    return rows


def _read_evaluation_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["confidence"] = float(row["confidence"])
        row["correct"] = _as_bool(row["correct"])
    return rows


def discover_runs(output_root: str | Path) -> list[Path]:
    """Return complete run directories from newest to oldest."""

    root = Path(output_root)
    if not root.is_dir():
        return []
    runs: list[tuple[str, Path]] = []
    for candidate in root.iterdir():
        manifest_path = candidate / "run_manifest.json"
        if not candidate.is_dir() or not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") == "complete":
            runs.append((str(manifest.get("created_at", "")), candidate))
    return [path for _, path in sorted(runs, reverse=True)]


def load_run(path: str | Path) -> RunBundle:
    """Load and validate one persisted pipeline run."""

    run_path = Path(path)
    required = {
        "run_manifest.json",
        "results.json",
        "inference_predictions.csv",
        "evaluation_predictions.csv",
    }
    missing = sorted(name for name in required if not (run_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Run {run_path} is missing: {', '.join(missing)}")
    return RunBundle(
        path=run_path,
        manifest=_read_json(run_path / "run_manifest.json"),
        results=_read_json(run_path / "results.json"),
        inference_predictions=_read_inference_predictions(
            run_path / "inference_predictions.csv"
        ),
        evaluation_predictions=_read_evaluation_predictions(
            run_path / "evaluation_predictions.csv"
        ),
    )


def filter_predictions(
    rows: Iterable[dict[str, Any]],
    *,
    sentiment: str | None = None,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    """Apply the dashboard's global sentiment and topic filters."""

    return [
        row
        for row in rows
        if (sentiment is None or row["sentiment"] == sentiment)
        and (topic is None or row["topic"] == topic)
    ]
