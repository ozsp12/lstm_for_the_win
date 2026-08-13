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


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_evaluation_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["confidence"] = float(row["confidence"])
        row["correct"] = _as_bool(row["correct"])
    return rows


def _suggested_action(sentiment: str, topic: str) -> str:
    team = topic.replace("_", " ").title()
    if sentiment == "negative":
        return f"Prioritize and route to the {team} support team."
    if sentiment == "positive":
        return f"Route to the {team} insights queue for advocacy analysis."
    return f"Add to the {team} monitoring queue."


def _enrich_predictions(
    predictions: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join detailed model probabilities without expanding predictions.csv."""

    details = {(row["ID"], row["task"]): row for row in evaluation}
    enriched: list[dict[str, Any]] = []
    for prediction in predictions:
        sentiment = details[(prediction["ID"], "sentiment")]
        topic = details[(prediction["ID"], "topic")]
        enriched.append(
            {
                **prediction,
                "sentiment": prediction["predicted_sentiment"],
                "topic": prediction["predicted_topic"],
                "sentiment_confidence": sentiment["confidence"],
                "topic_confidence": topic["confidence"],
                "sentiment_correct": sentiment["correct"],
                "topic_correct": topic["correct"],
                "suggested_action": _suggested_action(
                    prediction["predicted_sentiment"],
                    prediction["predicted_topic"],
                ),
            }
        )
    return enriched


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
        "predictions.csv",
        "evaluation_predictions.csv",
    }
    missing = sorted(name for name in required if not (run_path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Run {run_path} is missing: {', '.join(missing)}")
    evaluation = _read_evaluation_predictions(run_path / "evaluation_predictions.csv")
    predictions = _read_predictions(run_path / "predictions.csv")
    return RunBundle(
        path=run_path,
        manifest=_read_json(run_path / "run_manifest.json"),
        results=_read_json(run_path / "results.json"),
        inference_predictions=_enrich_predictions(predictions, evaluation),
        evaluation_predictions=evaluation,
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
