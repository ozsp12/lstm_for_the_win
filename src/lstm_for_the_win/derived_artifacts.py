"""Generate deterministic human-readable artifacts exclusively from run.json."""

from __future__ import annotations

import csv
import json
import shutil
from argparse import ArgumentParser
from html import escape
from pathlib import Path
from typing import Any, Mapping

COMPARABLE_METRICS = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score")
CSV_FIELDS = [
    "run_id", "input_generation", "review_id", "text",
    "expected_sentiment", "predicted_sentiment", "sentiment_confidence", "sentiment_correct",
    "expected_topic", "predicted_topic", "topic_confidence", "topic_correct",
    "linguistic_level", "flagprofanity", "hasemoji", "hasspellingerror", "hasslang",
    "length_class", "mixed_sentiment", "goldtest", "template_family", "input_timestamp",
    *[f"sentiment_lstm_{metric}" for metric in COMPARABLE_METRICS],
    *[f"sentiment_baseline_{metric}" for metric in COMPARABLE_METRICS],
    *[f"sentiment_delta_{metric}" for metric in COMPARABLE_METRICS],
    "sentiment_accuracy_ci95_low", "sentiment_accuracy_ci95_high", "sentiment_mcnemar_p_value",
    *[f"topic_lstm_{metric}" for metric in COMPARABLE_METRICS],
    *[f"topic_baseline_{metric}" for metric in COMPARABLE_METRICS],
    *[f"topic_delta_{metric}" for metric in COMPARABLE_METRICS],
    "topic_accuracy_ci95_low", "topic_accuracy_ci95_high", "topic_mcnemar_p_value",
    "benchmark_sentiment_accuracy", "benchmark_topic_accuracy",
    "external_full_label_space_accuracy", "external_binary_restricted_accuracy",
    "external_neutral_prediction_rate", "external_binary_ci95_low", "external_binary_ci95_high",
]


def _accuracy_interval(task: Mapping[str, Any]) -> Mapping[str, Any]:
    return (
        task.get("uncertainty", {}).get("across_seed_ci95", {}).get("accuracy")
        or task.get("uncertainty", {}).get("primary_seed_accuracy_ci95", {})
        or task.get("uncertainty", {}).get("accuracy_ci95", {})
    )


def _wide_row(run: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "run_id": run["run"]["run_id"],
        "input_generation": run["run"]["input_generation"],
        "review_id": review.get("ID", ""),
        "text": review.get("text", ""),
        "expected_sentiment": review.get("expected_sentiment", ""),
        "predicted_sentiment": review.get("predicted_sentiment", ""),
        "sentiment_confidence": review.get("sentiment_confidence", ""),
        "sentiment_correct": review.get("sentiment_correct", ""),
        "expected_topic": review.get("expected_topic", ""),
        "predicted_topic": review.get("predicted_topic", ""),
        "topic_confidence": review.get("topic_confidence", ""),
        "topic_correct": review.get("topic_correct", ""),
        "linguistic_level": review.get("linguistic_level", ""),
        "flagprofanity": review.get("flagprofanity", ""),
        "hasemoji": review.get("hasemoji", ""),
        "hasspellingerror": review.get("hasspellingerror", ""),
        "hasslang": review.get("hasslang", ""),
        "length_class": review.get("length_class", ""),
        "mixed_sentiment": review.get("mixed_sentiment", ""),
        "goldtest": review.get("goldtest", ""),
        "template_family": review.get("template_family", ""),
        "input_timestamp": review.get("input_timestamp", ""),
    })
    for task_name in ("sentiment", "topic"):
        task = run["tasks"][task_name]
        for metric in COMPARABLE_METRICS:
            row[f"{task_name}_lstm_{metric}"] = task.get("metrics", {}).get(metric, "")
            row[f"{task_name}_baseline_{metric}"] = task.get("baseline_metrics", {}).get(metric, "")
            row[f"{task_name}_delta_{metric}"] = task.get("metric_delta_vs_baseline", {}).get(metric, "")
        interval = _accuracy_interval(task)
        row[f"{task_name}_accuracy_ci95_low"] = interval.get("low", "")
        row[f"{task_name}_accuracy_ci95_high"] = interval.get("high", "")
        row[f"{task_name}_mcnemar_p_value"] = task.get("paired_comparison", {}).get("p_value", "")
    benchmark = run.get("benchmark", {}).get("tasks", {})
    row["benchmark_sentiment_accuracy"] = benchmark.get("sentiment", {}).get("metrics", {}).get("accuracy", "")
    row["benchmark_topic_accuracy"] = benchmark.get("topic", {}).get("metrics", {}).get("accuracy", "")
    external = run.get("external_validation", {})
    row["external_full_label_space_accuracy"] = external.get("full_label_space_accuracy", external.get("accuracy", ""))
    row["external_binary_restricted_accuracy"] = external.get("binary_restricted_accuracy", "")
    row["external_neutral_prediction_rate"] = external.get("neutral_prediction_rate", "")
    binary_interval = external.get("uncertainty", {}).get("binary_restricted_accuracy_ci95", {})
    row["external_binary_ci95_low"] = binary_interval.get("low", "")
    row["external_binary_ci95_high"] = binary_interval.get("high", "")
    return row


def build_article_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build one ordinary wide tabular row per incoming review."""
    return [_wide_row(run, review) for review in run.get("reviews", [])]


def write_article_analysis(run: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_article_rows(run))
    return destination


def _svg_document(title: str, body: str, *, width: int = 960, height: int = 600) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="40" y="48" font-family="DejaVu Sans, Arial, sans-serif" font-size="24" font-weight="700">{escape(title)}</text>\n'
        f'{body}</svg>\n'
    )


def _bar_svg(title: str, labels: list[str], values: list[float], destination: Path) -> None:
    x0, y0, plot_w, plot_h = 230, 90, 670, 440
    parts = [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#333"/>']
    parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#333"/>')
    slot = plot_h / max(1, len(values))
    bar_h = min(44, slot * 0.62)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = y0 + index * slot + (slot - bar_h) / 2
        width = max(0.0, min(1.0, float(value))) * plot_w
        parts.append(f'<text x="{x0 - 12}" y="{y + bar_h * 0.7:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">{escape(label)}</text>')
        parts.append(f'<rect x="{x0}" y="{y:.1f}" width="{width:.2f}" height="{bar_h:.1f}" fill="#4c78a8"/>')
        parts.append(f'<text x="{min(x0 + width + 8, 905):.1f}" y="{y + bar_h * 0.7:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="14">{float(value):.3f}</text>')
    destination.write_text(_svg_document(title, "\n".join(parts)), encoding="utf-8")


def _heatmap_svg(title: str, row_labels: list[str], column_labels: list[str], matrix: list[list[int]], destination: Path) -> None:
    if not row_labels or not column_labels or not matrix:
        raise ValueError(f"Cannot render empty confusion matrix: {title}")
    if len(matrix) != len(row_labels) or any(len(row) != len(column_labels) for row in matrix):
        raise ValueError(f"Confusion matrix shape does not match labels: {title}")
    cell = min(110, int(420 / max(1, max(len(row_labels), len(column_labels)))))
    x0, y0 = 280, 110
    maximum = max(1, max(max(row) for row in matrix))
    parts: list[str] = []
    for j, label in enumerate(column_labels):
        parts.append(f'<text x="{x0 + j * cell + cell / 2:.1f}" y="{y0 - 18}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="14">{escape(label.replace("_", " "))}</text>')
    for i, expected in enumerate(row_labels):
        parts.append(f'<text x="{x0 - 18}" y="{y0 + i * cell + cell / 2 + 5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="14">{escape(expected.replace("_", " "))}</text>')
        for j, value in enumerate(matrix[i]):
            opacity = 0.12 + 0.78 * (value / maximum)
            parts.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell}" height="{cell}" fill="#4c78a8" fill-opacity="{opacity:.4f}" stroke="white"/>')
            parts.append(f'<text x="{x0 + j * cell + cell / 2:.1f}" y="{y0 + i * cell + cell / 2 + 6:.1f}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="16">{value}</text>')
    parts.append(f'<text x="{x0 + len(column_labels) * cell / 2:.1f}" y="{y0 + len(row_labels) * cell + 44}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">Predicted</text>')
    parts.append(f'<text x="55" y="{y0 + len(row_labels) * cell / 2:.1f}" transform="rotate(-90 55 {y0 + len(row_labels) * cell / 2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">Expected</text>')
    destination.write_text(_svg_document(title, "\n".join(parts)), encoding="utf-8")


def write_figures(run: Mapping[str, Any], directory: Path) -> Path:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sentiment = run["tasks"]["sentiment"]
    topic = run["tasks"]["topic"]
    _bar_svg(
        "Incoming accuracy: LSTM vs baseline",
        ["Sentiment LSTM", "Sentiment baseline", "Topic LSTM", "Topic baseline"],
        [sentiment["metrics"]["accuracy"], sentiment["baseline_metrics"]["accuracy"], topic["metrics"]["accuracy"], topic["baseline_metrics"]["accuracy"]],
        directory / "incoming_model_accuracy.svg",
    )
    _heatmap_svg("Incoming sentiment confusion matrix", sentiment["labels"], sentiment["labels"], sentiment["confusion_matrix"], directory / "incoming_sentiment_confusion.svg")
    _heatmap_svg("Incoming topic confusion matrix", topic["labels"], topic["labels"], topic["confusion_matrix"], directory / "incoming_topic_confusion.svg")
    benchmark = run.get("benchmark", {}).get("tasks", {})
    if benchmark:
        _bar_svg(
            "Immutable benchmark accuracy",
            ["Sentiment", "Topic"],
            [benchmark["sentiment"]["metrics"]["accuracy"], benchmark["topic"]["metrics"]["accuracy"]],
            directory / "benchmark_accuracy.svg",
        )
    external = run.get("external_validation", {})
    if external:
        confusion = external["confusion_matrix"]
        rows = list(confusion["expected_labels"])
        columns = list(confusion["predicted_labels"])
        mapping = confusion["matrix"]
        matrix = [[int(mapping[row][column]) for column in columns] for row in rows]
        _heatmap_svg("External UCI sentiment confusion matrix", rows, columns, matrix, directory / "external_sentiment_confusion.svg")
    return directory


def materialize_derived_artifacts(run_json: str | Path) -> tuple[Path, Path]:
    path = Path(run_json)
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema_version") != "2.0.0" or run.get("artifact_type") != "experiment_run":
        raise ValueError("Unsupported run.json contract.")
    csv_path = write_article_analysis(run, path.parent / "article_analysis.csv")
    figures_path = write_figures(run, path.parent / "figures")
    return csv_path, figures_path


def main() -> int:
    parser = ArgumentParser(description="Regenerate article_analysis.csv and figures exclusively from run.json.")
    parser.add_argument("run_json")
    arguments = parser.parse_args()
    csv_path, figures_path = materialize_derived_artifacts(arguments.run_json)
    print(json.dumps({"article_analysis_csv": str(csv_path), "figures": str(figures_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
