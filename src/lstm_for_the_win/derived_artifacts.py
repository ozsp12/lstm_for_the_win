"""Generate deterministic human-readable artifacts exclusively from run.json."""

from __future__ import annotations

import csv
import json
import shutil
from argparse import ArgumentParser
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CSV_FIELDS = [
    "analysis_group", "run_id", "input_generation", "dataset", "task", "model",
    "metric", "dimension", "segment", "expected_label", "predicted_label", "seed",
    "review_id", "text", "expected_sentiment", "predicted_sentiment", "sentiment_confidence",
    "sentiment_correct", "expected_topic", "predicted_topic", "topic_confidence", "topic_correct",
    "linguistic_level", "flagprofanity", "hasemoji", "hasspellingerror", "hasslang",
    "length_class", "mixed_sentiment", "goldtest", "template_family",
    "value", "support", "low", "high",
]


def _blank(run: Mapping[str, Any], group: str, *, dataset: str = "incoming", task: str = "", model: str = "") -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        analysis_group=group,
        run_id=run["run"]["run_id"],
        input_generation=run["run"]["input_generation"],
        dataset=dataset,
        task=task,
        model=model,
    )
    return row


def _metric_rows(run: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for task, payload in run["tasks"].items():
        for model, key in (("lstm", "metrics"), ("tfidf_logistic_regression", "baseline_metrics")):
            for metric, value in sorted(payload[key].items()):
                row = _blank(run, "aggregate_metric", task=task, model=model)
                row.update(metric=metric, value=value, support=payload.get("incoming_size", ""))
                yield row
        for metric, value in sorted(payload.get("metric_delta_vs_baseline", {}).items()):
            row = _blank(run, "model_delta", task=task, model="lstm_minus_baseline")
            row.update(metric=metric, value=value, support=payload.get("incoming_size", ""))
            yield row
        ci = payload.get("uncertainty", {}).get("accuracy_ci95")
        if ci:
            row = _blank(run, "uncertainty", task=task, model="lstm")
            row.update(metric="accuracy", value=payload["metrics"]["accuracy"], support=ci.get("support", ""), low=ci.get("low", ""), high=ci.get("high", ""))
            yield row
        for model, group in (("lstm", "metrics"), ("tfidf_logistic_regression", "baseline_metrics")):
            for metric, stats in sorted(payload.get("replicates", {}).get(group, {}).items()):
                row = _blank(run, "replicate_summary", task=task, model=model)
                interval = stats.get("mean_ci95", {})
                row.update(metric=metric, value=stats.get("mean", ""), low=interval.get("low", ""), high=interval.get("high", ""), support=payload.get("replicates", {}).get("count", ""))
                yield row
        for dimension, segments in sorted(payload.get("segment_metrics", {}).items()):
            for segment, metrics in sorted(segments.items()):
                for metric, value in sorted(metrics.items()):
                    row = _blank(run, "segment_metric", task=task, model="lstm")
                    row.update(metric=metric, dimension=dimension, segment=segment, value=value)
                    yield row
        for metric, values in sorted(payload.get("history", {}).items()):
            for epoch, value in enumerate(values, start=1):
                row = _blank(run, "training_history", task=task, model="lstm")
                row.update(metric=metric, seed=payload.get("seed", ""), segment=epoch, value=value)
                yield row


def _confusion_rows(
    run: Mapping[str, Any],
    dataset: str,
    task: str,
    payload: Mapping[str, Any],
    labels: Sequence[str] | None = None,
) -> Iterable[dict[str, Any]]:
    resolved_labels = list(labels if labels is not None else payload["labels"])
    matrix = payload["confusion_matrix"]
    if len(matrix) != len(resolved_labels) or any(len(row) != len(resolved_labels) for row in matrix):
        raise ValueError(f"Confusion matrix shape does not match labels for {dataset}/{task}.")
    for i, expected in enumerate(resolved_labels):
        for j, predicted in enumerate(resolved_labels):
            row = _blank(run, "confusion_matrix", dataset=dataset, task=task, model="lstm")
            row.update(expected_label=expected, predicted_label=predicted, value=matrix[i][j])
            yield row


def _benchmark_rows(run: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    benchmark = run.get("benchmark") or {}
    for task, payload in benchmark.get("tasks", {}).items():
        for metric, value in sorted(payload.get("metrics", {}).items()):
            row = _blank(run, "benchmark_metric", dataset="synthetic_benchmark", task=task, model="lstm")
            row.update(metric=metric, value=value, support=payload.get("support", ""))
            yield row
        yield from _confusion_rows(run, "synthetic_benchmark", task, payload)

    external = run.get("external_validation") or {}
    if external:
        for metric, value in sorted(external.get("metrics", {}).items()):
            row = _blank(run, "external_metric", dataset="uci_amazon", task="sentiment", model="lstm")
            row.update(metric=metric, value=value, support=external.get("support", ""))
            yield row
        yield from _confusion_rows(
            run,
            "uci_amazon",
            "sentiment",
            external,
            labels=external.get("model_label_space", external.get("labels_in_source", [])),
        )


def _prediction_rows(run: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for review in run.get("reviews", []):
        row = _blank(run, "prediction_record", dataset="incoming")
        row.update(
            review_id=review.get("ID", ""), text=review.get("text", ""),
            expected_sentiment=review.get("expected_sentiment", ""), predicted_sentiment=review.get("predicted_sentiment", ""),
            sentiment_confidence=review.get("sentiment_confidence", ""), sentiment_correct=review.get("sentiment_correct", ""),
            expected_topic=review.get("expected_topic", ""), predicted_topic=review.get("predicted_topic", ""),
            topic_confidence=review.get("topic_confidence", ""), topic_correct=review.get("topic_correct", ""),
            linguistic_level=review.get("linguistic_level", ""), flagprofanity=review.get("flagprofanity", ""),
            hasemoji=review.get("hasemoji", ""), hasspellingerror=review.get("hasspellingerror", ""),
            hasslang=review.get("hasslang", ""), length_class=review.get("length_class", ""),
            mixed_sentiment=review.get("mixed_sentiment", ""), goldtest=review.get("goldtest", ""),
            template_family=review.get("template_family", ""),
        )
        yield row


def build_article_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(_metric_rows(run))
    for task, payload in run["tasks"].items():
        rows.extend(_confusion_rows(run, "incoming", task, payload))
    rows.extend(_benchmark_rows(run))
    rows.extend(_prediction_rows(run))
    return rows


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
    parts = [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+plot_h}" stroke="#333"/>']
    parts.append(f'<line x1="{x0}" y1="{y0+plot_h}" x2="{x0+plot_w}" y2="{y0+plot_h}" stroke="#333"/>')
    n = max(1, len(values))
    slot = plot_h / n
    bar_h = min(44, slot * 0.62)
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = y0 + index * slot + (slot - bar_h) / 2
        width = max(0.0, min(1.0, float(value))) * plot_w
        parts.append(f'<text x="{x0-12}" y="{y+bar_h*0.7:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">{escape(label)}</text>')
        parts.append(f'<rect x="{x0}" y="{y:.1f}" width="{width:.2f}" height="{bar_h:.1f}" fill="#4c78a8"/>')
        parts.append(f'<text x="{min(x0+width+8, 905):.1f}" y="{y+bar_h*0.7:.1f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="14">{float(value):.3f}</text>')
    for tick in range(0, 11, 2):
        x = x0 + plot_w * tick / 10
        parts.append(f'<text x="{x:.1f}" y="{y0+plot_h+28}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="13">{tick/10:.1f}</text>')
    destination.write_text(_svg_document(title, "\n".join(parts)), encoding="utf-8")


def _heatmap_svg(title: str, labels: list[str], matrix: list[list[int]], destination: Path) -> None:
    if not labels or not matrix:
        raise ValueError(f"Cannot render empty confusion matrix: {title}")
    n = len(labels)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"Confusion matrix shape does not match labels: {title}")
    cell = min(110, int(420 / max(1, n)))
    x0, y0 = 280, 110
    maximum = max(1, max(max(row) for row in matrix))
    parts: list[str] = []
    for j, label in enumerate(labels):
        parts.append(f'<text x="{x0+j*cell+cell/2:.1f}" y="{y0-18}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="14">{escape(label.replace("_", " "))}</text>')
    for i, expected in enumerate(labels):
        parts.append(f'<text x="{x0-18}" y="{y0+i*cell+cell/2+5:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="14">{escape(expected.replace("_", " "))}</text>')
        for j, value in enumerate(matrix[i]):
            opacity = 0.12 + 0.78 * (value / maximum)
            parts.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell}" height="{cell}" fill="#4c78a8" fill-opacity="{opacity:.4f}" stroke="white"/>')
            parts.append(f'<text x="{x0+j*cell+cell/2:.1f}" y="{y0+i*cell+cell/2+6:.1f}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="16">{value}</text>')
    parts.append(f'<text x="{x0 + n*cell/2:.1f}" y="{y0+n*cell+44}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">Predicted</text>')
    parts.append(f'<text x="55" y="{y0+n*cell/2:.1f}" transform="rotate(-90 55 {y0+n*cell/2:.1f})" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="15">Expected</text>')
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
    _heatmap_svg("Incoming sentiment confusion matrix", sentiment["labels"], sentiment["confusion_matrix"], directory / "incoming_sentiment_confusion.svg")
    _heatmap_svg("Incoming topic confusion matrix", topic["labels"], topic["confusion_matrix"], directory / "incoming_topic_confusion.svg")
    benchmark = run.get("benchmark", {}).get("tasks", {})
    if benchmark:
        _bar_svg(
            "Immutable benchmark accuracy",
            ["Sentiment", "Topic"],
            [benchmark["sentiment"]["metrics"]["accuracy"], benchmark["topic"]["metrics"]["accuracy"]],
            directory / "benchmark_accuracy.svg",
        )
    external = run.get("external_validation")
    if external:
        labels = list(external.get("model_label_space", external.get("labels_in_source", [])))
        _heatmap_svg("External UCI Amazon sentiment confusion matrix", labels, external["confusion_matrix"], directory / "external_sentiment_confusion.svg")
    return directory


def materialize_derived_artifacts(run_json: str | Path) -> tuple[Path, Path]:
    run_path = Path(run_json)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("artifact_type") != "experiment_run":
        raise ValueError("Expected an experiment_run run.json artifact.")
    csv_path = write_article_analysis(run, run_path.parent / "article_analysis.csv")
    figures_path = write_figures(run, run_path.parent / "figures")
    return csv_path, figures_path


def main() -> int:
    parser = ArgumentParser(description="Regenerate article_analysis.csv and figures from run.json.")
    parser.add_argument("run_json")
    arguments = parser.parse_args()
    csv_path, figures_path = materialize_derived_artifacts(arguments.run_json)
    print(json.dumps({"article_analysis": str(csv_path), "figures": str(figures_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
