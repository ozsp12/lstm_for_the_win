"""Export static counterparts of the charts rendered on the project website."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FORMATS = ("png", "svg")
DPI = 220


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, output: Path, stem: str, items: list[dict[str, Any]], figure_type: str) -> None:
    fig.tight_layout()
    for extension in FORMATS:
        path = output / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)
        items.append({
            "file": path.name,
            "format": extension,
            "figure_type": figure_type,
            "task": "dashboard",
            "segment_dimension": "",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    plt.close(fig)


def _model_accuracy(rows: list[dict[str, str]], output: Path, items: list[dict[str, Any]]) -> None:
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["analysis_group"] == "aggregate_metrics" and row["metric"] == "accuracy":
            values[(row["task"], row["model"])] = float(row["value"])
    tasks = [task for task in ("sentiment", "topic") if (task, "lstm") in values]
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = list(range(len(tasks)))
    width = 0.36
    ax.bar([position - width / 2 for position in x], [values.get((task, "lstm"), 0.0) for task in tasks], width=width, label="LSTM")
    ax.bar([position + width / 2 for position in x], [values.get((task, "tfidf_logistic"), 0.0) for task in tasks], width=width, label="TF-IDF + Logistic")
    ax.set_xticks(x, [task.title() for task in tasks])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model comparison on the incoming batch")
    ax.legend()
    _save(fig, output, "dashboard_model_accuracy", items, "dashboard_model_accuracy")


def _sentiment_recall(rows: list[dict[str, str]], output: Path, items: list[dict[str, Any]]) -> None:
    data = [row for row in rows if row["analysis_group"] == "classwise_metrics" and row["task"] == "sentiment" and row["model"] == "lstm" and row["metric"] == "recall"]
    data.sort(key=lambda row: row["class_label"])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    labels = [row["class_label"].replace("_", " ") for row in data]
    ax.bar(range(len(data)), [float(row["value"]) for row in data])
    ax.set_xticks(range(len(data)), labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Sentiment performance by expected class")
    _save(fig, output, "dashboard_sentiment_class_recall", items, "dashboard_sentiment_class_recall")


def _topic_volume_accuracy(rows: list[dict[str, str]], output: Path, items: list[dict[str, Any]]) -> None:
    support: dict[str, int] = {}
    recall: dict[str, float] = {}
    for row in rows:
        if row["analysis_group"] != "classwise_metrics" or row["task"] != "topic" or row["model"] != "lstm":
            continue
        label = row["class_label"]
        if row["metric"] == "support":
            support[label] = int(float(row["value"]))
        elif row["metric"] == "recall":
            recall[label] = float(row["value"])
    labels = sorted(set(support) | set(recall))
    if not labels:
        return
    fig, ax_volume = plt.subplots(figsize=(8.0, 4.8))
    x = list(range(len(labels)))
    ax_volume.bar(x, [support.get(label, 0) for label in labels], label="Expected reviews")
    ax_volume.set_ylabel("Expected reviews")
    ax_volume.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=20, ha="right")
    ax_accuracy = ax_volume.twinx()
    ax_accuracy.plot(x, [recall.get(label, 0.0) for label in labels], marker="o", label="Accuracy / recall")
    ax_accuracy.set_ylim(0.0, 1.05)
    ax_accuracy.set_ylabel("Accuracy")
    ax_volume.set_title("Topic volume and accuracy")
    lines_1, labels_1 = ax_volume.get_legend_handles_labels()
    lines_2, labels_2 = ax_accuracy.get_legend_handles_labels()
    ax_volume.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")
    _save(fig, output, "dashboard_topic_volume_accuracy", items, "dashboard_topic_volume_accuracy")


def _confidence_accuracy(rows: list[dict[str, str]], output: Path, items: list[dict[str, Any]]) -> None:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["analysis_group"] == "confidence_accuracy" and row["model"] == "lstm":
            values[row["task"]][row["metric"]] = float(row["value"])
    tasks = [task for task in ("sentiment", "topic") if task in values]
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = list(range(len(tasks)))
    width = 0.36
    ax.bar([position - width / 2 for position in x], [values[task].get("accuracy", 0.0) for task in tasks], width=width, label="Accuracy")
    ax.bar([position + width / 2 for position in x], [values[task].get("mean_selected_class_confidence", 0.0) for task in tasks], width=width, label="Mean confidence")
    ax.set_xticks(x, [task.title() for task in tasks])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Value")
    ax.set_title("Average confidence versus accuracy")
    ax.legend()
    _save(fig, output, "dashboard_confidence_vs_accuracy", items, "dashboard_confidence_vs_accuracy")


def export_dashboard_figures(run_path: str | Path) -> Path:
    """Append exact static dashboard chart counterparts to ``figures_manifest.json``."""
    run = Path(run_path)
    analysis_path = run / "article_analysis.csv"
    figures_dir = run / "figures"
    figures_manifest_path = figures_dir / "figures_manifest.json"
    run_manifest_path = run / "run_manifest.json"
    if not figures_manifest_path.is_file():
        raise FileNotFoundError("Run the general figure exporter before dashboard figure export.")

    rows = _read(analysis_path)
    figures_manifest = json.loads(figures_manifest_path.read_text(encoding="utf-8"))
    items = list(figures_manifest.get("figure_files", []))
    _model_accuracy(rows, figures_dir, items)
    _sentiment_recall(rows, figures_dir, items)
    _topic_volume_accuracy(rows, figures_dir, items)
    _confidence_accuracy(rows, figures_dir, items)
    figures_manifest["figure_files"] = items
    figures_manifest_path.write_text(json.dumps(figures_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["figures"] = {
        "directory": "figures",
        "files": len(items),
        "logical_figures": len({item["file"].rsplit(".", 1)[0] for item in items}),
        "formats": list(FORMATS),
        "manifest_sha256": _sha256(figures_manifest_path),
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return figures_manifest_path
