"""Generate manuscript and website figures from the versioned article-analysis CSV."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_FORMATS = ("png", "svg")
RASTER_DPI = 220


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _float(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def _save(fig: plt.Figure, stem: Path, manifest: list[dict[str, Any]], *, figure_type: str, task: str = "", dimension: str = "") -> None:
    fig.tight_layout()
    for extension in IMAGE_FORMATS:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = RASTER_DPI
        fig.savefig(path, **kwargs)
        manifest.append({
            "file": path.name,
            "format": extension,
            "figure_type": figure_type,
            "task": task,
            "segment_dimension": dimension,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    plt.close(fig)


def _rows(rows: Iterable[dict[str, str]], *, group: str, task: str = "", model: str = "") -> list[dict[str, str]]:
    result = [row for row in rows if row["analysis_group"] == group]
    if task:
        result = [row for row in result if row["task"] == task]
    if model:
        result = [row for row in result if row["model"] == model]
    return result


def _plot_model_performance(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    metrics = ("accuracy", "macro_f1", "weighted_f1")
    labels = {"lstm": "LSTM", "tfidf_logistic": "TF-IDF + Logistic"}
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _rows(rows, group="aggregate_metrics", task=task):
        if row["metric"] in metrics:
            by_model[row["model"]][row["metric"]] = _float(row["value"])
    if not by_model:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = list(range(len(metrics)))
    models = [model for model in ("lstm", "tfidf_logistic") if model in by_model]
    width = 0.8 / max(1, len(models))
    for index, model in enumerate(models):
        offsets = [position - 0.4 + width / 2 + index * width for position in x]
        ax.bar(offsets, [by_model[model].get(metric, 0.0) for metric in metrics], width=width, label=labels.get(model, model))
    ax.set_xticks(x, [metric.replace("_", " ").title() for metric in metrics])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{task.title()} classification performance")
    ax.legend()
    _save(fig, out / f"model_performance_{task}", manifest, figure_type="model_performance", task=task)


def _plot_probabilistic_metrics(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    metrics = ("log_loss", "brier_score", "expected_calibration_error")
    labels = {"lstm": "LSTM", "tfidf_logistic": "TF-IDF + Logistic"}
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _rows(rows, group="aggregate_metrics", task=task):
        if row["metric"] in metrics:
            by_model[row["model"]][row["metric"]] = _float(row["value"])
    if not by_model:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = list(range(len(metrics)))
    models = [model for model in ("lstm", "tfidf_logistic") if model in by_model]
    width = 0.8 / max(1, len(models))
    for index, model in enumerate(models):
        offsets = [position - 0.4 + width / 2 + index * width for position in x]
        ax.bar(offsets, [by_model[model].get(metric, 0.0) for metric in metrics], width=width, label=labels.get(model, model))
    ax.set_xticks(x, [metric.replace("_", " ").title() for metric in metrics])
    ax.set_ylabel("Value (lower is better)")
    ax.set_title(f"{task.title()} probabilistic metrics")
    ax.legend()
    _save(fig, out / f"probabilistic_metrics_{task}", manifest, figure_type="probabilistic_metrics", task=task)


def _plot_confusion_matrix(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = _rows(rows, group="confusion_matrix", task=task, model="lstm")
    if not data:
        return
    labels = sorted({row["expected_label"] for row in data} | {row["predicted_label"] for row in data})
    lookup = {(row["expected_label"], row["predicted_label"]): int(float(row["value"])) for row in data}
    matrix = [[lookup.get((expected, predicted), 0) for predicted in labels] for expected in labels]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), [label.replace("_", " ") for label in labels], rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Expected label")
    ax.set_title(f"{task.title()} confusion matrix")
    for row_index, expected in enumerate(labels):
        for column_index, predicted in enumerate(labels):
            ax.text(column_index, row_index, str(lookup.get((expected, predicted), 0)), ha="center", va="center")
    _save(fig, out / f"confusion_matrix_{task}", manifest, figure_type="confusion_matrix", task=task)


def _plot_classwise(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = _rows(rows, group="classwise_metrics", task=task, model="lstm")
    metrics = ("precision", "recall", "specificity", "f1")
    labels = sorted({row["class_label"] for row in data})
    if not labels:
        return
    values = {(row["class_label"], row["metric"]): _float(row["value"]) for row in data}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = list(range(len(labels)))
    width = 0.8 / len(metrics)
    for index, metric in enumerate(metrics):
        offsets = [position - 0.4 + width / 2 + index * width for position in x]
        ax.bar(offsets, [values.get((label, metric), 0.0) for label in labels], width=width, label=metric.title())
    ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{task.title()} classwise metrics")
    ax.legend(ncol=2)
    _save(fig, out / f"classwise_metrics_{task}", manifest, figure_type="classwise_metrics", task=task)


def _plot_class_distribution(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = [row for row in _rows(rows, group="class_distribution", task=task, model="lstm") if row["metric"] in {"expected_proportion", "predicted_proportion"}]
    labels = sorted({row["class_label"] for row in data})
    if not labels:
        return
    lookup = {(row["class_label"], row["metric"]): _float(row["value"]) for row in data}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([position - width / 2 for position in x], [lookup.get((label, "expected_proportion"), 0.0) for label in labels], width=width, label="Expected")
    ax.bar([position + width / 2 for position in x], [lookup.get((label, "predicted_proportion"), 0.0) for label in labels], width=width, label="Predicted")
    ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=20, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(f"{task.title()} expected and predicted distributions")
    ax.legend()
    _save(fig, out / f"class_distribution_{task}", manifest, figure_type="class_distribution", task=task)


def _plot_calibration(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = _rows(rows, group="calibration_bins", task=task, model="lstm")
    bins: dict[str, dict[str, float]] = defaultdict(dict)
    for row in data:
        bins[row["segment_value"]][row["metric"]] = _float(row["value"])
    points = [(values.get("mean_confidence", 0.0), values.get("accuracy", 0.0)) for values in bins.values()]
    points.sort()
    if not points:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot([point[0] for point in points], [point[1] for point in points], marker="o", label="LSTM")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{task.title()} calibration")
    ax.legend()
    _save(fig, out / f"calibration_{task}", manifest, figure_type="calibration", task=task)


def _plot_training_history(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = _rows(rows, group="training_history", task=task, model="lstm")
    history: dict[str, dict[int, float]] = defaultdict(dict)
    for row in data:
        history[row["metric"]][int(row["segment_value"])] = _float(row["value"])
    if not history:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for metric in ("accuracy", "val_accuracy"):
        if metric in history:
            epochs = sorted(history[metric])
            ax.plot(epochs, [history[metric][epoch] for epoch in epochs], marker="o", label=metric.replace("_", " ").title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{task.title()} training history")
    ax.legend()
    _save(fig, out / f"training_history_{task}", manifest, figure_type="training_history", task=task)


def _plot_segments(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = [row for row in _rows(rows, group="segment_metrics", task=task, model="lstm") if row["metric"] in {"accuracy", "macro_f1"}]
    for dimension in sorted({row["segment_dimension"] for row in data}):
        selected = [row for row in data if row["segment_dimension"] == dimension]
        values = sorted({row["segment_value"] for row in selected})
        lookup = {(row["segment_value"], row["metric"]): _float(row["value"]) for row in selected}
        fig, ax = plt.subplots(figsize=(max(6.4, len(values) * 0.9), 4.6))
        x = list(range(len(values)))
        width = 0.36
        ax.bar([position - width / 2 for position in x], [lookup.get((value, "accuracy"), 0.0) for value in values], width=width, label="Accuracy")
        ax.bar([position + width / 2 for position in x], [lookup.get((value, "macro_f1"), 0.0) for value in values], width=width, label="Macro F1")
        ax.set_xticks(x, [value.replace("_", " ") for value in values], rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{task.title()} robustness by {dimension.replace('_', ' ')}")
        ax.legend()
        _save(fig, out / f"segment_{_safe_name(dimension)}_{task}", manifest, figure_type="segment_robustness", task=task, dimension=dimension)


def _plot_error_confidence(rows: list[dict[str, str]], out: Path, task: str, manifest: list[dict[str, Any]]) -> None:
    data = _rows(rows, group="error_record", task=task, model="lstm")
    values = [_float(row["value"]) for row in data if row["metric"] == "confidence"]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bins = min(12, max(4, len(values)))
    ax.hist(values, bins=bins)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Confidence assigned to wrong prediction")
    ax.set_ylabel("Errors")
    ax.set_title(f"{task.title()} error-confidence distribution")
    _save(fig, out / f"error_confidence_{task}", manifest, figure_type="error_confidence", task=task)


def export_figures(run_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Export all manuscript/site figures for one immutable pipeline run."""
    run = Path(run_path)
    analysis_path = run / "article_analysis.csv"
    manifest_path = run / "run_manifest.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(f"Required analysis artifact not found: {analysis_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required run manifest not found: {manifest_path}")

    out = Path(output_dir) if output_dir is not None else run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for existing in out.iterdir():
        if existing.is_file():
            existing.unlink()

    rows = _read_rows(analysis_path)
    figure_manifest: list[dict[str, Any]] = []
    for task in ("sentiment", "topic"):
        _plot_model_performance(rows, out, task, figure_manifest)
        _plot_probabilistic_metrics(rows, out, task, figure_manifest)
        _plot_confusion_matrix(rows, out, task, figure_manifest)
        _plot_classwise(rows, out, task, figure_manifest)
        _plot_class_distribution(rows, out, task, figure_manifest)
        _plot_calibration(rows, out, task, figure_manifest)
        _plot_training_history(rows, out, task, figure_manifest)
        _plot_segments(rows, out, task, figure_manifest)
        _plot_error_confidence(rows, out, task, figure_manifest)

    figures_manifest_path = out / "figures_manifest.json"
    figures_manifest_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "source": "../article_analysis.csv",
        "formats": list(IMAGE_FORMATS),
        "figure_files": figure_manifest,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest.setdefault("outputs", {})["figures_manifest"] = "figures/figures_manifest.json"
    run_manifest["figures"] = {
        "directory": "figures",
        "files": len(figure_manifest),
        "logical_figures": len({item["file"].rsplit(".", 1)[0] for item in figure_manifest}),
        "formats": list(IMAGE_FORMATS),
        "manifest_sha256": _sha256(figures_manifest_path),
    }
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return figures_manifest_path
