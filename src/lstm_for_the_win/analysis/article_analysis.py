"""Export the analyses used by the manuscript and dashboard to one versioned CSV."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ANALYSIS_COLUMNS = (
    "run_id",
    "input_generation",
    "analysis_group",
    "task",
    "model",
    "segment_dimension",
    "segment_value",
    "class_label",
    "expected_label",
    "predicted_label",
    "metric",
    "value",
    "support",
    "record_id",
    "text",
    "linguistic_level",
    "flagprofanity",
    "hasemoji",
    "hasspellingerror",
    "hasslang",
    "length_class",
    "mixed_sentiment",
    "goldtest",
)

STYLE_FIELDS = (
    "linguistic_level",
    "flagprofanity",
    "hasemoji",
    "hasspellingerror",
    "hasslang",
    "length_class",
    "mixed_sentiment",
    "goldtest",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _add(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    *,
    analysis_group: str,
    task: str = "",
    model: str = "",
    metric: str,
    value: Any,
    support: Any = "",
    segment_dimension: str = "",
    segment_value: str = "",
    class_label: str = "",
    expected_label: str = "",
    predicted_label: str = "",
    record: dict[str, str] | None = None,
) -> None:
    output = {column: "" for column in ANALYSIS_COLUMNS}
    output.update(base)
    output.update(
        {
            "analysis_group": analysis_group,
            "task": task,
            "model": model,
            "segment_dimension": segment_dimension,
            "segment_value": segment_value,
            "class_label": class_label,
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "metric": metric,
            "value": value,
            "support": support,
        }
    )
    if record is not None:
        output["record_id"] = record.get("ID", "")
        output["text"] = record.get("text", "")
        for field in STYLE_FIELDS:
            output[field] = record.get(field, "")
    rows.append(output)


def _aggregate_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    for task in ("sentiment", "topic"):
        result = metrics[task]
        support = int(result["incoming_size"])
        for model, key in (("lstm", "metrics"), ("tfidf_logistic", "baseline_metrics")):
            for metric, value in sorted(result[key].items()):
                _add(
                    rows,
                    base,
                    analysis_group="aggregate_metrics",
                    task=task,
                    model=model,
                    metric=metric,
                    value=value,
                    support=support,
                )
        for metric, value in sorted(result.get("metric_delta_vs_baseline", {}).items()):
            _add(
                rows,
                base,
                analysis_group="model_delta",
                task=task,
                model="lstm_minus_tfidf_logistic",
                metric=metric,
                value=value,
                support=support,
            )


def _segment_metrics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    metrics: dict[str, Any],
    predictions: list[dict[str, str]],
) -> None:
    total = len(predictions)
    for task in ("sentiment", "topic"):
        for dimension, values in sorted(metrics[task].get("segment_metrics", {}).items()):
            counts = Counter(record.get(dimension, "") for record in predictions)
            for segment_value, segment_metrics in sorted(values.items()):
                support = counts.get(str(segment_value), 0)
                _add(
                    rows,
                    base,
                    analysis_group="segment_composition",
                    task=task,
                    model="lstm",
                    segment_dimension=dimension,
                    segment_value=str(segment_value),
                    metric="share",
                    value=(support / total if total else 0.0),
                    support=support,
                )
                for metric, value in sorted(segment_metrics.items()):
                    _add(
                        rows,
                        base,
                        analysis_group="segment_metrics",
                        task=task,
                        model="lstm",
                        segment_dimension=dimension,
                        segment_value=str(segment_value),
                        metric=metric,
                        value=value,
                        support=support,
                    )


def _classification_diagnostics(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    metrics: dict[str, Any],
    predictions: list[dict[str, str]],
) -> None:
    total = len(predictions)
    for task in ("sentiment", "topic"):
        expected_field = f"expected_{task}"
        predicted_field = f"predicted_{task}"
        confidence_field = f"{task}_confidence"
        correct_field = f"{task}_correct"
        labels = list(metrics[task]["labels"])

        expected_counts = Counter(record[expected_field] for record in predictions)
        predicted_counts = Counter(record[predicted_field] for record in predictions)
        pair_counts = Counter((record[expected_field], record[predicted_field]) for record in predictions)

        for expected_label in labels:
            for predicted_label in labels:
                _add(
                    rows,
                    base,
                    analysis_group="confusion_matrix",
                    task=task,
                    model="lstm",
                    expected_label=expected_label,
                    predicted_label=predicted_label,
                    metric="count",
                    value=pair_counts[(expected_label, predicted_label)],
                    support=expected_counts[expected_label],
                )

        for label in labels:
            tp = pair_counts[(label, label)]
            fp = sum(pair_counts[(other, label)] for other in labels if other != label)
            fn = sum(pair_counts[(label, other)] for other in labels if other != label)
            tn = total - tp - fp - fn
            support = expected_counts[label]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            specificity = tn / (tn + fp) if tn + fp else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_metrics = {
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "support": support,
                "predicted_count": predicted_counts[label],
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
            }
            for metric, value in class_metrics.items():
                _add(
                    rows,
                    base,
                    analysis_group="classwise_metrics",
                    task=task,
                    model="lstm",
                    class_label=label,
                    metric=metric,
                    value=value,
                    support=support,
                )

            for distribution, count in (
                ("expected_count", expected_counts[label]),
                ("predicted_count", predicted_counts[label]),
            ):
                _add(
                    rows,
                    base,
                    analysis_group="class_distribution",
                    task=task,
                    model="lstm",
                    class_label=label,
                    metric=distribution,
                    value=count,
                    support=total,
                )
                _add(
                    rows,
                    base,
                    analysis_group="class_distribution",
                    task=task,
                    model="lstm",
                    class_label=label,
                    metric=distribution.replace("count", "proportion"),
                    value=(count / total if total else 0.0),
                    support=total,
                )

        confidences = [float(record[confidence_field]) for record in predictions]
        correctness = [_as_bool(record[correct_field]) for record in predictions]
        accuracy = sum(correctness) / total if total else 0.0
        mean_confidence = sum(confidences) / total if total else 0.0
        for metric, value in (
            ("accuracy", accuracy),
            ("mean_selected_class_confidence", mean_confidence),
            ("confidence_minus_accuracy", mean_confidence - accuracy),
        ):
            _add(
                rows,
                base,
                analysis_group="confidence_accuracy",
                task=task,
                model="lstm",
                metric=metric,
                value=value,
                support=total,
            )

        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            indices = [
                index
                for index, confidence in enumerate(confidences)
                if confidence >= lower and (confidence <= upper if bin_index == 9 else confidence < upper)
            ]
            if not indices:
                continue
            bin_support = len(indices)
            bin_confidence = sum(confidences[index] for index in indices) / bin_support
            bin_accuracy = sum(correctness[index] for index in indices) / bin_support
            abs_gap = abs(bin_accuracy - bin_confidence)
            weight = bin_support / total
            label = f"[{lower:.1f},{upper:.1f}{']' if bin_index == 9 else ')'}"
            for metric, value in (
                ("mean_confidence", bin_confidence),
                ("accuracy", bin_accuracy),
                ("absolute_gap", abs_gap),
                ("weight", weight),
                ("ece_contribution", weight * abs_gap),
            ):
                _add(
                    rows,
                    base,
                    analysis_group="calibration_bins",
                    task=task,
                    model="lstm",
                    segment_dimension="confidence_bin",
                    segment_value=label,
                    metric=metric,
                    value=value,
                    support=bin_support,
                )

        errors = [record for record in predictions if not _as_bool(record[correct_field])]
        _add(
            rows,
            base,
            analysis_group="error_summary",
            task=task,
            model="lstm",
            metric="error_count",
            value=len(errors),
            support=total,
        )
        for record in errors:
            _add(
                rows,
                base,
                analysis_group="error_record",
                task=task,
                model="lstm",
                expected_label=record[expected_field],
                predicted_label=record[predicted_field],
                metric="confidence",
                value=record[confidence_field],
                support=len(errors),
                record=record,
            )


def _training_history(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    metrics: dict[str, Any],
    results: dict[str, Any] | None,
) -> None:
    if not results:
        return
    for task in ("sentiment", "topic"):
        history = results.get(task, {}).get("history", {})
        if not history:
            continue
        epochs = max((len(values) for values in history.values()), default=0)
        for epoch in range(epochs):
            for metric, values in sorted(history.items()):
                if epoch >= len(values):
                    continue
                _add(
                    rows,
                    base,
                    analysis_group="training_history",
                    task=task,
                    model="lstm",
                    segment_dimension="epoch",
                    segment_value=str(epoch + 1),
                    metric=metric,
                    value=values[epoch],
                    support=metrics[task]["fit_size"],
                )

        val_loss = history.get("val_loss", [])
        if val_loss:
            best_index = min(range(len(val_loss)), key=val_loss.__getitem__)
            _add(
                rows,
                base,
                analysis_group="validation_summary",
                task=task,
                model="lstm",
                metric="best_epoch",
                value=best_index + 1,
                support=metrics[task]["validation_size"],
            )
            for metric in ("val_loss", "val_accuracy", "loss", "accuracy"):
                values = history.get(metric, [])
                if best_index < len(values):
                    _add(
                        rows,
                        base,
                        analysis_group="validation_summary",
                        task=task,
                        model="lstm",
                        metric=f"{metric}_at_best_val_loss",
                        value=values[best_index],
                        support=metrics[task]["validation_size"] if metric.startswith("val_") else metrics[task]["fit_size"],
                    )
            val_accuracy = history.get("val_accuracy", [])
            if best_index < len(val_accuracy):
                incoming_accuracy = float(metrics[task]["metrics"]["accuracy"])
                _add(
                    rows,
                    base,
                    analysis_group="validation_incoming_gap",
                    task=task,
                    model="lstm",
                    metric="validation_accuracy_minus_incoming_accuracy",
                    value=float(val_accuracy[best_index]) - incoming_accuracy,
                    support=metrics[task]["incoming_size"],
                )


def _run_metadata(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    parameters = manifest.get("parameters", {})
    for metric, value in sorted(parameters.items()):
        _add(rows, base, analysis_group="run_metadata", metric=metric, value=value)
    for metric in ("train_size", "fit_size", "validation_size", "incoming_size"):
        _add(
            rows,
            base,
            analysis_group="run_metadata",
            task="sentiment",
            model="lstm",
            metric=metric,
            value=metrics["sentiment"][metric],
        )


def build_article_analysis(run_path: str | Path) -> list[dict[str, Any]]:
    """Build all manuscript/dashboard diagnostics for one immutable run directory."""

    path = Path(run_path)
    metrics_path = path / "metrics.json"
    predictions_path = path / "predictions.csv"
    manifest_path = path / "run_manifest.json"
    for required in (metrics_path, predictions_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required run artifact not found: {required}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = _read_csv(predictions_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_path = path / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else None

    base = {
        "run_id": manifest.get("run_id", path.name),
        "input_generation": manifest.get("input_generation", ""),
    }
    rows: list[dict[str, Any]] = []
    _run_metadata(rows, base, manifest, metrics)
    _aggregate_metrics(rows, base, metrics)
    _segment_metrics(rows, base, metrics, predictions)
    _classification_diagnostics(rows, base, metrics, predictions)
    _training_history(rows, base, metrics, results)
    return rows


def export_article_analysis(run_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Write article_analysis.csv and register it in the run manifest."""

    path = Path(run_path)
    destination = Path(output_path) if output_path is not None else path / "article_analysis.csv"
    rows = build_article_analysis(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANALYSIS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = path / "run_manifest.json"
    if destination.parent.resolve() == path.resolve():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("outputs", {})["article_analysis"] = destination.name
        manifest["article_analysis"] = {
            "rows": len(rows),
            "sha256": _sha256(destination),
            "schema_version": "1.0.0",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
