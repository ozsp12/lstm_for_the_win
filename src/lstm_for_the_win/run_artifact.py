"""Build the single immutable JSON artifact for one experiment run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from .classification import PipelineExecution, PipelineResult
from .classification.data import label_for, load_incoming
from .classification.model import (
    build_confusion_matrix,
    classification_metrics,
    predict_probabilities,
    wilson_interval,
)
from .external_benchmark import load_external_sentiment

SCHEMA_VERSION = "2.0.0"
COMPARABLE_METRICS = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_payload(execution: PipelineExecution) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = execution.result.to_dict()
    predictions = {str(row["ID"]): row for row in payload.pop("predictions")}
    total = int(payload["incoming_size"])
    correct = sum(bool(row["correct"]) for row in predictions.values())
    payload["uncertainty"] = {"primary_seed_accuracy_ci95": wilson_interval(correct, total)}
    return payload, predictions


def merge_reviews(
    source_rows: Sequence[Mapping[str, str]],
    sentiment_predictions: Mapping[str, Mapping[str, Any]],
    topic_predictions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_ids = {row["ID"] for row in source_rows}
    if set(sentiment_predictions) != expected_ids or set(topic_predictions) != expected_ids:
        raise ValueError("Task predictions do not cover every evaluation ID.")
    reviews: list[dict[str, Any]] = []
    for source in source_rows:
        sid = source["ID"]
        sentiment = sentiment_predictions[sid]
        topic = topic_predictions[sid]
        reviews.append({
            "ID": int(sid),
            "text": source["text"],
            "expected_sentiment": source["expected_sentiment"],
            "expected_topic": source["expected_topic"],
            "predicted_sentiment": sentiment["predicted"],
            "predicted_topic": topic["predicted"],
            "sentiment_confidence": float(sentiment["confidence"]),
            "topic_confidence": float(topic["confidence"]),
            "sentiment_correct": bool(sentiment["correct"]),
            "topic_correct": bool(topic["correct"]),
            "linguistic_level": source["linguistic_level"],
            "flagprofanity": int(source["flagprofanity"]),
            "hasemoji": int(source.get("hasemoji", sentiment.get("hasemoji", 0))),
            "hasspellingerror": int(source.get("hasspellingerror", sentiment.get("hasspellingerror", 0))),
            "hasslang": int(source.get("hasslang", sentiment.get("hasslang", 0))),
            "length_class": source.get("length_class", sentiment.get("length_class", "")),
            "mixed_sentiment": int(source.get("mixed_sentiment", sentiment.get("mixed_sentiment", 0))),
            "goldtest": int(source["goldtest"]),
            "template_family": source.get("template_family", "legacy-unmaterialized"),
            "input_timestamp": source["input_timestamp"],
        })
    return reviews


def summarize_replicates(results: Sequence[PipelineResult]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "metrics": {}, "baseline_metrics": {}}

    def summarize(group: str) -> dict[str, dict[str, float | dict[str, float | str]]]:
        output: dict[str, dict[str, float | dict[str, float | str]]] = {}
        for metric in COMPARABLE_METRICS:
            values = [float(getattr(result, group)[metric]) for result in results]
            mean = fmean(values)
            sample_std = stdev(values) if len(values) > 1 else 0.0
            if len(values) > 1:
                critical = float(student_t.ppf(0.975, df=len(values) - 1))
                half_width = critical * sample_std / (len(values) ** 0.5)
            else:
                half_width = 0.0
            output[metric] = {
                "mean": mean,
                "std_population": pstdev(values) if len(values) > 1 else 0.0,
                "std_sample": sample_std,
                "min": min(values),
                "max": max(values),
                "mean_ci95": {
                    "method": "student_t_across_model_seeds",
                    "low": mean - half_width,
                    "high": mean + half_width,
                },
            }
        return output

    return {
        "count": len(results),
        "seeds": [int(result.seed) for result in results if hasattr(result, "seed")],
        "metrics": summarize("metrics"),
        "baseline_metrics": summarize("baseline_metrics"),
    }


def _promote_replicate_estimates(payload: dict[str, Any]) -> None:
    """Make across-seed means canonical while preserving the primary-seed diagnostics."""

    replicates = payload.get("replicates", {})
    if not replicates or not replicates.get("count"):
        return
    payload["primary_seed_metrics"] = dict(payload["metrics"])
    payload["primary_seed_baseline_metrics"] = dict(payload["baseline_metrics"])
    payload["primary_seed_metric_delta_vs_baseline"] = dict(payload.get("metric_delta_vs_baseline", {}))

    for destination, source in (("metrics", "metrics"), ("baseline_metrics", "baseline_metrics")):
        canonical = dict(payload[destination])
        for metric, stats in replicates.get(source, {}).items():
            canonical[metric] = float(stats["mean"])
        payload[destination] = canonical

    payload["metric_delta_vs_baseline"] = {
        metric: float(payload["metrics"][metric] - payload["baseline_metrics"][metric])
        for metric in COMPARABLE_METRICS
    }
    payload["canonical_estimate"] = {
        "method": "mean_across_model_seeds",
        "seed_count": int(replicates["count"]),
        "seeds": list(replicates.get("seeds", [])),
        "metrics": list(COMPARABLE_METRICS),
    }
    payload.setdefault("uncertainty", {})["across_seed_ci95"] = {
        metric: dict(stats["mean_ci95"])
        for metric, stats in replicates.get("metrics", {}).items()
    }


def evaluate_benchmark(
    executions: Mapping[str, PipelineExecution],
    benchmark_path: Path,
    raw_rows: Sequence[Mapping[str, str]],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = load_incoming(benchmark_path)
    task_output: dict[str, Any] = {}
    prediction_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for task in ("sentiment", "topic"):
        execution = executions[task]
        labels = list(execution.result.labels)
        label_to_index = {label: index for index, label in enumerate(labels)}
        expected_labels = [label_for(record, task) for record in records]
        if not set(expected_labels).issubset(label_to_index):
            raise ValueError(f"benchmark.csv contains a {task} label absent from training.")
        expected = [label_to_index[label] for label in expected_labels]
        probabilities = predict_probabilities(execution.model, [record.text for record in records])
        predicted_indices = probabilities.argmax(axis=1)
        predictions: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(records):
            predicted_index = int(predicted_indices[index])
            predictions[str(record.ID)] = {
                "ID": record.ID,
                "predicted": labels[predicted_index],
                "confidence": float(probabilities[index][predicted_index]),
                "correct": bool(expected[index] == predicted_index),
                "hasemoji": record.hasemoji,
                "hasspellingerror": record.hasspellingerror,
                "hasslang": record.hasslang,
                "length_class": record.length_class,
                "mixed_sentiment": record.mixed_sentiment,
            }
        correct = sum(bool(row["correct"]) for row in predictions.values())
        task_output[task] = {
            "labels": labels,
            "support": len(records),
            "metrics": classification_metrics(expected, probabilities, len(labels)),
            "confusion_matrix": build_confusion_matrix(expected, predicted_indices, len(labels)),
            "confusion_matrix_contract": {
                "axis_convention": "rows_expected_columns_predicted",
                "expected_labels": labels,
                "predicted_labels": labels,
            },
            "uncertainty": {"accuracy_ci95": wilson_interval(correct, len(records))},
        }
        prediction_maps[task] = predictions
    return {
        "immutable": True,
        "source": "non-gold rows from the first incoming batch observed after benchmark support was introduced",
        "provenance": dict(provenance or {}),
        "tasks": task_output,
        "reviews": merge_reviews(raw_rows, prediction_maps["sentiment"], prediction_maps["topic"]),
    }


def _source_label_metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    source_labels = ("negative", "positive")
    class_metrics: dict[str, dict[str, float | int]] = {}
    for label in source_labels:
        tp = sum(left == label and right == label for left, right in zip(expected, predicted, strict=True))
        fp = sum(left != label and right == label for left, right in zip(expected, predicted, strict=True))
        fn = sum(left == label and right != label for left, right in zip(expected, predicted, strict=True))
        support = sum(left == label for left in expected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    return {
        "labels": list(source_labels),
        "precision_macro": fmean(float(class_metrics[label]["precision"]) for label in source_labels),
        "recall_macro": fmean(float(class_metrics[label]["recall"]) for label in source_labels),
        "macro_f1": fmean(float(class_metrics[label]["f1"]) for label in source_labels),
        "per_class": class_metrics,
    }


def evaluate_external_sentiment(
    execution: PipelineExecution,
    external_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the three-class model and a source-compatible binary restriction on UCI Amazon reviews."""

    rows = load_external_sentiment(external_path)
    labels = list(execution.result.labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    expected_labels = [row["expected_sentiment"] for row in rows]
    if not set(expected_labels).issubset(label_to_index):
        raise ValueError("External sentiment benchmark contains a label absent from training.")

    expected_indices = [label_to_index[label] for label in expected_labels]
    probabilities = predict_probabilities(execution.model, [row["text"] for row in rows])
    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = [labels[int(index)] for index in predicted_indices]
    full_correct = int(sum(left == right for left, right in zip(expected_labels, predicted_labels, strict=True)))
    full_metrics = classification_metrics(expected_indices, probabilities, len(labels))

    source_labels = ["negative", "positive"]
    source_indices = [label_to_index[label] for label in source_labels]
    restricted_probabilities = probabilities[:, source_indices]
    restricted_probabilities = restricted_probabilities / restricted_probabilities.sum(axis=1, keepdims=True)
    restricted_indices = restricted_probabilities.argmax(axis=1)
    restricted_predictions = [source_labels[int(index)] for index in restricted_indices]
    restricted_expected = [source_labels.index(label) for label in expected_labels]
    restricted_correct = int(sum(left == right for left, right in zip(expected_labels, restricted_predictions, strict=True)))
    restricted_metrics = classification_metrics(restricted_expected, restricted_probabilities, 2)

    full_confusion = {
        expected_label: {
            predicted_label: sum(
                left == expected_label and right == predicted_label
                for left, right in zip(expected_labels, predicted_labels, strict=True)
            )
            for predicted_label in labels
        }
        for expected_label in source_labels
    }
    restricted_confusion = {
        expected_label: {
            predicted_label: sum(
                left == expected_label and right == predicted_label
                for left, right in zip(expected_labels, restricted_predictions, strict=True)
            )
            for predicted_label in source_labels
        }
        for expected_label in source_labels
    }

    reviews = []
    for index, row in enumerate(rows):
        full_index = int(predicted_indices[index])
        restricted_index = int(restricted_indices[index])
        reviews.append({
            "ID": row["ID"],
            "text": row["text"],
            "expected_sentiment": row["expected_sentiment"],
            "predicted_sentiment": predicted_labels[index],
            "confidence": float(probabilities[index][full_index]),
            "correct": bool(row["expected_sentiment"] == predicted_labels[index]),
            "probabilities": {label: float(probabilities[index][position]) for position, label in enumerate(labels)},
            "binary_restricted_predicted_sentiment": restricted_predictions[index],
            "binary_restricted_confidence": float(restricted_probabilities[index][restricted_index]),
            "binary_restricted_correct": bool(row["expected_sentiment"] == restricted_predictions[index]),
        })

    return {
        "immutable": True,
        "real_world": True,
        "task": "sentiment",
        "topic_evaluation": "not_available_in_source_dataset",
        "dataset": dict(manifest),
        "labels_in_source": source_labels,
        "model_label_space": labels,
        "support": len(rows),
        "accuracy": float(full_metrics["accuracy"]),
        "full_label_space_accuracy": float(full_metrics["accuracy"]),
        "binary_restricted_accuracy": float(restricted_metrics["accuracy"]),
        "source_label_metrics": _source_label_metrics(expected_labels, restricted_predictions),
        "full_label_space_source_metrics": _source_label_metrics(expected_labels, predicted_labels),
        "probabilistic_metrics": {
            key: float(full_metrics[key])
            for key in ("log_loss", "brier_score", "expected_calibration_error")
        },
        "binary_restricted_probabilistic_metrics": {
            key: float(restricted_metrics[key])
            for key in ("log_loss", "brier_score", "expected_calibration_error")
        },
        "neutral_prediction_rate": predicted_labels.count("neutral") / len(predicted_labels) if predicted_labels else 0.0,
        "confusion_matrix": {
            "axis_convention": "rows_expected_columns_predicted",
            "expected_labels": source_labels,
            "predicted_labels": labels,
            "matrix": full_confusion,
        },
        "binary_restricted_confusion_matrix": {
            "axis_convention": "rows_expected_columns_predicted",
            "expected_labels": source_labels,
            "predicted_labels": source_labels,
            "matrix": restricted_confusion,
        },
        "uncertainty": {
            "full_label_space_accuracy_ci95": wilson_interval(full_correct, len(rows)),
            "binary_restricted_accuracy_ci95": wilson_interval(restricted_correct, len(rows)),
        },
        "reviews": reviews,
    }


def build_run_document(
    *,
    run_metadata: dict[str, Any],
    scope: dict[str, Any],
    executions: Mapping[str, PipelineExecution],
    incoming_rows: Sequence[Mapping[str, str]],
    input_files: Sequence[Path],
    replicate_results: Mapping[str, Sequence[PipelineResult]] | None = None,
    benchmark: dict[str, Any] | None = None,
    external_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sentiment, sentiment_predictions = task_payload(executions["sentiment"])
    topic, topic_predictions = task_payload(executions["topic"])
    if replicate_results:
        sentiment["replicates"] = summarize_replicates(replicate_results.get("sentiment", []))
        topic["replicates"] = summarize_replicates(replicate_results.get("topic", []))
        _promote_replicate_estimates(sentiment)
        _promote_replicate_estimates(topic)
    run_metadata = dict(run_metadata)
    run_metadata["input_files"] = {
        path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in input_files
    }
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "experiment_run",
        "run": run_metadata,
        "scope": scope,
        "tasks": {"sentiment": sentiment, "topic": topic},
        "reviews": merge_reviews(incoming_rows, sentiment_predictions, topic_predictions),
    }
    if benchmark is not None:
        document["benchmark"] = benchmark
    if external_validation is not None:
        document["external_validation"] = external_validation
    return document


def write_run_json(path: Path, document: Mapping[str, Any]) -> Path:
    destination = path / "run.json"
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
