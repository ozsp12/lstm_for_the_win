"""Build the single immutable JSON artifact for one experiment run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev, stdev
from typing import Any, Mapping, Sequence

from .classification import PipelineExecution, PipelineResult
from .classification.data import label_for, load_incoming
from .classification.model import build_confusion_matrix, classification_metrics, predict_probabilities
from .external_benchmark import load_external_sentiment

SCHEMA_VERSION = "2.0.0"
COMPARABLE_METRICS = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int | str]:
    if total <= 0:
        low = high = 0.0
    else:
        p = correct / total
        z2 = z * z
        denominator = 1.0 + z2 / total
        center = (p + z2 / (2.0 * total)) / denominator
        margin = z * ((p * (1.0 - p) / total + z2 / (4.0 * total * total)) ** 0.5) / denominator
        low, high = max(0.0, center - margin), min(1.0, center + margin)
    return {
        "method": "wilson",
        "confidence_level": 0.95,
        "low": low,
        "high": high,
        "support": total,
    }


def task_payload(execution: PipelineExecution) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = execution.result.to_dict()
    predictions = {str(row["ID"]): row for row in payload.pop("predictions")}
    total = int(payload["incoming_size"])
    correct = sum(bool(row["correct"]) for row in predictions.values())
    payload["uncertainty"] = {"accuracy_ci95": wilson_interval(correct, total)}
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
        reviews.append(
            {
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
            }
        )
    return reviews


def summarize_replicates(results: Sequence[PipelineResult]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "metrics": {}, "baseline_metrics": {}}

    def summarize(group: str) -> dict[str, dict[str, float | dict[str, float]]]:
        output: dict[str, dict[str, float | dict[str, float]]] = {}
        for metric in COMPARABLE_METRICS:
            values = [float(getattr(result, group)[metric]) for result in results]
            mean = fmean(values)
            sample_std = stdev(values) if len(values) > 1 else 0.0
            half_width = 1.959963984540054 * sample_std / (len(values) ** 0.5) if len(values) > 1 else 0.0
            output[metric] = {
                "mean": mean,
                "std_population": pstdev(values) if len(values) > 1 else 0.0,
                "std_sample": sample_std,
                "min": min(values),
                "max": max(values),
                "mean_ci95_normal": {
                    "method": "normal_approximation_across_model_seeds",
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


def evaluate_external_sentiment(
    execution: PipelineExecution,
    external_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the trained three-class sentiment model on real UCI Amazon review sentences."""

    rows = load_external_sentiment(external_path)
    labels = list(execution.result.labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    expected_labels = [row["expected_sentiment"] for row in rows]
    if not set(expected_labels).issubset(label_to_index):
        raise ValueError("External sentiment benchmark contains a label absent from training.")
    expected = [label_to_index[label] for label in expected_labels]
    probabilities = predict_probabilities(execution.model, [row["text"] for row in rows])
    predicted = probabilities.argmax(axis=1)
    correct = int(sum(int(left == right) for left, right in zip(expected, predicted, strict=True)))
    reviews = [
        {
            "ID": row["ID"],
            "text": row["text"],
            "expected_sentiment": row["expected_sentiment"],
            "predicted_sentiment": labels[int(predicted[index])],
            "confidence": float(probabilities[index][int(predicted[index])]),
            "correct": bool(expected[index] == int(predicted[index])),
        }
        for index, row in enumerate(rows)
    ]
    return {
        "immutable": True,
        "real_world": True,
        "task": "sentiment",
        "topic_evaluation": "not_available_in_source_dataset",
        "dataset": dict(manifest),
        "labels_in_source": ["negative", "positive"],
        "model_label_space": labels,
        "support": len(rows),
        "metrics": classification_metrics(expected, probabilities, len(labels)),
        "confusion_matrix": build_confusion_matrix(expected, predicted, len(labels)),
        "uncertainty": {"accuracy_ci95": wilson_interval(correct, len(rows))},
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
