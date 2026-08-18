from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from lstm_for_the_win.derived_artifacts import materialize_derived_artifacts


def _task(task: str, labels: list[str]) -> dict[str, object]:
    matrix = [[2 if i == j else 0 for j in range(len(labels))] for i in range(len(labels))]
    metrics = {
        "accuracy": 1.0,
        "precision_macro": 1.0,
        "recall_macro": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "log_loss": 0.1,
        "brier_score": 0.05,
        "expected_calibration_error": 0.02,
    }
    comparable = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score")
    replicate_stats = {
        metric: {
            "mean": metrics[metric],
            "std_population": 0.0,
            "std_sample": 0.0,
            "min": metrics[metric],
            "max": metrics[metric],
            "mean_ci95": {"method": "student_t_across_model_seeds", "low": metrics[metric], "high": metrics[metric]},
        }
        for metric in comparable
    }
    return {
        "task": task,
        "seed": 42,
        "incoming_size": len(labels) * 2,
        "labels": labels,
        "metrics": metrics,
        "baseline_metrics": dict(metrics),
        "metric_delta_vs_baseline": {metric: 0.0 for metric in comparable},
        "uncertainty": {"accuracy_ci95": {"method": "wilson", "low": 0.7, "high": 1.0, "support": len(labels) * 2}},
        "replicates": {"count": 2, "seeds": [42, 43], "metrics": replicate_stats, "baseline_metrics": replicate_stats},
        "segment_metrics": {"linguistic_level": {"standard": {"accuracy": 1.0}}},
        "history": {"accuracy": [0.8, 1.0], "loss": [0.4, 0.1]},
        "confusion_matrix": matrix,
    }


def test_materialized_artifacts_are_deterministic_and_complete(tmp_path: Path) -> None:
    sentiment_labels = ["negative", "neutral", "positive"]
    topic_labels = ["refrigerator", "smartphone", "television", "washing_machine"]
    benchmark = {
        "sentiment": {"labels": sentiment_labels, "support": 6, "metrics": {"accuracy": 0.9}, "confusion_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]},
        "topic": {"labels": topic_labels, "support": 8, "metrics": {"accuracy": 0.8}, "confusion_matrix": [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]},
    }
    external_confusion = {
        "negative": {"negative": 1, "neutral": 0, "positive": 0},
        "positive": {"negative": 0, "neutral": 1, "positive": 1},
    }
    run = {
        "schema_version": "2.0.0",
        "artifact_type": "experiment_run",
        "run": {"run_id": "unit", "input_generation": 0},
        "tasks": {"sentiment": _task("sentiment", sentiment_labels), "topic": _task("topic", topic_labels)},
        "benchmark": {"tasks": benchmark},
        "external_validation": {
            "real_world": True,
            "task": "sentiment",
            "support": 3,
            "labels_in_source": ["negative", "positive"],
            "model_label_space": sentiment_labels,
            "accuracy": 2 / 3,
            "source_label_metrics": {
                "precision_macro": 0.75,
                "recall_macro": 0.75,
                "macro_f1": 0.75,
                "per_class": {
                    "negative": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1},
                    "positive": {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2},
                },
            },
            "probabilistic_metrics": {"log_loss": 0.5, "brier_score": 0.2, "expected_calibration_error": 0.1},
            "neutral_prediction_rate": 1 / 3,
            "confusion_matrix": {
                "expected_labels": ["negative", "positive"],
                "predicted_labels": sentiment_labels,
                "matrix": external_confusion,
            },
        },
        "reviews": [
            {
                "ID": 11,
                "text": "A test review",
                "expected_sentiment": "positive",
                "predicted_sentiment": "positive",
                "sentiment_confidence": 0.9,
                "sentiment_correct": True,
                "expected_topic": "smartphone",
                "predicted_topic": "smartphone",
                "topic_confidence": 0.8,
                "topic_correct": True,
                "linguistic_level": "standard",
                "flagprofanity": 0,
                "hasemoji": 0,
                "hasspellingerror": 0,
                "hasslang": 0,
                "length_class": "short",
                "mixed_sentiment": 0,
                "goldtest": 0,
                "template_family": "using",
            }
        ],
    }
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run, sort_keys=True), encoding="utf-8")

    materialize_derived_artifacts(run_path)

    csv_path = tmp_path / "article_analysis.csv"
    figures = sorted((tmp_path / "figures").glob("*.svg"))
    assert {path.name for path in figures} == {
        "benchmark_accuracy.svg",
        "external_sentiment_confusion.svg",
        "incoming_model_accuracy.svg",
        "incoming_sentiment_confusion.svg",
        "incoming_topic_confusion.svg",
    }
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {row["analysis_group"] for row in rows}
    assert {"aggregate_metric", "benchmark_metric", "external_metric", "external_class_metric", "confusion_matrix", "prediction_record"}.issubset(groups)
    assert sum(row["analysis_group"] == "prediction_record" for row in rows) == 1
    assert sum(row["analysis_group"] == "confusion_matrix" and row["dataset"] == "uci_amazon" for row in rows) == 6

    def hashes() -> dict[str, str]:
        files = [csv_path, *figures]
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    before = hashes()
    materialize_derived_artifacts(run_path)
    after = hashes()
    assert before == after
