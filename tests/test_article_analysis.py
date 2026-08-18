from __future__ import annotations

import csv
import json
from pathlib import Path

from lstm_for_the_win.analysis.article_analysis import export_article_analysis


def _metric_block(accuracy: float) -> dict[str, float]:
    return {
        "accuracy": accuracy, "precision_macro": accuracy, "recall_macro": accuracy,
        "macro_f1": accuracy, "weighted_f1": accuracy, "log_loss": 0.2,
        "brier_score": 0.1, "expected_calibration_error": 0.05,
    }


def test_export_article_analysis_uses_only_canonical_json(tmp_path: Path) -> None:
    run_path = tmp_path / "github-test"
    run_path.mkdir()
    reviews = [
        {"ID": 1, "text": "works well 🙂", "expected_sentiment": "positive", "expected_topic": "smartphone", "predicted_sentiment": "positive", "predicted_topic": "smartphone", "sentiment_confidence": 0.9, "topic_confidence": 0.8, "sentiment_correct": True, "topic_correct": True, "linguistic_level": "informal", "flagprofanity": 0, "hasemoji": 1, "hasspellingerror": 0, "hasslang": 1, "length_class": "short", "mixed_sentiment": 0, "goldtest": 1},
        {"ID": 2, "text": "bad phone", "expected_sentiment": "negative", "expected_topic": "smartphone", "predicted_sentiment": "positive", "predicted_topic": "washing_machine", "sentiment_confidence": 0.7, "topic_confidence": 0.6, "sentiment_correct": False, "topic_correct": False, "linguistic_level": "limited", "flagprofanity": 1, "hasemoji": 0, "hasspellingerror": 1, "hasslang": 0, "length_class": "medium", "mixed_sentiment": 1, "goldtest": 0},
        {"ID": 3, "text": "average fridge", "expected_sentiment": "neutral", "expected_topic": "refrigerator", "predicted_sentiment": "neutral", "predicted_topic": "refrigerator", "sentiment_confidence": 0.8, "topic_confidence": 0.9, "sentiment_correct": True, "topic_correct": True, "linguistic_level": "standard", "flagprofanity": 0, "hasemoji": 0, "hasspellingerror": 0, "hasslang": 0, "length_class": "medium", "mixed_sentiment": 0, "goldtest": 0},
        {"ID": 4, "text": "solid washer", "expected_sentiment": "positive", "expected_topic": "washing_machine", "predicted_sentiment": "positive", "predicted_topic": "washing_machine", "sentiment_confidence": 0.95, "topic_confidence": 0.95, "sentiment_correct": True, "topic_correct": True, "linguistic_level": "technical", "flagprofanity": 0, "hasemoji": 0, "hasspellingerror": 0, "hasslang": 0, "length_class": "long", "mixed_sentiment": 0, "goldtest": 1},
    ]
    task = {
        "task": "sentiment", "train_size": 1200, "fit_size": 1000, "validation_size": 200,
        "incoming_size": 4, "labels": ["negative", "neutral", "positive"],
        "metrics": _metric_block(0.75), "baseline_metrics": _metric_block(0.5),
        "metric_delta_vs_baseline": {"accuracy": 0.25},
        "segment_metrics": {"linguistic_level": {"informal": _metric_block(1.0), "limited": _metric_block(0.0)}, "hasemoji": {"0": _metric_block(2 / 3), "1": _metric_block(1.0)}},
        "history": {"accuracy": [0.8, 0.9], "loss": [0.5, 0.3], "val_accuracy": [0.7, 0.85], "val_loss": [0.4, 0.2]},
        "confusion_matrix": [],
        "uncertainty": {"accuracy_ci95": {"method": "wilson", "confidence_level": 0.95, "low": 0.30, "high": 0.95, "support": 4}},
    }
    topic = dict(task)
    topic["task"] = "topic"
    topic["labels"] = ["refrigerator", "smartphone", "washing_machine"]
    analysis = {
        "schema_version": "1.0.0",
        "run": {"run_id": "github-test", "input_generation": 3, "pipeline_version": "0.7.0", "model_timestamp": "2026-08-17T00:00:00Z", "parameters": {"max_tokens": 20000, "sequence_length": 96}},
        "scope": {"data_origin": "synthetic", "external_validation": False},
        "tasks": {"sentiment": task, "topic": topic},
        "reviews": reviews,
    }
    (run_path / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

    output = export_article_analysis(run_path)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {row["analysis_group"] for row in rows}
    assert {"aggregate_metrics", "model_delta", "segment_metrics", "segment_composition", "confusion_matrix", "classwise_metrics", "class_distribution", "confidence_accuracy", "calibration_bins", "error_summary", "error_record", "training_history", "validation_summary", "validation_incoming_gap", "run_metadata", "uncertainty"}.issubset(groups)
    assert any(row["metric"] == "specificity" for row in rows)
    assert any(row["analysis_group"] == "error_record" and row["record_id"] == "2" for row in rows)
    assert any(row["segment_dimension"] == "hasemoji" for row in rows)
    assert any(row["analysis_group"] == "uncertainty" and row["metric"] == "accuracy_ci95_low" for row in rows)
