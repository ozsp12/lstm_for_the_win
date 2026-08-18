from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lstm_for_the_win.run_artifact as artifact


def _raw_rows() -> list[dict[str, str]]:
    base = {
        "linguistic_level": "standard",
        "flagprofanity": "0",
        "hasemoji": "0",
        "hasspellingerror": "0",
        "hasslang": "0",
        "length_class": "short",
        "mixed_sentiment": "0",
        "goldtest": "0",
        "template_family": "using",
        "input_timestamp": "2026-08-18T12:00:00+00:00",
    }
    return [
        {"ID": "1", "text": "bad phone", "expected_sentiment": "negative", "expected_topic": "phone", **base},
        {"ID": "2", "text": "good tv", "expected_sentiment": "positive", "expected_topic": "tv", **base},
    ]


def _prediction(identifier: int, label: str, correct: bool = True) -> dict[str, object]:
    return {"ID": identifier, "predicted": label, "confidence": 0.9, "correct": correct}


def test_wilson_merge_and_replicate_summary() -> None:
    assert artifact.wilson_interval(0, 0)["low"] == 0.0
    interval = artifact.wilson_interval(8, 10)
    assert 0.0 < interval["low"] < interval["high"] <= 1.0

    rows = _raw_rows()
    sentiment = {"1": _prediction(1, "negative"), "2": _prediction(2, "positive")}
    topic = {"1": _prediction(1, "phone"), "2": _prediction(2, "tv")}
    merged = artifact.merge_reviews(rows, sentiment, topic)
    assert merged[0]["predicted_sentiment"] == "negative"
    assert merged[1]["predicted_topic"] == "tv"
    with pytest.raises(ValueError):
        artifact.merge_reviews(rows, {"1": sentiment["1"]}, topic)

    metrics1 = {name: 0.8 for name in artifact.COMPARABLE_METRICS}
    metrics2 = {name: 0.9 for name in artifact.COMPARABLE_METRICS}
    summary = artifact.summarize_replicates([
        SimpleNamespace(seed=42, metrics=metrics1, baseline_metrics=metrics1),
        SimpleNamespace(seed=43, metrics=metrics2, baseline_metrics=metrics2),
    ])
    assert summary["count"] == 2
    assert summary["seeds"] == [42, 43]
    assert summary["metrics"]["accuracy"]["mean"] == pytest.approx(0.85)
    assert summary["metrics"]["accuracy"]["mean_ci95"]["method"] == "student_t_across_model_seeds"
    assert artifact.summarize_replicates([])["count"] == 0


def test_evaluate_synthetic_benchmark(monkeypatch, tmp_path: Path) -> None:
    records = [
        SimpleNamespace(ID=1, text="bad phone", sentiment="negative", topic="phone", hasemoji=0, hasspellingerror=0, hasslang=0, length_class="short", mixed_sentiment=0),
        SimpleNamespace(ID=2, text="good tv", sentiment="positive", topic="tv", hasemoji=0, hasspellingerror=0, hasslang=0, length_class="short", mixed_sentiment=0),
    ]
    monkeypatch.setattr(artifact, "load_incoming", lambda path: records)

    def probabilities(model, texts):
        assert len(texts) == 2
        return np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)

    monkeypatch.setattr(artifact, "predict_probabilities", probabilities)
    executions = {
        "sentiment": SimpleNamespace(model="sent", result=SimpleNamespace(labels=["negative", "positive"])),
        "topic": SimpleNamespace(model="topic", result=SimpleNamespace(labels=["phone", "tv"])),
    }
    result = artifact.evaluate_benchmark(executions, tmp_path / "benchmark.csv", _raw_rows(), provenance={"source_generation": 0})
    assert result["immutable"] is True
    assert result["provenance"]["source_generation"] == 0
    assert result["tasks"]["sentiment"]["metrics"]["accuracy"] == 1.0
    assert result["tasks"]["topic"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert len(result["reviews"]) == 2

    bad_records = [SimpleNamespace(**{**vars(records[0]), "sentiment": "neutral"})]
    monkeypatch.setattr(artifact, "load_incoming", lambda path: bad_records)
    with pytest.raises(ValueError):
        artifact.evaluate_benchmark(executions, tmp_path / "benchmark.csv", _raw_rows()[:1])


def test_evaluate_external_sentiment_with_neutral_prediction(monkeypatch, tmp_path: Path) -> None:
    rows = [
        {"ID": 1, "text": "bad", "expected_sentiment": "negative"},
        {"ID": 2, "text": "mixed", "expected_sentiment": "positive"},
        {"ID": 3, "text": "good", "expected_sentiment": "positive"},
    ]
    monkeypatch.setattr(artifact, "load_external_sentiment", lambda path: rows)
    monkeypatch.setattr(
        artifact,
        "predict_probabilities",
        lambda model, texts: np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
        ], dtype=float),
    )
    execution = SimpleNamespace(model=object(), result=SimpleNamespace(labels=["negative", "neutral", "positive"]))
    result = artifact.evaluate_external_sentiment(execution, tmp_path / "external.tsv", {"license": "CC BY 4.0"})
    assert result["real_world"] is True
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["neutral_prediction_rate"] == pytest.approx(1 / 3)
    assert result["confusion_matrix"]["matrix"]["positive"]["neutral"] == 1
    assert result["source_label_metrics"]["per_class"]["positive"]["support"] == 2
    assert len(result["reviews"]) == 3

    monkeypatch.setattr(artifact, "load_external_sentiment", lambda path: [{"ID": 1, "text": "x", "expected_sentiment": "other"}])
    with pytest.raises(ValueError):
        artifact.evaluate_external_sentiment(execution, tmp_path / "external.tsv", {})


class _Result:
    def __init__(self, task: str, labels: list[str], predictions: list[dict[str, object]], seed: int = 42) -> None:
        self.task = task
        self.labels = labels
        self.seed = seed
        self.metrics = {name: 0.9 for name in artifact.COMPARABLE_METRICS}
        self.baseline_metrics = {name: 0.8 for name in artifact.COMPARABLE_METRICS}
        self._predictions = predictions

    def to_dict(self):
        return {
            "task": self.task,
            "seed": self.seed,
            "incoming_size": len(self._predictions),
            "labels": self.labels,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "metric_delta_vs_baseline": {name: 0.1 for name in artifact.COMPARABLE_METRICS},
            "predictions": self._predictions,
        }


def test_build_and_write_run_document(tmp_path: Path) -> None:
    rows = _raw_rows()
    sent_predictions = [
        {**_prediction(1, "negative"), "hasemoji": 0},
        {**_prediction(2, "positive"), "hasemoji": 0},
    ]
    topic_predictions = [_prediction(1, "phone"), _prediction(2, "tv")]
    executions = {
        "sentiment": SimpleNamespace(result=_Result("sentiment", ["negative", "positive"], sent_predictions)),
        "topic": SimpleNamespace(result=_Result("topic", ["phone", "tv"], topic_predictions)),
    }
    input_file = tmp_path / "input.txt"
    input_file.write_text("abc", encoding="utf-8")
    replicate = SimpleNamespace(
        seed=42,
        metrics={name: 0.9 for name in artifact.COMPARABLE_METRICS},
        baseline_metrics={name: 0.8 for name in artifact.COMPARABLE_METRICS},
    )
    document = artifact.build_run_document(
        run_metadata={"run_id": "r1", "input_generation": 0},
        scope={"data_origin": "synthetic"},
        executions=executions,
        incoming_rows=rows,
        input_files=[input_file],
        replicate_results={"sentiment": [replicate], "topic": [replicate]},
        benchmark={"immutable": True},
        external_validation={"real_world": True},
    )
    assert document["artifact_type"] == "experiment_run"
    assert document["run"]["input_files"]["input.txt"]["sha256"] == artifact.sha256(input_file)
    assert document["tasks"]["sentiment"]["uncertainty"]["accuracy_ci95"]["support"] == 2
    assert document["tasks"]["topic"]["replicates"]["count"] == 1
    assert document["benchmark"]["immutable"] is True
    assert len(document["reviews"]) == 2

    destination = tmp_path / "run"
    destination.mkdir()
    path = artifact.write_run_json(destination, document)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run"]["run_id"] == "r1"
