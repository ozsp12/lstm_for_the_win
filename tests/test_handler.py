from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import lstm_for_the_win.handler as handler_module
from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.handler import PipelineHandler


class _DummyResult:
    def __init__(self, task: str, incoming_rows: list[dict[str, str]]) -> None:
        expected_field = f"expected_{task}"
        self.predictions = [
            {
                "ID": int(row["ID"]), "text": row["text"], "expected": row[expected_field],
                "predicted": row[expected_field], "confidence": 0.9, "correct": True,
                "linguistic_level": row["linguistic_level"], "flagprofanity": int(row["flagprofanity"]),
                "hasemoji": int(row["hasemoji"]), "hasspellingerror": int(row["hasspellingerror"]),
                "hasslang": int(row["hasslang"]), "length_class": row["length_class"],
                "mixed_sentiment": int(row["mixed_sentiment"]), "goldtest": int(row["goldtest"]),
                "input_timestamp": row["input_timestamp"],
            }
            for row in incoming_rows
        ]
        self.task = task
        self.size = len(incoming_rows)

    def to_dict(self) -> dict[str, object]:
        metrics = {
            "accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0,
            "macro_f1": 1.0, "weighted_f1": 1.0, "log_loss": 0.1,
            "brier_score": 0.05, "expected_calibration_error": 0.02,
        }
        return {
            "task": self.task, "train_size": 1200, "fit_size": 960, "validation_size": 240,
            "incoming_size": self.size, "labels": ["a", "b"], "label_counts": {},
            "metrics": metrics, "baseline_metrics": metrics,
            "metric_delta_vs_baseline": {"accuracy": 0.0}, "segment_metrics": {},
            "history": {"accuracy": [1.0], "loss": [0.1], "val_accuracy": [1.0], "val_loss": [0.1]},
            "confusion_matrix": [], "predictions": self.predictions,
        }


def test_handler_publishes_lean_atomic_run_and_advance_is_incremental(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(
        initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0,
        profanity_fraction=0.50, goldtest_fraction=0.50, validation_fraction=0.20,
        vary_counts=False,
    )
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")

    def fake_execute(pipeline_config):
        with Path(pipeline_config.incoming_path).open("r", encoding="utf-8", newline="") as file:
            incoming_rows = list(csv.DictReader(file))
        return SimpleNamespace(result=_DummyResult(pipeline_config.task, incoming_rows))

    monkeypatch.setattr(handler_module, "execute_pipeline", fake_execute)
    handler = PipelineHandler()
    run_path = handler.train_and_publish(input_dir, output_dir, run_id="unit-run", epochs=1, validation_fraction=0.20, patience=0)

    assert (run_path / "analysis.json").is_file()
    assert (run_path / "article_analysis.csv").is_file()
    assert not (run_path / "predictions.csv").exists()
    assert not (run_path / "metrics.json").exists()
    assert not (run_path / "results.json").exists()
    assert not (run_path / "run_manifest.json").exists()
    assert not (run_path / "figures").exists()
    assert not (run_path / "models").exists()

    analysis = json.loads((run_path / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["run"]["input_generation"] == 0
    assert analysis["run"]["pipeline_version"] == "0.7.0"
    assert analysis["scope"]["external_validation"] is False
    assert analysis["tasks"]["sentiment"]["uncertainty"]["accuracy_ci95"]["support"] == 1200
    assert json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))["run_id"] == "unit-run"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    before_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    handler.generate_inputs(config_path, input_dir, mode="advance", input_timestamp="2026-08-16T12:00:00+00:00")
    after_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    assert after_train_rows == before_train_rows + 600
    assert json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))["generation"] == 1


def test_handler_links_runs_without_deleting_previous_runs(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, vary_counts=False)
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")

    def fake_execute(pipeline_config):
        with Path(pipeline_config.incoming_path).open("r", encoding="utf-8", newline="") as file:
            incoming_rows = list(csv.DictReader(file))
        return SimpleNamespace(result=_DummyResult(pipeline_config.task, incoming_rows))

    monkeypatch.setattr(handler_module, "execute_pipeline", fake_execute)
    handler = PipelineHandler()
    first = handler.train_and_publish(input_dir, output_dir, run_id="run-1", epochs=1, patience=0)
    second = handler.train_and_publish(input_dir, output_dir, run_id="run-2", epochs=1, patience=0)
    assert first.is_dir() and second.is_dir()
    second_analysis = json.loads((second / "analysis.json").read_text(encoding="utf-8"))
    assert second_analysis["run"]["parent_run_id"] == "run-1"
