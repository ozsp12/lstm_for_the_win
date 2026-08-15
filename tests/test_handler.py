from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import lstm_for_the_win.handler as handler_module
from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.handler import PipelineHandler


class _DummyModel:
    def save(self, path: Path) -> None:
        Path(path).write_text("dummy-model", encoding="utf-8")


class _DummyResult:
    def __init__(self, task: str, incoming_rows: list[dict[str, str]]) -> None:
        expected_field = f"expected_{task}"
        self.predictions = [
            {
                "ID": int(row["ID"]),
                "text": row["text"],
                "expected": row[expected_field],
                "predicted": row[expected_field],
                "confidence": 0.9,
                "correct": True,
                "linguistic_level": row["linguistic_level"],
                "flagprofanity": int(row["flagprofanity"]),
                "hasemoji": int(row["hasemoji"]),
                "hasspellingerror": int(row["hasspellingerror"]),
                "hasslang": int(row["hasslang"]),
                "length_class": row["length_class"],
                "mixed_sentiment": int(row["mixed_sentiment"]),
                "goldtest": int(row["goldtest"]),
                "input_timestamp": row["input_timestamp"],
            }
            for row in incoming_rows
        ]
        self.task = task
        self.size = len(incoming_rows)

    def to_dict(self) -> dict[str, object]:
        metrics = {"accuracy": 1.0}
        return {
            "task": self.task,
            "train_size": 120,
            "fit_size": 96,
            "validation_size": 24,
            "incoming_size": self.size,
            "labels": ["a", "b"],
            "label_counts": {},
            "metrics": metrics,
            "baseline_metrics": metrics,
            "metric_delta_vs_baseline": {"accuracy": 0.0},
            "segment_metrics": {},
            "history": {},
            "confusion_matrix": [],
            "predictions": self.predictions,
        }


def test_handler_publishes_atomic_run_and_advance_changes_only_input_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(
        initial_train_rows=120,
        incoming_rows=120,
        profanity_fraction=0.50,
        goldtest_fraction=0.50,
        validation_fraction=0.20,
    )
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")

    def fake_execute(pipeline_config):
        with Path(pipeline_config.incoming_path).open("r", encoding="utf-8", newline="") as file:
            incoming_rows = list(csv.DictReader(file))
        return SimpleNamespace(model=_DummyModel(), result=_DummyResult(pipeline_config.task, incoming_rows))

    monkeypatch.setattr(handler_module, "execute_pipeline", fake_execute)
    handler = PipelineHandler()
    run_path = handler.train_and_publish(
        input_dir,
        output_dir,
        run_id="unit-run",
        epochs=1,
        validation_fraction=0.20,
        patience=0,
    )

    assert (run_path / "predictions.csv").is_file()
    assert (run_path / "metrics.json").is_file()
    assert (run_path / "run_manifest.json").is_file()
    assert (run_path / "models" / "sentiment.keras").is_file()
    manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_generation"] == 0
    assert manifest["pipeline_version"] == "0.6.0"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    before_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    handler.generate_inputs(
        config_path,
        input_dir,
        mode="advance",
        input_timestamp="2026-08-16T12:00:00+00:00",
    )
    after_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    assert after_train_rows == before_train_rows + 60
    assert json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))["generation"] == 1
