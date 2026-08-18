from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import lstm_for_the_win.experiment as experiment_module
from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.handler import PipelineHandler


class _DummyResult:
    def __init__(self, task: str, incoming_rows: list[dict[str, str]], seed: int) -> None:
        expected_field = f"expected_{task}"
        self.labels = sorted({row[expected_field] for row in incoming_rows})
        self.predictions = [
            {
                "ID": int(row["ID"]), "text": row["text"], "expected": row[expected_field],
                "predicted": row[expected_field], "confidence": 0.9, "correct": True,
                "linguistic_level": row["linguistic_level"], "flagprofanity": int(row["flagprofanity"]),
                "hasemoji": int(row["hasemoji"]), "hasspellingerror": int(row["hasspellingerror"]),
                "hasslang": int(row["hasslang"]), "length_class": row["length_class"],
                "mixed_sentiment": int(row["mixed_sentiment"]), "goldtest": int(row["goldtest"]),
                "template_family": row["template_family"], "input_timestamp": row["input_timestamp"],
            }
            for row in incoming_rows
        ]
        self.task = task
        self.seed = seed
        self.incoming_size = len(incoming_rows)
        self.metrics = {
            "accuracy": 1.0, "precision_macro": 1.0, "recall_macro": 1.0,
            "macro_f1": 1.0, "weighted_f1": 1.0, "log_loss": 0.1,
            "brier_score": 0.05, "expected_calibration_error": 0.02,
        }
        self.baseline_metrics = dict(self.metrics)
        self.metric_delta_vs_baseline = {key: 0.0 for key in ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score")}
        self.paired_comparison = {
            "method": "mcnemar_exact_two_sided", "both_correct": len(incoming_rows),
            "lstm_only_correct": 0, "baseline_only_correct": 0, "neither_correct": 0,
            "discordant_pairs": 0, "p_value": 1.0,
        }
        counts = {label: sum(row[expected_field] == label for row in incoming_rows) for label in self.labels}
        self.confusion_matrix = [
            [counts[label] if left == right else 0 for right in self.labels]
            for left, label in enumerate(self.labels)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task, "seed": self.seed,
            "train_size": 1200, "fit_size": 960, "validation_size": 240,
            "incoming_size": self.incoming_size, "labels": self.labels, "label_counts": {},
            "validation_split": {
                "method": "template_family_grouped", "family_source": "persisted_metadata",
                "heldout_families": ["using"],
            },
            "metrics": self.metrics, "baseline_metrics": self.baseline_metrics,
            "metric_delta_vs_baseline": self.metric_delta_vs_baseline,
            "paired_comparison": self.paired_comparison, "segment_metrics": {},
            "history": {"accuracy": [1.0], "loss": [0.1], "val_accuracy": [1.0], "val_loss": [0.1]},
            "confusion_matrix": self.confusion_matrix, "predictions": self.predictions,
        }


def _patch_training(monkeypatch, tmp_path: Path) -> None:
    def fake_execute(pipeline_config):
        with Path(pipeline_config.incoming_path).open("r", encoding="utf-8", newline="") as file:
            incoming_rows = list(csv.DictReader(file))
        return SimpleNamespace(result=_DummyResult(pipeline_config.task, incoming_rows, pipeline_config.seed), model=object())

    def fake_external(root):
        directory = Path(root) / "uci_sentiment_labelled_sentences"
        directory.mkdir(parents=True, exist_ok=True)
        data = directory / "amazon_cells_labelled.tsv"
        data.write_text("good phone\t1\nbad phone\t0\n", encoding="utf-8")
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"dataset_doi": "test", "license": "test"}), encoding="utf-8")
        return data, {"dataset_doi": "test", "license": "test"}

    def fake_external_evaluation(execution, external_path, manifest):
        labels = list(execution.result.labels)
        size = len(labels)
        matrix = [[0 for _ in range(size)] for _ in range(size)]
        matrix[0][0] = 1
        matrix[-1][-1] = 1
        return {
            "immutable": True, "real_world": True, "task": "sentiment", "support": 2,
            "dataset": dict(manifest), "metrics": {"accuracy": 1.0}, "reviews": [],
            "labels_in_source": [labels[0], labels[-1]], "model_label_space": labels,
            "confusion_matrix": matrix,
        }

    monkeypatch.setattr(experiment_module, "execute_pipeline", fake_execute)
    monkeypatch.setattr(
        experiment_module,
        "evaluate_benchmark",
        lambda executions, benchmark_path, benchmark_rows, provenance=None: {
            "immutable": True,
            "source": "unit-test",
            "provenance": dict(provenance or {}),
            "tasks": {},
            "reviews": [],
        },
    )
    monkeypatch.setattr(experiment_module, "ensure_external_sentiment_benchmark", fake_external)
    monkeypatch.setattr(experiment_module, "evaluate_external_sentiment", fake_external_evaluation)


def test_handler_publishes_atomic_run_bundle_and_advance_is_incremental(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(
        initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0,
        profanity_fraction=0.50, goldtest_fraction=0.50, validation_fraction=0.20,
        vary_counts=False,
    )
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")
    _patch_training(monkeypatch, tmp_path)

    handler = PipelineHandler()
    run_path = handler.train_and_publish(
        input_dir, output_dir, run_id="unit-run", epochs=1,
        validation_fraction=0.20, patience=0,
    )

    assert (run_path / "run.json").is_file()
    assert (run_path / "article_analysis.csv").is_file()
    figures = sorted((run_path / "figures").glob("*.svg"))
    assert len(figures) >= 4
    assert not (run_path / "analysis.json").exists()
    assert not (run_path / "predictions.csv").exists()
    assert not (run_path / "metrics.json").exists()
    assert not (run_path / "results.json").exists()
    assert not (run_path / "run_manifest.json").exists()
    assert not (run_path / "models").exists()
    assert (input_dir / "benchmark.csv").is_file()
    assert (input_dir / "benchmark_manifest.json").is_file()

    with (run_path / "article_analysis.csv").open(encoding="utf-8", newline="") as handle:
        article_rows = list(csv.DictReader(handle))
    assert sum(row["analysis_group"] == "prediction_record" for row in article_rows) == 1200

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert run["schema_version"] == "2.0.0"
    assert run["artifact_type"] == "experiment_run"
    assert run["run"]["input_generation"] == 0
    assert run["run"]["pipeline_version"] == "0.10.0"
    assert run["scope"]["external_validation"] is True
    assert run["scope"]["external_validation_tasks"] == ["sentiment"]
    assert run["scope"]["topic_external_validation"] is False
    assert run["scope"]["immutable_benchmark"] is True
    assert run["external_validation"]["real_world"] is True
    assert run["tasks"]["sentiment"]["uncertainty"]["accuracy_ci95"]["support"] == 1200
    assert run["tasks"]["sentiment"]["replicates"]["count"] == 1
    assert json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))["run_id"] == "unit-run"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    before_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    before_incoming = (input_dir / "incoming.csv").read_bytes()
    handler.generate_inputs(config_path, input_dir, mode="advance", input_timestamp="2026-08-16T12:00:00+00:00")
    after_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    assert after_train_rows == before_train_rows + 600
    assert (input_dir / "incoming.csv").read_bytes() != before_incoming
    assert json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))["generation"] == 1
    with (input_dir / "incoming.csv").open(encoding="utf-8", newline="") as handle:
        assert "template_family" in next(csv.reader(handle))


def test_handler_links_runs_before_workflow_retention(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, vary_counts=False)
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")
    _patch_training(monkeypatch, tmp_path)

    handler = PipelineHandler()
    first = handler.train_and_publish(input_dir, output_dir, run_id="run-1", epochs=1, patience=0)
    second = handler.train_and_publish(input_dir, output_dir, run_id="run-2", epochs=1, patience=0)
    assert first.is_dir() and second.is_dir()
    second_run = json.loads((second / "run.json").read_text(encoding="utf-8"))
    assert second_run["run"]["parent_run_id"] == "run-1"
