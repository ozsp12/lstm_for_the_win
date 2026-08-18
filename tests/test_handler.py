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
        bins = [
            {"index": index, "lower": index / 10, "upper": (index + 1) / 10, "support": self.incoming_size if index == 9 else 0, "mean_confidence": 0.9 if index == 9 else 0.0, "accuracy": 1.0 if index == 9 else 0.0, "absolute_gap": 0.1 if index == 9 else 0.0}
            for index in range(10)
        ]
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
            "paired_comparison": self.paired_comparison,
            "segment_metrics": {},
            "calibration_bins": bins,
            "baseline_calibration_bins": bins,
            "history": {"accuracy": [1.0], "loss": [0.1], "val_accuracy": [1.0], "val_loss": [0.1]},
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_contract": {
                "axis_convention": "rows_expected_columns_predicted",
                "expected_labels": self.labels,
                "predicted_labels": self.labels,
            },
            "predictions": self.predictions,
        }


def _patch_training(monkeypatch) -> None:
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
        source_labels = [labels[0], labels[-1]]
        confusion = {expected: {predicted: 0 for predicted in labels} for expected in source_labels}
        confusion[source_labels[0]][source_labels[0]] = 1
        confusion[source_labels[1]][source_labels[1]] = 1
        return {
            "immutable": True, "real_world": True, "task": "sentiment", "support": 2,
            "dataset": dict(manifest), "labels_in_source": source_labels, "model_label_space": labels,
            "accuracy": 1.0, "full_label_space_accuracy": 1.0, "binary_restricted_accuracy": 1.0,
            "source_label_metrics": {"labels": source_labels, "precision_macro": 1.0, "recall_macro": 1.0, "macro_f1": 1.0, "per_class": {}},
            "probabilistic_metrics": {"log_loss": 0.1, "brier_score": 0.05, "expected_calibration_error": 0.02},
            "binary_restricted_probabilistic_metrics": {"log_loss": 0.1, "brier_score": 0.05, "expected_calibration_error": 0.02},
            "neutral_prediction_rate": 0.0,
            "confusion_matrix": {
                "axis_convention": "rows_expected_columns_predicted",
                "expected_labels": source_labels, "predicted_labels": labels, "matrix": confusion,
            },
            "binary_restricted_confusion_matrix": {
                "axis_convention": "rows_expected_columns_predicted",
                "expected_labels": source_labels, "predicted_labels": source_labels,
                "matrix": {source_labels[0]: {source_labels[0]: 1, source_labels[1]: 0}, source_labels[1]: {source_labels[0]: 0, source_labels[1]: 1}},
            },
            "uncertainty": {
                "full_label_space_accuracy_ci95": {"method": "wilson", "low": 0.34, "high": 1.0, "support": 2},
                "binary_restricted_accuracy_ci95": {"method": "wilson", "low": 0.34, "high": 1.0, "support": 2},
            },
            "reviews": [],
        }

    monkeypatch.setattr(experiment_module, "execute_pipeline", fake_execute)
    monkeypatch.setattr(experiment_module, "evaluate_benchmark", lambda executions, benchmark_path, benchmark_rows, provenance=None: {"immutable": True, "source": "unit-test", "provenance": dict(provenance or {}), "tasks": {}, "reviews": []})
    monkeypatch.setattr(experiment_module, "ensure_external_sentiment_benchmark", fake_external)
    monkeypatch.setattr(experiment_module, "evaluate_external_sentiment", fake_external_evaluation)


def test_handler_publishes_wide_atomic_bundle_and_advance_is_incremental(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, profanity_fraction=0.50, goldtest_fraction=0.50, validation_fraction=0.20, vary_counts=False)
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")
    _patch_training(monkeypatch)

    handler = PipelineHandler()
    run_path = handler.train_and_publish(input_dir, output_dir, run_id="unit-run", epochs=1, validation_fraction=0.20, patience=0)
    assert (run_path / "run.json").is_file()
    assert (run_path / "article_analysis.csv").is_file()
    assert len(list((run_path / "figures").glob("*.svg"))) >= 4
    assert (input_dir / "benchmark.csv").is_file()

    with (run_path / "article_analysis.csv").open(encoding="utf-8", newline="") as handle:
        article_rows = list(csv.DictReader(handle))
    assert len(article_rows) == 1200
    assert "analysis_group" not in article_rows[0]
    assert {"review_id", "sentiment_lstm_accuracy", "topic_lstm_accuracy"}.issubset(article_rows[0])

    run = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
    assert run["run"]["input_generation"] == 0
    assert run["run"]["pipeline_version"] == "0.11.0"
    assert run["run"]["determinism"]["tensorflow_op_determinism"] is True
    assert run["tasks"]["sentiment"]["uncertainty"]["primary_seed_accuracy_ci95"]["support"] == 1200
    assert run["tasks"]["sentiment"]["canonical_estimate"]["method"] == "mean_across_model_seeds"
    assert run["external_validation"]["binary_restricted_accuracy"] == 1.0
    assert json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))["run_id"] == "unit-run"

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    with (input_dir / "incoming.csv").open(encoding="utf-8", newline="") as handle:
        before_incoming_rows = list(csv.DictReader(handle))
    promoted_count = sum(row["goldtest"] == "1" for row in before_incoming_rows)
    before_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    before_incoming = (input_dir / "incoming.csv").read_bytes()
    handler.generate_inputs(config_path, input_dir, mode="advance", input_timestamp="2026-08-16T12:00:00+00:00")
    after_train_rows = sum(1 for _ in (input_dir / "train.csv").open(encoding="utf-8")) - 1
    assert after_train_rows == before_train_rows + promoted_count
    assert (input_dir / "incoming.csv").read_bytes() != before_incoming
    assert json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))["generation"] == 1


def test_handler_links_runs_before_workflow_retention(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, vary_counts=False)
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")
    _patch_training(monkeypatch)
    handler = PipelineHandler()
    first = handler.train_and_publish(input_dir, output_dir, run_id="run-1", epochs=1, patience=0)
    second = handler.train_and_publish(input_dir, output_dir, run_id="run-2", epochs=1, patience=0)
    assert first.is_dir() and second.is_dir()
    assert json.loads((second / "run.json").read_text(encoding="utf-8"))["run"]["parent_run_id"] == "run-1"
