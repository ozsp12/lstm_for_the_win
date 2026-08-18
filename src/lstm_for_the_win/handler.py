"""Command boundary for incremental data transitions, model training, and analysis publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import tensorflow as tf

from .agents import SyntheticDataAgent, SyntheticDataConfig
from .analysis import export_article_analysis
from .classification import PipelineConfig, PipelineExecution, execute_pipeline

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
PIPELINE_VERSION = "0.7.0"
LEGACY_TRAIN_COLUMNS = {
    "ID", "text", "sentiment", "topic", "linguistic_level", "flagprofanity",
    "source", "training_generation", "input_timestamp",
}
LEGACY_INCOMING_COLUMNS = {
    "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
    "flagprofanity", "goldtest", "input_timestamp",
}
RICH_STYLE_COLUMNS = {"hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment"}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _default_timestamp() -> str:
    return _utc_now().isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _validate_input_schema(train_rows: list[dict[str, str]], incoming_rows: list[dict[str, str]]) -> None:
    if not train_rows or not LEGACY_TRAIN_COLUMNS.issubset(train_rows[0]):
        raise ValueError("train.csv does not use the expected schema.")
    if not incoming_rows or not LEGACY_INCOMING_COLUMNS.issubset(incoming_rows[0]):
        raise ValueError("incoming.csv does not use the expected schema.")
    train_extra = set(train_rows[0]) - LEGACY_TRAIN_COLUMNS
    incoming_extra = set(incoming_rows[0]) - LEGACY_INCOMING_COLUMNS
    if train_extra and not RICH_STYLE_COLUMNS.issubset(train_extra):
        raise ValueError("train.csv contains an unsupported partial metadata schema.")
    if incoming_extra and not RICH_STYLE_COLUMNS.issubset(incoming_extra):
        raise ValueError("incoming.csv contains an unsupported partial metadata schema.")
    train_ids = {row["ID"] for row in train_rows}
    incoming_ids = {row["ID"] for row in incoming_rows}
    if train_ids & incoming_ids:
        raise ValueError("train.csv and incoming.csv must contain disjoint IDs.")
    train_text = {row["text"] for row in train_rows}
    incoming_text = {row["text"] for row in incoming_rows}
    if train_text & incoming_text:
        raise ValueError("train.csv and incoming.csv must contain disjoint text.")


def _wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * ((p * (1.0 - p) / total + z2 / (4.0 * total * total)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _previous_run_id(output_path: Path) -> str | None:
    latest = output_path / "latest.json"
    if not latest.is_file():
        return None
    try:
        value = json.loads(latest.read_text(encoding="utf-8")).get("run_id")
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, str) and RUN_ID_PATTERN.fullmatch(value) else None


def _task_payload(execution: PipelineExecution) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = execution.result.to_dict()
    predictions = {str(row["ID"]): row for row in payload.pop("predictions")}
    total = int(payload["incoming_size"])
    correct = sum(bool(row["correct"]) for row in predictions.values())
    low, high = _wilson_interval(correct, total)
    payload["uncertainty"] = {
        "accuracy_ci95": {
            "method": "wilson",
            "confidence_level": 0.95,
            "low": low,
            "high": high,
            "support": total,
        }
    }
    return payload, predictions


def _merged_reviews(
    incoming_rows: list[dict[str, str]],
    sentiment_predictions: dict[str, dict[str, Any]],
    topic_predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_ids = {row["ID"] for row in incoming_rows}
    if set(sentiment_predictions) != expected_ids or set(topic_predictions) != expected_ids:
        raise ValueError("Task predictions do not cover every incoming ID.")

    reviews: list[dict[str, Any]] = []
    for source in incoming_rows:
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
                "input_timestamp": source["input_timestamp"],
            }
        )
    return reviews


class PipelineHandler:
    """Single controlled boundary for all pipeline state changes."""

    def generate_inputs(
        self,
        config_path: str | Path,
        input_dir: str | Path,
        *,
        mode: str = "advance",
        input_timestamp: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        config = SyntheticDataConfig.from_json(config_path)
        agent = SyntheticDataAgent(config)
        timestamp = input_timestamp or _default_timestamp()
        if mode == "initialize":
            return agent.initialize(input_dir, timestamp, overwrite=overwrite)
        if mode == "advance":
            if overwrite:
                raise ValueError("overwrite is valid only when mode=initialize.")
            return agent.advance(input_dir, timestamp)
        raise ValueError("mode must be initialize or advance.")

    def train_and_publish(
        self,
        input_dir: str | Path,
        output_root: str | Path,
        *,
        run_id: str | None = None,
        epochs: int = 20,
        validation_fraction: float = 0.15,
        patience: int = 3,
        seed: int = 42,
    ) -> Path:
        resolved_run_id = run_id or _default_run_id()
        if not RUN_ID_PATTERN.fullmatch(resolved_run_id):
            raise ValueError("run_id may contain only letters, numbers, dot, dash, and underscore.")
        if epochs < 1 or patience < 0:
            raise ValueError("epochs must be positive and patience cannot be negative.")

        input_path = Path(input_dir).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / resolved_run_id
        if final_path.exists():
            raise FileExistsError(f"Run already exists: {final_path}")

        train_path = input_path / "train.csv"
        incoming_path = input_path / "incoming.csv"
        input_manifest_path = input_path / "input_manifest.json"
        train_rows = _read_csv(train_path)
        incoming_rows = _read_csv(incoming_path)
        _validate_input_schema(train_rows, incoming_rows)
        if not input_manifest_path.is_file():
            raise FileNotFoundError(f"Input manifest not found: {input_manifest_path}")
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        parent_run_id = _previous_run_id(output_path)

        temporary_path = Path(tempfile.mkdtemp(prefix=f".{resolved_run_id}-", dir=output_path))
        model_timestamp = _default_timestamp()
        try:
            executions: dict[str, PipelineExecution] = {}
            for task in ("sentiment", "topic"):
                executions[task] = execute_pipeline(
                    PipelineConfig(
                        train_path=train_path,
                        incoming_path=incoming_path,
                        task=task,
                        epochs=epochs,
                        validation_fraction=validation_fraction,
                        early_stopping_patience=patience,
                        seed=seed,
                    )
                )

            sentiment, sentiment_predictions = _task_payload(executions["sentiment"])
            topic, topic_predictions = _task_payload(executions["topic"])
            reviews = _merged_reviews(incoming_rows, sentiment_predictions, topic_predictions)
            input_files = (train_path, incoming_path, input_manifest_path)
            analysis = {
                "schema_version": "1.0.0",
                "run": {
                    "run_id": resolved_run_id,
                    "parent_run_id": parent_run_id,
                    "created_at": _default_timestamp(),
                    "model_timestamp": model_timestamp,
                    "status": "complete",
                    "pipeline_version": PIPELINE_VERSION,
                    "input_generation": int(input_manifest["generation"]),
                    "agent_version": input_manifest.get("agent_version"),
                    "git_sha": os.getenv("GITHUB_SHA", "local"),
                    "python_version": platform.python_version(),
                    "tensorflow_version": tf.__version__,
                    "parameters": {
                        "epochs": epochs,
                        "validation_fraction": validation_fraction,
                        "early_stopping_patience": patience,
                        "seed": seed,
                        "max_tokens": 20_000,
                        "sequence_length": 96,
                    },
                    "input_files": {
                        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                        for path in input_files
                    },
                },
                "scope": {
                    "data_origin": "synthetic",
                    "evaluation_split": "incoming",
                    "external_validation": False,
                    "generalization_claim": "controlled synthetic benchmark only",
                },
                "tasks": {"sentiment": sentiment, "topic": topic},
                "reviews": reviews,
            }
            (temporary_path / "analysis.json").write_text(
                json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            export_article_analysis(temporary_path)
            temporary_path.rename(final_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        (output_path / "latest.json").write_text(
            json.dumps({"run_id": resolved_run_id}, indent=2) + "\n", encoding="utf-8"
        )
        return final_path


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="data/input")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("PIPELINE_EPOCHS", "20")))
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--patience", type=int, default=int(os.getenv("PIPELINE_PATIENCE", "3")))
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lstm-pipeline",
        description="Train LSTM classifiers on train.csv and evaluate new incoming reviews.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-data", help="Initialize or advance incremental input data.")
    generate.add_argument("--config", default="config/synthetic_data.json")
    generate.add_argument("--input-dir", default="data/input")
    generate.add_argument("--mode", choices=("initialize", "advance"), default="advance")
    generate.add_argument("--data-timestamp", default=os.getenv("DATA_TIMESTAMP"))
    generate.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="Train and evaluate using the current input state.")
    _add_training_arguments(train)

    run = commands.add_parser(
        "run",
        help="Train, evaluate incoming reviews, then optionally promote goldtest and refresh incoming.",
    )
    run.add_argument("--config", default="config/synthetic_data.json")
    run.add_argument("--advance-data", action="store_true")
    run.add_argument("--data-timestamp", default=os.getenv("DATA_TIMESTAMP"))
    _add_training_arguments(run)
    return parser


def _resolve_validation_fraction(arguments: argparse.Namespace) -> float:
    if arguments.validation_fraction is not None:
        return float(arguments.validation_fraction)
    env_value = os.getenv("PIPELINE_VALIDATION_FRACTION")
    if env_value is not None:
        return float(env_value)
    if arguments.command != "run":
        return 0.15
    config = SyntheticDataConfig.from_json(arguments.config)
    manifest_path = Path(arguments.input_dir) / "input_manifest.json"
    generation = 0
    if manifest_path.is_file():
        generation = int(json.loads(manifest_path.read_text(encoding="utf-8"))["generation"])
    return float(config.effective_generation(generation)["validation_fraction"])


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    handler = PipelineHandler()

    if arguments.command == "generate-data":
        manifest = handler.generate_inputs(
            arguments.config,
            arguments.input_dir,
            mode=arguments.mode,
            input_timestamp=arguments.data_timestamp,
            overwrite=arguments.overwrite,
        )
        print(json.dumps({"status": "ok", "input_manifest": str(manifest.resolve())}))
        return 0

    validation_fraction = _resolve_validation_fraction(arguments)
    run_path = handler.train_and_publish(
        arguments.input_dir,
        arguments.output_root,
        run_id=arguments.run_id,
        epochs=arguments.epochs,
        validation_fraction=validation_fraction,
        patience=arguments.patience,
        seed=arguments.seed,
    )

    response: dict[str, Any] = {
        "status": "ok",
        "run_path": str(run_path.resolve()),
        "validation_fraction": validation_fraction,
    }
    if arguments.command == "run" and arguments.advance_data:
        manifest = handler.generate_inputs(
            arguments.config,
            arguments.input_dir,
            mode="advance",
            input_timestamp=arguments.data_timestamp,
        )
        response["next_input_manifest"] = str(manifest.resolve())
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
