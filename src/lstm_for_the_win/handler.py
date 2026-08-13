"""Command handler for synthetic data, model training, and artifact publication."""

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
from .classification import PipelineConfig, PipelineExecution, execute_pipeline


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
REVIEW_COLUMNS = {
    "ID",
    "text",
    "expected_sentiment",
    "expected_topic",
    "type",
    "input_timestamp",
}
SAMPLE_COLUMNS = {"ID", "text", "label", "type", "input_timestamp"}


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _suggested_action(sentiment: str, topic: str) -> str:
    team = topic.replace("_", " ").title()
    if sentiment == "negative":
        return f"Prioritize and route to the {team} support team."
    if sentiment == "positive":
        return f"Route to the {team} insights queue for advocacy analysis."
    return f"Add to the {team} monitoring queue."


def _validate_input_alignment(
    review_rows: list[dict[str, str]],
    sentiment_rows: list[dict[str, str]],
    topic_rows: list[dict[str, str]],
) -> None:
    """Guarantee that all three inputs are projections of the same review IDs."""

    if not review_rows or not REVIEW_COLUMNS.issubset(review_rows[0]):
        raise ValueError(f"reviews.csv must contain: {', '.join(sorted(REVIEW_COLUMNS))}.")
    for name, rows in (
        ("sentiment_samples.csv", sentiment_rows),
        ("topic_samples.csv", topic_rows),
    ):
        if not rows or not SAMPLE_COLUMNS.issubset(rows[0]):
            raise ValueError(f"{name} must contain: {', '.join(sorted(SAMPLE_COLUMNS))}.")
        if len(rows) != len(review_rows):
            raise ValueError(f"{name} must contain the same IDs as reviews.csv.")

    ids = [int(row["ID"]) for row in review_rows]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Review IDs must be unique and monotonically increasing.")

    for review, sentiment, topic in zip(
        review_rows,
        sentiment_rows,
        topic_rows,
        strict=True,
    ):
        shared = ("ID", "text", "type", "input_timestamp")
        if any(review[field] != sentiment[field] for field in shared):
            raise ValueError("Sentiment input is not aligned with reviews.csv.")
        if any(review[field] != topic[field] for field in shared):
            raise ValueError("Topic input is not aligned with reviews.csv.")
        if review["expected_sentiment"] != sentiment["label"]:
            raise ValueError("Sentiment labels are not aligned with reviews.csv.")
        if review["expected_topic"] != topic["label"]:
            raise ValueError("Topic labels are not aligned with reviews.csv.")


class PipelineHandler:
    """Single application boundary for all controlled pipeline operations."""

    def generate_inputs(
        self,
        config_path: str | Path,
        input_dir: str | Path,
        *,
        mode: str = "append",
        input_timestamp: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Initialize or append versioned synthetic input through the local agent."""

        config = SyntheticDataConfig.from_json(config_path)
        agent = SyntheticDataAgent(config)
        timestamp = input_timestamp or _default_timestamp()
        if mode == "initialize":
            return agent.initialize(input_dir, timestamp, overwrite=overwrite)
        if mode == "append":
            if overwrite:
                raise ValueError("overwrite is valid only when mode=initialize.")
            return agent.append(input_dir, timestamp)
        raise ValueError("mode must be initialize or append.")

    def train_and_publish(
        self,
        input_dir: str | Path,
        output_root: str | Path,
        *,
        run_id: str | None = None,
        epochs: int = 20,
        seed: int = 42,
    ) -> Path:
        """Train both models and atomically publish a complete run directory."""

        resolved_run_id = run_id or _default_run_id()
        if not RUN_ID_PATTERN.fullmatch(resolved_run_id):
            raise ValueError("run_id may contain only letters, numbers, dot, dash, and underscore.")
        if epochs < 1:
            raise ValueError("epochs must be positive.")

        input_path = Path(input_dir).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / resolved_run_id
        if final_path.exists():
            raise FileExistsError(f"Run already exists: {final_path}")

        sentiment_path = input_path / "sentiment_samples.csv"
        topic_path = input_path / "topic_samples.csv"
        reviews_path = input_path / "reviews.csv"
        review_rows = _read_csv(reviews_path)
        sentiment_rows = _read_csv(sentiment_path)
        topic_rows = _read_csv(topic_path)
        _validate_input_alignment(review_rows, sentiment_rows, topic_rows)

        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{resolved_run_id}-", dir=output_path)
        )
        model_timestamp = _default_timestamp()
        try:
            models_path = temporary_path / "models"
            models_path.mkdir()
            sentiment = execute_pipeline(
                PipelineConfig(
                    dataset_path=sentiment_path,
                    epochs=epochs,
                    seed=seed,
                )
            )
            topic = execute_pipeline(
                PipelineConfig(
                    dataset_path=topic_path,
                    epochs=epochs,
                    seed=seed,
                )
            )

            sentiment.model.save(models_path / "sentiment.keras")
            topic.model.save(models_path / "topic.keras")

            results = {
                "sentiment": sentiment.result.to_dict(),
                "topic": topic.result.to_dict(),
            }
            (temporary_path / "results.json").write_text(
                json.dumps(results, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._write_evaluation_predictions(
                temporary_path,
                sentiment,
                topic,
                model_timestamp,
            )
            self._write_predictions(
                temporary_path,
                review_rows,
                sentiment,
                topic,
                model_timestamp,
            )

            created_at = _default_timestamp()
            input_files = (sentiment_path, topic_path, reviews_path)
            input_timestamps = sorted({row["input_timestamp"] for row in review_rows})
            manifest = {
                "run_id": resolved_run_id,
                "created_at": created_at,
                "model_timestamp": model_timestamp,
                "input_timestamps": input_timestamps,
                "status": "complete",
                "pipeline_version": "0.4.0",
                "git_sha": os.getenv("GITHUB_SHA", "local"),
                "python_version": platform.python_version(),
                "tensorflow_version": tf.__version__,
                "parameters": {"epochs": epochs, "seed": seed},
                "input_files": {
                    path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                    for path in input_files
                },
                "outputs": {
                    "results": "results.json",
                    "predictions": "predictions.csv",
                    "evaluation_predictions": "evaluation_predictions.csv",
                    "sentiment_model": "models/sentiment.keras",
                    "topic_model": "models/topic.keras",
                },
            }
            (temporary_path / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_path.rename(final_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        (output_path / "latest.json").write_text(
            json.dumps({"run_id": resolved_run_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        return final_path

    @staticmethod
    def _write_evaluation_predictions(
        destination: Path,
        sentiment: PipelineExecution,
        topic: PipelineExecution,
        model_timestamp: str,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for task, execution in (("sentiment", sentiment), ("topic", topic)):
            rows.extend(
                {"task": task, **prediction, "model_timestamp": model_timestamp}
                for prediction in execution.result.predictions
            )
        _write_csv(
            destination / "evaluation_predictions.csv",
            rows,
            (
                "ID",
                "task",
                "text",
                "expected",
                "predicted",
                "confidence",
                "correct",
                "type",
                "input_timestamp",
                "model_timestamp",
            ),
        )

    @staticmethod
    def _write_predictions(
        destination: Path,
        review_rows: list[dict[str, str]],
        sentiment: PipelineExecution,
        topic: PipelineExecution,
        model_timestamp: str,
    ) -> None:
        sentiment_by_id = {
            str(prediction["ID"]): prediction
            for prediction in sentiment.result.predictions
        }
        topic_by_id = {
            str(prediction["ID"]): prediction
            for prediction in topic.result.predictions
        }
        test_rows = [row for row in review_rows if row["type"] == "test"]
        expected_ids = {row["ID"] for row in test_rows}
        if set(sentiment_by_id) != expected_ids or set(topic_by_id) != expected_ids:
            raise ValueError("Model predictions do not cover every versioned test ID.")

        rows = [
            {
                "ID": source["ID"],
                "text": source["text"],
                "expected_sentiment": source["expected_sentiment"],
                "expected_topic": source["expected_topic"],
                "predicted_sentiment": sentiment_by_id[source["ID"]]["predicted"],
                "predicted_topic": topic_by_id[source["ID"]]["predicted"],
                "type": "test",
                "input_timestamp": source["input_timestamp"],
                "model_timestamp": model_timestamp,
            }
            for source in test_rows
        ]
        _write_csv(
            destination / "predictions.csv",
            rows,
            (
                "ID",
                "text",
                "expected_sentiment",
                "expected_topic",
                "predicted_sentiment",
                "predicted_topic",
                "type",
                "input_timestamp",
                "model_timestamp",
            ),
        )


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="data/input")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("PIPELINE_EPOCHS", "20")))
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line contract for controlled execution."""

    parser = argparse.ArgumentParser(
        prog="lstm-pipeline",
        description="Version synthetic data, train both LSTM models, and publish artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-data", help="Initialize or append input data.")
    generate.add_argument("--config", default="config/synthetic_data.json")
    generate.add_argument("--input-dir", default="data/input")
    generate.add_argument("--mode", choices=("initialize", "append"), default="append")
    generate.add_argument("--data-timestamp", default=os.getenv("DATA_TIMESTAMP"))
    generate.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="Train using existing versioned input data.")
    _add_training_arguments(train)

    run = commands.add_parser(
        "run",
        help="Optionally append one batch, train both models, and publish artifacts.",
    )
    run.add_argument("--config", default="config/synthetic_data.json")
    run.add_argument("--append-data", action="store_true")
    run.add_argument("--data-timestamp", default=os.getenv("DATA_TIMESTAMP"))
    _add_training_arguments(run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one handler command and print a machine-readable result."""

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

    if arguments.command == "run" and arguments.append_data:
        handler.generate_inputs(
            arguments.config,
            arguments.input_dir,
            mode="append",
            input_timestamp=arguments.data_timestamp,
        )

    run_path = handler.train_and_publish(
        arguments.input_dir,
        arguments.output_root,
        run_id=arguments.run_id,
        epochs=arguments.epochs,
        seed=arguments.seed,
    )
    print(json.dumps({"status": "ok", "run_path": str(run_path.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
