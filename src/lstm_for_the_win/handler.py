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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _default_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


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


class PipelineHandler:
    """Single application boundary for all controlled pipeline operations."""

    def generate_inputs(
        self,
        config_path: str | Path,
        input_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Run the configured synthetic-data agent."""

        config = SyntheticDataConfig.from_json(config_path)
        return SyntheticDataAgent(config).write(input_dir, overwrite=overwrite)

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
        required_review_columns = {"text", "expected_sentiment", "expected_topic"}
        if not review_rows or not required_review_columns.issubset(review_rows[0]):
            raise ValueError(
                "reviews.csv must contain text, expected_sentiment, and expected_topic."
            )
        demo_texts = tuple(row["text"] for row in review_rows)

        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{resolved_run_id}-", dir=output_path)
        )
        try:
            models_path = temporary_path / "models"
            models_path.mkdir()
            sentiment = execute_pipeline(
                PipelineConfig(
                    dataset_path=sentiment_path,
                    epochs=epochs,
                    seed=seed,
                    demo_texts=demo_texts,
                )
            )
            topic = execute_pipeline(
                PipelineConfig(
                    dataset_path=topic_path,
                    epochs=epochs,
                    seed=seed,
                    demo_texts=demo_texts,
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
            self._write_evaluation_predictions(temporary_path, sentiment, topic)
            self._write_inference_predictions(
                temporary_path,
                review_rows,
                sentiment,
                topic,
            )

            created_at = _utc_now().isoformat()
            input_files = (sentiment_path, topic_path, reviews_path)
            manifest = {
                "run_id": resolved_run_id,
                "created_at": created_at,
                "status": "complete",
                "pipeline_version": "0.3.0",
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
                    "evaluation_predictions": "evaluation_predictions.csv",
                    "inference_predictions": "inference_predictions.csv",
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
    ) -> None:
        rows: list[dict[str, Any]] = []
        for task, execution in (("sentiment", sentiment), ("topic", topic)):
            rows.extend(
                {"task": task, **prediction}
                for prediction in execution.result.predictions
            )
        _write_csv(
            destination / "evaluation_predictions.csv",
            rows,
            ("task", "text", "expected", "predicted", "confidence", "correct"),
        )

    @staticmethod
    def _write_inference_predictions(
        destination: Path,
        review_rows: list[dict[str, str]],
        sentiment: PipelineExecution,
        topic: PipelineExecution,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for source, sentiment_prediction, topic_prediction in zip(
            review_rows,
            sentiment.result.demo_predictions,
            topic.result.demo_predictions,
            strict=True,
        ):
            predicted_sentiment = str(sentiment_prediction["predicted"])
            predicted_topic = str(topic_prediction["predicted"])
            rows.append(
                {
                    "text": source["text"],
                    "expected_sentiment": source["expected_sentiment"],
                    "sentiment": predicted_sentiment,
                    "sentiment_confidence": sentiment_prediction["confidence"],
                    "sentiment_correct": predicted_sentiment == source["expected_sentiment"],
                    "expected_topic": source["expected_topic"],
                    "topic": predicted_topic,
                    "topic_confidence": topic_prediction["confidence"],
                    "topic_correct": predicted_topic == source["expected_topic"],
                    "suggested_action": _suggested_action(
                        predicted_sentiment,
                        predicted_topic,
                    ),
                }
            )
        _write_csv(
            destination / "inference_predictions.csv",
            rows,
            (
                "text",
                "expected_sentiment",
                "sentiment",
                "sentiment_confidence",
                "sentiment_correct",
                "expected_topic",
                "topic",
                "topic_confidence",
                "topic_correct",
                "suggested_action",
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
        description="Generate data, train both LSTM models, and publish run artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-data", help="Generate synthetic input data.")
    generate.add_argument("--config", default="config/synthetic_data.json")
    generate.add_argument("--input-dir", default="data/input")
    generate.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train", help="Train using existing input data.")
    _add_training_arguments(train)

    run = commands.add_parser(
        "run",
        help="Regenerate synthetic inputs, train both models, and publish artifacts.",
    )
    run.add_argument("--config", default="config/synthetic_data.json")
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
            overwrite=arguments.overwrite,
        )
        print(json.dumps({"status": "ok", "input_manifest": str(manifest.resolve())}))
        return 0

    if arguments.command == "run":
        handler.generate_inputs(
            arguments.config,
            arguments.input_dir,
            overwrite=True,
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
