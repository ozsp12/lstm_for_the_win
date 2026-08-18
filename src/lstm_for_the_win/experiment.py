"""Experiment orchestration separated from the CLI/state-transition handler."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

import tensorflow as tf

from .benchmark import ensure_immutable_benchmark
from .classification import PipelineConfig, PipelineExecution, PipelineResult, execute_pipeline
from .derived_artifacts import materialize_derived_artifacts
from .external_benchmark import ensure_external_sentiment_benchmark
from .run_artifact import build_run_document, evaluate_benchmark, evaluate_external_sentiment, write_run_json
from .template_metadata import ensure_template_metadata

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
PIPELINE_VERSION = "0.10.0"
LEGACY_TRAIN_COLUMNS = {
    "ID", "text", "sentiment", "topic", "linguistic_level", "flagprofanity",
    "source", "training_generation", "input_timestamp",
}
LEGACY_INCOMING_COLUMNS = {
    "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
    "flagprofanity", "goldtest", "input_timestamp",
}
RICH_STYLE_COLUMNS = {
    "hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment", "template_family"
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def default_run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def default_timestamp() -> str:
    return _utc_now().isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_input_schema(train_rows: list[dict[str, str]], incoming_rows: list[dict[str, str]]) -> None:
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


def previous_run_id(output_path: Path) -> str | None:
    latest = output_path / "latest.json"
    if not latest.is_file():
        return None
    try:
        value = json.loads(latest.read_text(encoding="utf-8")).get("run_id")
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, str) and RUN_ID_PATTERN.fullmatch(value) else None


def parse_replicate_seeds(value: str | Sequence[int] | None, primary_seed: int) -> list[int]:
    if value is None:
        return [primary_seed]
    if isinstance(value, str):
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        seeds = [int(item) for item in value]
    seeds = list(dict.fromkeys([primary_seed, *seeds]))
    if any(seed < 0 for seed in seeds):
        raise ValueError("replicate seeds must be non-negative integers.")
    return seeds


def environment_versions() -> dict[str, str]:
    return {
        "tensorflow": tf.__version__,
        "scikit_learn": version("scikit-learn"),
        "numpy": version("numpy"),
    }


class ExperimentRunner:
    """Train, evaluate, and atomically publish one immutable experiment run."""

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
        split_seed: int = 42,
        replicate_seeds: str | Sequence[int] | None = None,
    ) -> Path:
        resolved_run_id = run_id or default_run_id()
        if not RUN_ID_PATTERN.fullmatch(resolved_run_id):
            raise ValueError("run_id may contain only letters, numbers, dot, dash, and underscore.")
        if epochs < 1 or patience < 0:
            raise ValueError("epochs must be positive and patience cannot be negative.")

        seeds = parse_replicate_seeds(replicate_seeds, seed)
        input_path = Path(input_dir).resolve()
        output_path = Path(output_root).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / resolved_run_id
        if final_path.exists():
            raise FileExistsError(f"Run already exists: {final_path}")

        ensure_template_metadata(input_path)
        train_path = input_path / "train.csv"
        incoming_path = input_path / "incoming.csv"
        input_manifest_path = input_path / "input_manifest.json"
        train_rows = read_csv(train_path)
        incoming_rows = read_csv(incoming_path)
        validate_input_schema(train_rows, incoming_rows)
        if not input_manifest_path.is_file():
            raise FileNotFoundError(f"Input manifest not found: {input_manifest_path}")
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))

        benchmark_path, benchmark_manifest = ensure_immutable_benchmark(input_path)
        benchmark_rows = read_csv(benchmark_path)
        external_path, external_manifest = ensure_external_sentiment_benchmark(input_path.parent / "external")
        parent_run_id = previous_run_id(output_path)

        temporary_path = Path(tempfile.mkdtemp(prefix=f".{resolved_run_id}-", dir=output_path))
        model_timestamp = default_timestamp()
        try:
            primary_executions: dict[str, PipelineExecution] = {}
            replicate_results: dict[str, list[PipelineResult]] = {"sentiment": [], "topic": []}
            for task in ("sentiment", "topic"):
                primary = execute_pipeline(
                    PipelineConfig(
                        train_path=train_path,
                        incoming_path=incoming_path,
                        task=task,
                        epochs=epochs,
                        validation_fraction=validation_fraction,
                        early_stopping_patience=patience,
                        seed=seed,
                        split_seed=split_seed,
                    )
                )
                primary_executions[task] = primary
                replicate_results[task].append(primary.result)
                for replicate_seed in seeds:
                    if replicate_seed == seed:
                        continue
                    replicate = execute_pipeline(
                        PipelineConfig(
                            train_path=train_path,
                            incoming_path=incoming_path,
                            task=task,
                            epochs=epochs,
                            validation_fraction=validation_fraction,
                            early_stopping_patience=patience,
                            seed=replicate_seed,
                            split_seed=split_seed,
                        )
                    )
                    replicate_results[task].append(replicate.result)

            benchmark = evaluate_benchmark(
                primary_executions,
                benchmark_path,
                benchmark_rows,
                provenance=benchmark_manifest,
            )
            external_validation = evaluate_external_sentiment(
                primary_executions["sentiment"], external_path, external_manifest
            )
            run_metadata = {
                "run_id": resolved_run_id,
                "parent_run_id": parent_run_id,
                "created_at": default_timestamp(),
                "model_timestamp": model_timestamp,
                "status": "complete",
                "pipeline_version": PIPELINE_VERSION,
                "input_generation": int(input_manifest["generation"]),
                "agent_version": input_manifest.get("agent_version"),
                "git_sha": os.getenv("GITHUB_SHA", "local"),
                "python_version": platform.python_version(),
                "tensorflow_version": tf.__version__,
                "environment": environment_versions(),
                "parameters": {
                    "epochs": epochs,
                    "validation_fraction": validation_fraction,
                    "early_stopping_patience": patience,
                    "seed": seed,
                    "split_seed": split_seed,
                    "replicate_seeds": seeds,
                    "max_tokens": 20_000,
                    "sequence_length": 96,
                },
            }
            scope = {
                "data_origin": "synthetic",
                "evaluation_split": "incoming",
                "immutable_benchmark": True,
                "external_validation": True,
                "external_validation_tasks": ["sentiment"],
                "topic_external_validation": False,
                "generalization_claim": "external sentiment evidence only; topic remains synthetic-only",
            }
            document = build_run_document(
                run_metadata=run_metadata,
                scope=scope,
                executions=primary_executions,
                incoming_rows=incoming_rows,
                input_files=(
                    train_path,
                    incoming_path,
                    benchmark_path,
                    input_manifest_path,
                    external_path,
                    external_path.parent / "manifest.json",
                ),
                replicate_results=replicate_results,
                benchmark=benchmark,
                external_validation=external_validation,
            )
            run_json = write_run_json(temporary_path, document)
            materialize_derived_artifacts(run_json)
            temporary_path.rename(final_path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        (output_path / "latest.json").write_text(
            json.dumps({"run_id": resolved_run_id}, indent=2) + "\n", encoding="utf-8"
        )
        return final_path
