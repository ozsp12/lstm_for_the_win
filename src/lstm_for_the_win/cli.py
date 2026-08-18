"""Command-line interface for the continual-learning experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .agents import SyntheticDataConfig
from .handler import PipelineHandler


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="data/input")
    parser.add_argument("--output-root", default="data/output")
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("PIPELINE_EPOCHS", "20")))
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--patience", type=int, default=int(os.getenv("PIPELINE_PATIENCE", "3")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=int(os.getenv("PIPELINE_SPLIT_SEED", "42")))
    parser.add_argument("--replicate-seeds", default=os.getenv("PIPELINE_REPLICATE_SEEDS"))


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
        split_seed=arguments.split_seed,
        replicate_seeds=arguments.replicate_seeds,
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
