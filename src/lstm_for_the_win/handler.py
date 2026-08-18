"""Controlled boundary for input-state transitions and experiment delegation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .agents import SyntheticDataAgent, SyntheticDataConfig
from .experiment import ExperimentRunner, default_timestamp
from .template_metadata import ensure_template_metadata


class PipelineHandler:
    """Coordinate persistent input state and delegate experiment execution."""

    def __init__(self, runner: ExperimentRunner | None = None) -> None:
        self.runner = runner or ExperimentRunner()

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
        timestamp = input_timestamp or default_timestamp()
        if mode == "initialize":
            manifest = agent.initialize(input_dir, timestamp, overwrite=overwrite)
        elif mode == "advance":
            if overwrite:
                raise ValueError("overwrite is valid only when mode=initialize.")
            manifest = agent.advance(input_dir, timestamp)
        else:
            raise ValueError("mode must be initialize or advance.")
        ensure_template_metadata(input_dir)
        return manifest

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
        return self.runner.train_and_publish(
            input_dir,
            output_root,
            run_id=run_id,
            epochs=epochs,
            validation_fraction=validation_fraction,
            patience=patience,
            seed=seed,
            split_seed=split_seed,
            replicate_seeds=replicate_seeds,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Backward-compatible console entry point."""

    from .cli import main as cli_main

    return cli_main(argv)
