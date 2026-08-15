from __future__ import annotations

from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.classification import PipelineConfig, execute_pipeline


def test_pipeline_reports_baseline_segments_and_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    config = SyntheticDataConfig(
        initial_train_rows=120,
        incoming_rows=120,
        profanity_fraction=0.50,
        goldtest_fraction=0.50,
        validation_fraction=0.20,
    )
    SyntheticDataAgent(config).initialize(
        input_dir,
        "2026-08-15T12:00:00+00:00",
    )

    execution = execute_pipeline(
        PipelineConfig(
            train_path=input_dir / "train.csv",
            incoming_path=input_dir / "incoming.csv",
            task="sentiment",
            max_tokens=500,
            sequence_length=24,
            embedding_dim=8,
            lstm_units=8,
            epochs=1,
            batch_size=16,
            validation_fraction=0.20,
            early_stopping_patience=0,
            seed=42,
        )
    )

    result = execution.result
    assert result.train_size == 120
    assert result.fit_size + result.validation_size == 120
    assert result.incoming_size == 120
    assert set(result.metrics) >= {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "expected_calibration_error",
    }
    assert set(result.baseline_metrics) == set(result.metrics)
    assert set(result.segment_metrics["linguistic_level"]) == {
        "limited",
        "informal",
        "standard",
        "advanced",
        "technical",
    }
    assert set(result.segment_metrics["flagprofanity"]) == {"0", "1"}
    assert set(result.segment_metrics["goldtest"]) == {"0", "1"}
    assert len(result.predictions) == 120
