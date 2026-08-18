from __future__ import annotations

from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.classification import PipelineConfig, execute_pipeline
from lstm_for_the_win.template_metadata import ensure_template_metadata


def test_pipeline_reports_baseline_segments_and_structural_validation(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    config = SyntheticDataConfig(
        initial_train_rows=1200,
        incoming_rows=1200,
        incoming_rows_jitter=0,
        profanity_fraction=0.50,
        goldtest_fraction=0.50,
        emoji_fraction=0.50,
        spelling_error_fraction=0.50,
        slang_fraction=0.50,
        mixed_sentiment_fraction=0.50,
        validation_fraction=0.20,
        vary_counts=False,
    )
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")
    ensure_template_metadata(input_dir)

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
            batch_size=32,
            validation_fraction=0.20,
            early_stopping_patience=0,
            seed=42,
            split_seed=42,
        )
    )

    result = execution.result
    assert result.seed == 42
    assert result.train_size == 1200
    assert result.fit_size + result.validation_size == 1200
    assert result.incoming_size == 1200
    assert result.validation_split["method"] in {"template_family_grouped", "stratified_random_fallback"}
    assert result.validation_split["family_source"] == "persisted_metadata"
    assert 0.0 < float(result.validation_split["actual_fraction"]) < 1.0
    assert set(result.metrics) >= {
        "accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1",
        "log_loss", "brier_score", "expected_calibration_error",
    }
    assert set(result.baseline_metrics) == set(result.metrics)
    assert result.paired_comparison["method"] == "mcnemar_exact_two_sided"
    assert 0.0 <= float(result.paired_comparison["p_value"]) <= 1.0
    assert set(result.segment_metrics["linguistic_level"]) == {
        "limited", "informal", "standard", "advanced", "technical",
    }
    for dimension in ("flagprofanity", "hasemoji", "hasspellingerror", "hasslang", "mixed_sentiment", "goldtest"):
        assert set(result.segment_metrics[dimension]) == {"0", "1"}
    assert result.segment_metrics["template_family"]
    assert set(result.segment_metrics["length_class"]).issubset({"short", "medium", "long"})
    assert len(result.predictions) == 1200
    assert {
        "hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment", "template_family"
    }.issubset(result.predictions[0])
