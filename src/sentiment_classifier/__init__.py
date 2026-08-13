"""Public entry point for LSTM sentiment classification."""

from src.text_classifier import (
    PipelineConfig as SentimentPipelineConfig,
    PipelineResult as SentimentPipelineResult,
    run_pipeline as run_sentiment_pipeline,
)

__all__ = [
    "SentimentPipelineConfig",
    "SentimentPipelineResult",
    "run_sentiment_pipeline",
]
