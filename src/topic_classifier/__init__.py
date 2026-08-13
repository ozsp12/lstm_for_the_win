"""Public entry point for LSTM topic classification."""

from src.text_classifier import (
    PipelineConfig as TopicPipelineConfig,
    PipelineResult as TopicPipelineResult,
    run_pipeline as run_topic_pipeline,
)

__all__ = ["TopicPipelineConfig", "TopicPipelineResult", "run_topic_pipeline"]
