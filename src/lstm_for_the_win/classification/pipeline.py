"""End-to-end LSTM text-classification pipeline orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import load_dataset, split_by_type
from .model import (
    build_confusion_matrix,
    build_lstm_model,
    build_vectorizer,
    evaluate_model,
    predict_probabilities,
    set_global_seed,
    train_model,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Reproducible pipeline parameters."""

    dataset_path: str | Path
    max_tokens: int = 2_000
    sequence_length: int = 24
    embedding_dim: int = 24
    lstm_units: int = 24
    epochs: int = 20
    batch_size: int = 16
    seed: int = 42


@dataclass(frozen=True)
class PipelineResult:
    """Serializable evaluation and inference results."""

    dataset_size: int
    train_size: int
    test_size: int
    labels: list[str]
    label_counts: dict[str, int]
    metrics: dict[str, float]
    history: dict[str, list[float]]
    confusion_matrix: list[list[int]]
    predictions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class PipelineExecution:
    """A trained model paired with its serializable result."""

    model: Any
    result: PipelineResult


def execute_pipeline(config: PipelineConfig) -> PipelineExecution:
    """Train, evaluate, and return both the model and serializable results."""

    set_global_seed(config.seed)
    records = load_dataset(config.dataset_path)
    train_records, test_records = split_by_type(records)

    labels = sorted({record.label for record in records})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_texts = [record.text for record in train_records]
    train_labels = [label_to_index[record.label] for record in train_records]
    test_texts = [record.text for record in test_records]
    test_labels = [label_to_index[record.label] for record in test_records]

    vectorizer = build_vectorizer(
        train_texts,
        max_tokens=config.max_tokens,
        sequence_length=config.sequence_length,
    )
    model = build_lstm_model(
        vectorizer,
        class_count=len(labels),
        embedding_dim=config.embedding_dim,
        lstm_units=config.lstm_units,
    )
    history = train_model(
        model,
        train_texts,
        train_labels,
        epochs=config.epochs,
        batch_size=config.batch_size,
    )
    metrics = evaluate_model(model, test_texts, test_labels)
    test_probabilities = predict_probabilities(model, test_texts)
    predicted_indices = test_probabilities.argmax(axis=1)

    predictions = [
        {
            "ID": record.ID,
            "text": record.text,
            "expected": labels[expected],
            "predicted": labels[int(predicted)],
            "confidence": float(test_probabilities[index][int(predicted)]),
            "correct": bool(expected == int(predicted)),
            "type": record.type,
            "input_timestamp": record.input_timestamp,
        }
        for index, (record, expected, predicted) in enumerate(
            zip(test_records, test_labels, predicted_indices, strict=True)
        )
    ]

    result = PipelineResult(
        dataset_size=len(records),
        train_size=len(train_records),
        test_size=len(test_records),
        labels=labels,
        label_counts=dict(sorted(Counter(record.label for record in records).items())),
        metrics=metrics,
        history=history,
        confusion_matrix=build_confusion_matrix(
            test_labels,
            predicted_indices,
            class_count=len(labels),
        ),
        predictions=predictions,
    )
    return PipelineExecution(model=model, result=result)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the pipeline when only serializable results are required."""

    return execute_pipeline(config).result
