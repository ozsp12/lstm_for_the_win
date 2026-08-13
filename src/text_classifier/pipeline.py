"""End-to-end LSTM text-classification pipeline orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import load_dataset, stratified_split
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
    test_fraction: float = 0.20
    max_tokens: int = 2_000
    sequence_length: int = 24
    embedding_dim: int = 24
    lstm_units: int = 24
    epochs: int = 20
    batch_size: int = 16
    seed: int = 42
    demo_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    """Results required for human review and automated validation."""

    dataset_size: int
    train_size: int
    test_size: int
    labels: list[str]
    label_counts: dict[str, int]
    metrics: dict[str, float]
    history: dict[str, list[float]]
    confusion_matrix: list[list[int]]
    predictions: list[dict[str, Any]]
    demo_predictions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a serializable dictionary."""

        return asdict(self)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run data loading, splitting, training, evaluation, and prediction."""

    set_global_seed(config.seed)
    records = load_dataset(config.dataset_path)
    train_records, test_records = stratified_split(
        records,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )

    labels = sorted({label for _, label in records})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_texts = [text for text, _ in train_records]
    train_labels = [label_to_index[label] for _, label in train_records]
    test_texts = [text for text, _ in test_records]
    test_labels = [label_to_index[label] for _, label in test_records]

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
            "text": text,
            "expected": labels[expected],
            "predicted": labels[int(predicted)],
            "confidence": float(test_probabilities[index][int(predicted)]),
            "correct": bool(expected == int(predicted)),
        }
        for index, (text, expected, predicted) in enumerate(
            zip(test_texts, test_labels, predicted_indices, strict=True)
        )
    ]

    demo_predictions: list[dict[str, Any]] = []
    if config.demo_texts:
        demo_probabilities = predict_probabilities(model, config.demo_texts)
        for text, probabilities in zip(
            config.demo_texts,
            demo_probabilities,
            strict=True,
        ):
            predicted_index = int(probabilities.argmax())
            demo_predictions.append(
                {
                    "text": text,
                    "predicted": labels[predicted_index],
                    "confidence": float(probabilities[predicted_index]),
                    "probabilities": {
                        label: float(probabilities[index])
                        for index, label in enumerate(labels)
                    },
                }
            )

    return PipelineResult(
        dataset_size=len(records),
        train_size=len(train_records),
        test_size=len(test_records),
        labels=labels,
        label_counts=dict(sorted(Counter(label for _, label in records).items())),
        metrics=metrics,
        history=history,
        confusion_matrix=build_confusion_matrix(
            test_labels,
            predicted_indices,
            class_count=len(labels),
        ),
        predictions=predictions,
        demo_predictions=demo_predictions,
    )
