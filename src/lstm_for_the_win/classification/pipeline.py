"""End-to-end LSTM training and incoming-review evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .baseline import fit_predict_baseline
from .data import ReviewRecord, label_for, load_incoming, load_train, stratified_validation_split
from .model import (
    build_confusion_matrix,
    build_lstm_model,
    build_vectorizer,
    classification_metrics,
    predict_probabilities,
    set_global_seed,
    train_model,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Reproducible parameters for one classification task."""

    train_path: str | Path
    incoming_path: str | Path
    task: str
    max_tokens: int = 20_000
    sequence_length: int = 96
    embedding_dim: int = 48
    lstm_units: int = 48
    epochs: int = 20
    batch_size: int = 32
    validation_fraction: float = 0.15
    early_stopping_patience: int = 3
    seed: int = 42


@dataclass(frozen=True)
class PipelineResult:
    """Serializable evaluation results for one task."""

    task: str
    train_size: int
    fit_size: int
    validation_size: int
    incoming_size: int
    labels: list[str]
    label_counts: dict[str, int]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    metric_delta_vs_baseline: dict[str, float]
    segment_metrics: dict[str, dict[str, dict[str, float]]]
    history: dict[str, list[float]]
    confusion_matrix: list[list[int]]
    predictions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineExecution:
    model: Any
    result: PipelineResult


def _segment_metrics(
    records: list[ReviewRecord],
    expected: list[int],
    probabilities: np.ndarray,
    class_count: int,
) -> dict[str, dict[str, dict[str, float]]]:
    dimensions: dict[str, tuple[Callable[[ReviewRecord], str], list[str]]] = {
        "linguistic_level": (
            lambda record: record.linguistic_level,
            ["limited", "informal", "standard", "advanced", "technical"],
        ),
        "flagprofanity": (lambda record: str(record.flagprofanity), ["0", "1"]),
        "hasemoji": (lambda record: str(record.hasemoji), ["0", "1"]),
        "hasspellingerror": (lambda record: str(record.hasspellingerror), ["0", "1"]),
        "hasslang": (lambda record: str(record.hasslang), ["0", "1"]),
        "length_class": (lambda record: record.length_class, ["short", "medium", "long"]),
        "mixed_sentiment": (lambda record: str(record.mixed_sentiment), ["0", "1"]),
        "goldtest": (lambda record: str(record.goldtest), ["0", "1"]),
    }
    output: dict[str, dict[str, dict[str, float]]] = {}
    for dimension, (getter, values) in dimensions.items():
        output[dimension] = {}
        for value in values:
            indices = [index for index, record in enumerate(records) if getter(record) == value]
            if not indices:
                continue
            subset_expected = [expected[index] for index in indices]
            subset_probabilities = probabilities[indices]
            output[dimension][value] = classification_metrics(
                subset_expected,
                subset_probabilities,
                class_count,
            )
    return output


def execute_pipeline(config: PipelineConfig) -> PipelineExecution:
    """Train one LSTM, evaluate new incoming reviews, and compare with a linear baseline."""

    if config.task not in {"sentiment", "topic"}:
        raise ValueError("task must be sentiment or topic.")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative.")

    set_global_seed(config.seed)
    train_records = load_train(config.train_path)
    incoming_records = load_incoming(config.incoming_path)
    fit_records, validation_records = stratified_validation_split(
        train_records,
        config.task,
        config.validation_fraction,
        config.seed,
    )

    labels = sorted({label_for(record, config.task) for record in train_records})
    label_to_index = {label: index for index, label in enumerate(labels)}
    incoming_labels = {label_for(record, config.task) for record in incoming_records}
    if not incoming_labels.issubset(label_to_index):
        raise ValueError("incoming.csv contains a label absent from train.csv.")

    fit_texts = [record.text for record in fit_records]
    fit_labels = [label_to_index[label_for(record, config.task)] for record in fit_records]
    validation_texts = [record.text for record in validation_records]
    validation_labels = [label_to_index[label_for(record, config.task)] for record in validation_records]
    incoming_texts = [record.text for record in incoming_records]
    expected = [label_to_index[label_for(record, config.task)] for record in incoming_records]

    vectorizer = build_vectorizer(
        fit_texts,
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
        fit_texts,
        fit_labels,
        validation_texts,
        validation_labels,
        epochs=config.epochs,
        batch_size=config.batch_size,
        patience=config.early_stopping_patience,
    )

    probabilities = predict_probabilities(model, incoming_texts)
    predicted_indices = probabilities.argmax(axis=1)
    metrics = classification_metrics(expected, probabilities, len(labels))
    baseline_probabilities = fit_predict_baseline(
        fit_texts,
        fit_labels,
        incoming_texts,
        seed=config.seed,
    )
    baseline_metrics = classification_metrics(expected, baseline_probabilities, len(labels))
    comparable = ("accuracy", "macro_f1", "weighted_f1", "log_loss", "brier_score")
    delta = {name: float(metrics[name] - baseline_metrics[name]) for name in comparable}

    predictions = [
        {
            "ID": record.ID,
            "text": record.text,
            "expected": labels[expected_index],
            "predicted": labels[int(predicted_index)],
            "confidence": float(probabilities[index][int(predicted_index)]),
            "correct": bool(expected_index == int(predicted_index)),
            "linguistic_level": record.linguistic_level,
            "flagprofanity": record.flagprofanity,
            "hasemoji": record.hasemoji,
            "hasspellingerror": record.hasspellingerror,
            "hasslang": record.hasslang,
            "length_class": record.length_class,
            "mixed_sentiment": record.mixed_sentiment,
            "goldtest": record.goldtest,
            "input_timestamp": record.input_timestamp,
        }
        for index, (record, expected_index, predicted_index) in enumerate(
            zip(incoming_records, expected, predicted_indices, strict=True)
        )
    ]

    result = PipelineResult(
        task=config.task,
        train_size=len(train_records),
        fit_size=len(fit_records),
        validation_size=len(validation_records),
        incoming_size=len(incoming_records),
        labels=labels,
        label_counts=dict(sorted(Counter(label_for(record, config.task) for record in train_records).items())),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        metric_delta_vs_baseline=delta,
        segment_metrics=_segment_metrics(incoming_records, expected, probabilities, len(labels)),
        history=history,
        confusion_matrix=build_confusion_matrix(expected, predicted_indices, class_count=len(labels)),
        predictions=predictions,
    )
    return PipelineExecution(model=model, result=result)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    return execute_pipeline(config).result
