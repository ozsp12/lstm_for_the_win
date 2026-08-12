"""Orquestração do pipeline completo de classificação de tópicos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import load_dataset, stratified_split
from .model import (
    build_confusion_matrix,
    build_lstm_model,
    build_vectorizer,
    evaluate_model,
    predict_classes,
    set_global_seed,
    train_model,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Parâmetros reproduzíveis do pipeline."""

    dataset_path: str | Path
    test_fraction: float = 0.20
    max_tokens: int = 2_000
    sequence_length: int = 24
    embedding_dim: int = 24
    lstm_units: int = 24
    epochs: int = 10
    batch_size: int = 16
    seed: int = 42


@dataclass(frozen=True)
class PipelineResult:
    """Resultados necessários para leitura humana e validação automática."""

    dataset_size: int
    train_size: int
    test_size: int
    labels: list[str]
    metrics: dict[str, float]
    history: dict[str, list[float]]
    confusion_matrix: list[list[int]]
    predictions: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Converte o resultado em um dicionário serializável."""

        return asdict(self)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Executa leitura, divisão, treino, avaliação e predição."""

    set_global_seed(config.seed)
    records = load_dataset(config.dataset_path)
    train_records, test_records = stratified_split(
        records,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )

    labels = sorted({topic for _, topic in records})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_texts = [text for text, _ in train_records]
    train_labels = [label_to_index[topic] for _, topic in train_records]
    test_texts = [text for text, _ in test_records]
    test_labels = [label_to_index[topic] for _, topic in test_records]

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
    predicted_indices = predict_classes(model, test_texts)

    predictions = [
        {
            "text": text,
            "expected": labels[expected],
            "predicted": labels[int(predicted)],
        }
        for text, expected, predicted in zip(
            test_texts,
            test_labels,
            predicted_indices,
            strict=True,
        )
    ]

    return PipelineResult(
        dataset_size=len(records),
        train_size=len(train_records),
        test_size=len(test_records),
        labels=labels,
        metrics=metrics,
        history=history,
        confusion_matrix=build_confusion_matrix(
            test_labels,
            predicted_indices,
            class_count=len(labels),
        ),
        predictions=predictions,
    )
