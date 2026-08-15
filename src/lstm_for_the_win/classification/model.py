"""LSTM model construction, training, evaluation, and classification metrics."""

from __future__ import annotations

import os
import random
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_vectorizer(
    train_texts: Sequence[str],
    max_tokens: int,
    sequence_length: int,
) -> tf.keras.layers.TextVectorization:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorizer.adapt(tf.constant(list(train_texts), dtype=tf.string))
    return vectorizer


def build_lstm_model(
    vectorizer: tf.keras.layers.TextVectorization,
    class_count: int,
    embedding_dim: int,
    lstm_units: int,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    tokens = vectorizer(inputs)
    embedding = tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(tokens)
    encoded = tf.keras.layers.LSTM(lstm_units, name="lstm")(embedding)
    regularized = tf.keras.layers.Dropout(0.20, name="dropout")(encoded)
    hidden = tf.keras.layers.Dense(32, activation="relu", name="dense")(regularized)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="class")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="text_classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    validation_texts: Sequence[str],
    validation_labels: Sequence[int],
    epochs: int,
    batch_size: int,
    patience: int,
    verbose: int = 0,
) -> dict[str, list[float]]:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        tf.constant(list(train_texts), dtype=tf.string),
        np.asarray(train_labels, dtype=np.int32),
        validation_data=(
            tf.constant(list(validation_texts), dtype=tf.string),
            np.asarray(validation_labels, dtype=np.int32),
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
        callbacks=callbacks,
    )
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def predict_probabilities(model: tf.keras.Model, texts: Sequence[str]) -> np.ndarray:
    return model.predict(
        tf.constant(list(texts), dtype=tf.string),
        verbose=0,
    )


def build_confusion_matrix(
    expected: Sequence[int],
    predicted: Sequence[int],
    class_count: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for expected_class, predicted_class in zip(expected, predicted, strict=True):
        matrix[int(expected_class)][int(predicted_class)] += 1
    return matrix


def classification_metrics(
    expected: Sequence[int],
    probabilities: np.ndarray,
    class_count: int,
    *,
    calibration_bins: int = 10,
) -> dict[str, float]:
    """Calculate accuracy, macro/weighted metrics, log-loss, Brier score, and ECE."""

    y_true = np.asarray(expected, dtype=np.int32)
    probs = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or probs.ndim != 2 or len(y_true) != len(probs):
        raise ValueError("Expected labels and probabilities have incompatible shapes.")
    if probs.shape[1] != class_count:
        raise ValueError("Probability columns must match class_count.")
    if len(y_true) == 0:
        raise ValueError("At least one example is required for metrics.")

    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    accuracy = float(np.mean(y_pred == y_true))

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    for cls in range(class_count):
        true_positive = int(np.sum((y_true == cls) & (y_pred == cls)))
        false_positive = int(np.sum((y_true != cls) & (y_pred == cls)))
        false_negative = int(np.sum((y_true == cls) & (y_pred != cls)))
        support = int(np.sum(y_true == cls))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total_support = sum(supports)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1s, supports, strict=True)) / total_support
    log_loss = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))
    targets = np.eye(class_count, dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probs - targets) ** 2, axis=1)))

    confidences = probs.max(axis=1)
    correctness = (y_pred == y_true).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_weight = float(np.mean(mask))
        ece += bin_weight * abs(float(np.mean(correctness[mask])) - float(np.mean(confidences[mask])))

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(weighted_f1),
        "log_loss": log_loss,
        "brier_score": brier,
        "expected_calibration_error": float(ece),
    }
