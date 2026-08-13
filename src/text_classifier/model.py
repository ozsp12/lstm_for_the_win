"""LSTM model construction, training, and evaluation."""

from __future__ import annotations

import os
import random
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Configure every random source used by the pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_vectorizer(
    train_texts: Sequence[str],
    max_tokens: int,
    sequence_length: int,
) -> tf.keras.layers.TextVectorization:
    """Create and adapt the vectorization layer using training data only."""

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
    """Build and compile a multiclass LSTM network."""

    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    tokens = vectorizer(inputs)
    embedding = tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(tokens)
    encoded = tf.keras.layers.LSTM(lstm_units, name="lstm")(embedding)
    hidden = tf.keras.layers.Dense(32, activation="relu", name="dense")(encoded)
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
    epochs: int,
    batch_size: int,
    verbose: int = 0,
) -> dict[str, list[float]]:
    """Train the model and return a serializable history."""

    history = model.fit(
        tf.constant(list(train_texts), dtype=tf.string),
        np.asarray(train_labels, dtype=np.int32),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
    )
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def evaluate_model(
    model: tf.keras.Model,
    test_texts: Sequence[str],
    test_labels: Sequence[int],
) -> dict[str, float]:
    """Calculate loss and accuracy on the test set."""

    loss, accuracy = model.evaluate(
        tf.constant(list(test_texts), dtype=tf.string),
        np.asarray(test_labels, dtype=np.int32),
        verbose=0,
    )
    return {"loss": float(loss), "accuracy": float(accuracy)}


def predict_classes(model: tf.keras.Model, texts: Sequence[str]) -> np.ndarray:
    """Return the most likely class index for each text."""

    probabilities = model.predict(
        tf.constant(list(texts), dtype=tf.string),
        verbose=0,
    )
    return np.argmax(probabilities, axis=1)


def build_confusion_matrix(
    expected: Sequence[int],
    predicted: Sequence[int],
    class_count: int,
) -> list[list[int]]:
    """Build a confusion matrix without a scikit-learn dependency."""

    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for expected_class, predicted_class in zip(expected, predicted, strict=True):
        matrix[int(expected_class)][int(predicted_class)] += 1
    return matrix
