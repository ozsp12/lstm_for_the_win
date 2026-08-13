"""Persisted-model loading and live dual-model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..classification.data import clean_text


def load_models(run_path: str | Path) -> tuple[Any, Any]:
    """Load the sentiment and topic Keras models for one completed run."""

    import tensorflow as tf

    models_path = Path(run_path) / "models"
    return (
        tf.keras.models.load_model(models_path / "sentiment.keras"),
        tf.keras.models.load_model(models_path / "topic.keras"),
    )


def classify_text(model: Any, labels: list[str], text: str) -> dict[str, Any]:
    """Classify one review and return every class probability."""

    import tensorflow as tf

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Enter a review containing at least one word.")
    probabilities = model.predict(tf.constant([cleaned], dtype=tf.string), verbose=0)[0]
    predicted_index = int(probabilities.argmax())
    return {
        "predicted": labels[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            label: float(probabilities[index])
            for index, label in enumerate(labels)
        },
    }
