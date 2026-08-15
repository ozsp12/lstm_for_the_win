from __future__ import annotations

import numpy as np

from lstm_for_the_win.classification.model import (
    build_confusion_matrix,
    build_lstm_model,
    build_vectorizer,
    classification_metrics,
    predict_probabilities,
)


def test_classification_metrics_are_correct_for_perfect_predictions() -> None:
    expected = [0, 1, 2, 0, 1, 2]
    probabilities = np.asarray(
        [
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.95, 0.03, 0.02],
            [0.02, 0.96, 0.02],
            [0.01, 0.02, 0.97],
        ]
    )
    metrics = classification_metrics(expected, probabilities, 3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["log_loss"] < 0.1
    assert metrics["brier_score"] < 0.01
    assert build_confusion_matrix(expected, probabilities.argmax(axis=1), 3) == [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]


def test_lstm_model_accepts_raw_text_and_returns_probabilities() -> None:
    texts = ["good phone battery", "bad television screen", "average fridge"]
    vectorizer = build_vectorizer(texts, max_tokens=100, sequence_length=8)
    model = build_lstm_model(vectorizer, class_count=3, embedding_dim=4, lstm_units=4)
    probabilities = predict_probabilities(model, ["good phone", "bad screen"])
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)
