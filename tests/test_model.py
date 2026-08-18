from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support

from lstm_for_the_win.classification.model import (
    build_confusion_matrix,
    build_lstm_model,
    build_vectorizer,
    calibration_profile,
    classification_metrics,
    predict_probabilities,
)


def test_classification_metrics_match_sklearn_with_unpredicted_class() -> None:
    expected = np.asarray([0, 0, 1, 1, 2, 2])
    probabilities = np.asarray([
        [0.80, 0.15, 0.05],
        [0.70, 0.25, 0.05],
        [0.60, 0.35, 0.05],
        [0.55, 0.40, 0.05],
        [0.45, 0.40, 0.15],
        [0.40, 0.35, 0.25],
    ])
    predicted = probabilities.argmax(axis=1)
    metrics = classification_metrics(expected, probabilities, 3)
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=[0, 1, 2], zero_division=0
    )
    assert 2 not in set(predicted)
    assert metrics["accuracy"] == pytest.approx(accuracy_score(expected, predicted))
    assert metrics["precision_macro"] == pytest.approx(float(np.mean(precision)))
    assert metrics["recall_macro"] == pytest.approx(float(np.mean(recall)))
    assert metrics["macro_f1"] == pytest.approx(float(np.mean(f1)))
    assert metrics["weighted_f1"] == pytest.approx(float(np.average(f1, weights=support)))
    assert metrics["log_loss"] == pytest.approx(log_loss(expected, probabilities, labels=[0, 1, 2]))
    assert build_confusion_matrix(expected, predicted, 3) == [
        [2, 0, 0],
        [2, 0, 0],
        [2, 0, 0],
    ]

    bins = calibration_profile(expected, probabilities, 3)
    assert len(bins) == 10
    assert sum(int(item["support"]) for item in bins) == len(expected)
    assert all(0.0 <= float(item["mean_confidence"]) <= 1.0 for item in bins)


def test_classification_metrics_are_correct_for_perfect_predictions() -> None:
    expected = [0, 1, 2, 0, 1, 2]
    probabilities = np.asarray([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98],
        [0.95, 0.03, 0.02],
        [0.02, 0.96, 0.02],
        [0.01, 0.02, 0.97],
    ])
    metrics = classification_metrics(expected, probabilities, 3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["log_loss"] < 0.1
    assert metrics["brier_score"] < 0.01


def test_lstm_model_accepts_raw_text_and_returns_probabilities() -> None:
    texts = ["good phone battery", "bad television screen", "average fridge"]
    vectorizer = build_vectorizer(texts, max_tokens=100, sequence_length=8)
    model = build_lstm_model(vectorizer, class_count=3, embedding_dim=4, lstm_units=4)
    probabilities = predict_probabilities(model, ["good phone", "bad screen"])
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)
