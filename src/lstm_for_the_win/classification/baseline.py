"""TF-IDF logistic-regression baseline for comparison with the LSTM."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def fit_predict_baseline(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    incoming_texts: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    """Fit a conventional linear text baseline and return class probabilities."""

    classifier = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    classifier.fit(list(train_texts), list(train_labels))
    probabilities = classifier.predict_proba(list(incoming_texts))
    classes = list(classifier.named_steps["logistic"].classes_)
    order = np.argsort(np.asarray(classes, dtype=np.int32))
    return np.asarray(probabilities, dtype=np.float64)[:, order]
