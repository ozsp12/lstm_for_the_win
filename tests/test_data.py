from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from lstm_for_the_win.classification.data import (
    clean_text,
    label_for,
    load_incoming,
    load_train,
    stratified_validation_split,
)


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _alpha(index: int) -> str:
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(97 + remainder))
    return "".join(reversed(chars))


def test_clean_text_preserves_lexical_profanity_signal() -> None:
    assert clean_text("This DAMN phone is f***ing bad!") == "this damn phone is f ing bad"


def test_loaders_and_stratified_validation_split(tmp_path: Path) -> None:
    train_rows = []
    incoming_rows = []
    sentiments = ("positive", "neutral", "negative")
    topics = ("smartphone", "television", "refrigerator", "washing_machine")
    levels = ("limited", "informal", "standard", "advanced", "technical")
    for index in range(60):
        token = _alpha(index)
        train_rows.append(
            {
                "ID": index + 1,
                "text": f"training review {token} damn" if index % 4 == 0 else f"training review {token}",
                "sentiment": sentiments[index % 3],
                "topic": topics[index % 4],
                "linguistic_level": levels[index % 5],
                "flagprofanity": 1 if index % 4 == 0 else 0,
                "source": "initial",
                "training_generation": 0,
                "input_timestamp": "2026-08-15T12:00:00+00:00",
            }
        )
        incoming_rows.append(
            {
                "ID": 1000 + index,
                "text": f"incoming review {token}",
                "expected_sentiment": sentiments[index % 3],
                "expected_topic": topics[index % 4],
                "linguistic_level": levels[index % 5],
                "flagprofanity": 0,
                "goldtest": 1 if index % 5 == 0 else 0,
                "input_timestamp": "2026-08-15T12:00:00+00:00",
            }
        )

    train_path = tmp_path / "train.csv"
    incoming_path = tmp_path / "incoming.csv"
    _write(train_path, list(train_rows[0]), train_rows)
    _write(incoming_path, list(incoming_rows[0]), incoming_rows)

    train = load_train(train_path)
    incoming = load_incoming(incoming_path)
    fit, validation = stratified_validation_split(train, "sentiment", 0.20, 42)

    assert len(train) == 60 and len(incoming) == 60
    assert len(fit) + len(validation) == 60
    assert Counter(label_for(row, "sentiment") for row in validation) == {
        "negative": 4,
        "neutral": 4,
        "positive": 4,
    }
