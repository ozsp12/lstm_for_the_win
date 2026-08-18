from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from lstm_for_the_win.classification.data import (
    ReviewRecord,
    clean_text,
    label_for,
    load_incoming,
    load_train,
    stratified_validation_split,
    validation_split,
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


def test_clean_text_preserves_lexical_profanity_and_emoji_signal() -> None:
    assert clean_text("This DAMN phone is f***ing bad! 😬") == "this damn phone is f ing bad 😬"


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
                "text": f"incoming review {token} 😅" if index % 6 == 0 else f"incoming review {token}",
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
    assert incoming[0].hasemoji == 1
    assert Counter(label_for(row, "sentiment") for row in validation) == {
        "negative": 4, "neutral": 4, "positive": 4,
    }


def test_validation_split_holds_out_a_persisted_whole_template_family() -> None:
    families = (
        ("noticed", "I noticed that the battery on this phone works well token {token}"),
        ("using", "I have been using this phone during normal use and the battery works well token {token}"),
        ("stood_out", "The battery is what stood out during normal use it works well token {token}"),
    )
    sentiments = ("positive", "neutral", "negative")
    records: list[ReviewRecord] = []
    review_id = 1
    for family, pattern in families:
        for sentiment in sentiments:
            for repetition in range(4):
                records.append(
                    ReviewRecord(
                        ID=review_id,
                        text=clean_text(pattern.format(token=f"{sentiment}{repetition}")),
                        sentiment=sentiment,
                        topic="smartphone",
                        linguistic_level="standard",
                        flagprofanity=0,
                        input_timestamp="2026-08-15T12:00:00+00:00",
                        source="initial",
                        training_generation=0,
                        template_family=family,
                    )
                )
                review_id += 1

    fit, validation, metadata = validation_split(records, "sentiment", 1 / 3, 42)
    validation_families = {record.template_family for record in validation}
    fit_families = {record.template_family for record in fit}

    assert metadata["method"] == "template_family_grouped"
    assert metadata["family_source"] == "persisted_metadata"
    assert len(validation_families) == 1
    assert validation_families.isdisjoint(fit_families)
    assert set(label_for(record, "sentiment") for record in validation) == set(sentiments)
