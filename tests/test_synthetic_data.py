from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig


ROOT = Path(__file__).resolve().parents[1]
INITIAL_TIMESTAMP = "2026-08-13T12:00:00+00:00"
APPEND_TIMESTAMP = "2026-08-14T12:00:00+00:00"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_synthetic_agent_initializes_aligned_versioned_data(tmp_path: Path) -> None:
    config = SyntheticDataConfig.from_json(ROOT / "config" / "synthetic_data.json")
    agent = SyntheticDataAgent(config)
    first = tmp_path / "first"
    second = tmp_path / "second"

    agent.initialize(first, INITIAL_TIMESTAMP)
    agent.initialize(second, INITIAL_TIMESTAMP)

    for filename in ("sentiment_samples.csv", "topic_samples.csv", "reviews.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    reviews = _read_rows(first / "reviews.csv")
    sentiment = _read_rows(first / "sentiment_samples.csv")
    topic = _read_rows(first / "topic_samples.csv")

    assert len(reviews) == 1_000
    assert Counter(row["type"] for row in reviews) == {"train": 500, "test": 500}
    assert [int(row["ID"]) for row in reviews] == list(range(1, 1_001))
    assert len({row["text"] for row in reviews}) == len(reviews)
    assert {row["input_timestamp"] for row in reviews} == {INITIAL_TIMESTAMP}
    assert [row["ID"] for row in sentiment] == [row["ID"] for row in reviews]
    assert [row["ID"] for row in topic] == [row["ID"] for row in reviews]
    assert [row["label"] for row in sentiment] == [
        row["expected_sentiment"] for row in reviews
    ]
    assert [row["label"] for row in topic] == [row["expected_topic"] for row in reviews]


def test_synthetic_agent_appends_100_train_and_100_test_rows(tmp_path: Path) -> None:
    config = SyntheticDataConfig.from_json(ROOT / "config" / "synthetic_data.json")
    destination = tmp_path / "input"
    agent = SyntheticDataAgent(config)
    agent.initialize(destination, INITIAL_TIMESTAMP)
    agent.append(destination, APPEND_TIMESTAMP)

    reviews = _read_rows(destination / "reviews.csv")
    appended = reviews[1_000:]
    assert len(reviews) == 1_200
    assert Counter(row["type"] for row in reviews) == {"train": 600, "test": 600}
    assert Counter(row["type"] for row in appended) == {"train": 100, "test": 100}
    assert [int(row["ID"]) for row in reviews] == list(range(1, 1_201))
    assert {row["input_timestamp"] for row in appended} == {APPEND_TIMESTAMP}
    assert len({row["text"] for row in reviews}) == len(reviews)
