from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig


ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_synthetic_agent_is_balanced_and_deterministic(tmp_path: Path) -> None:
    config = SyntheticDataConfig.from_json(ROOT / "config" / "synthetic_data.json")
    agent = SyntheticDataAgent(config)
    first = tmp_path / "first"
    second = tmp_path / "second"

    agent.write(first)
    agent.write(second)

    for filename in ("sentiment_samples.csv", "topic_samples.csv", "reviews.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    sentiment = _read_rows(first / "sentiment_samples.csv")
    topic = _read_rows(first / "topic_samples.csv")
    reviews = _read_rows(first / "reviews.csv")

    assert Counter(row["label"] for row in sentiment) == {
        "positive": config.sentiment_samples_per_label,
        "neutral": config.sentiment_samples_per_label,
        "negative": config.sentiment_samples_per_label,
    }
    assert set(Counter(row["label"] for row in topic).values()) == {
        config.topic_samples_per_label
    }
    assert len(reviews) == 4 * 3 * config.reviews_per_sentiment_topic_pair
    assert len({row["text"] for row in reviews}) == len(reviews)
