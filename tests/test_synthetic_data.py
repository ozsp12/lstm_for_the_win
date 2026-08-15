from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig


TIMESTAMP_0 = "2026-08-15T12:00:00+00:00"
TIMESTAMP_1 = "2026-08-16T12:00:00+00:00"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _config() -> SyntheticDataConfig:
    return SyntheticDataConfig(
        initial_train_rows=1_200,
        incoming_rows=1_200,
        profanity_fraction=0.25,
        goldtest_fraction=0.20,
        validation_fraction=0.15,
    )


def test_initialize_creates_balanced_disjoint_train_and_incoming(tmp_path: Path) -> None:
    agent = SyntheticDataAgent(_config())
    manifest_path = agent.initialize(tmp_path, TIMESTAMP_0)

    train = _rows(tmp_path / "train.csv")
    incoming = _rows(tmp_path / "incoming.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(train) == 1_200
    assert len(incoming) == 1_200
    assert len({row["ID"] for row in train}) == len(train)
    assert len({row["ID"] for row in incoming}) == len(incoming)
    assert not ({row["ID"] for row in train} & {row["ID"] for row in incoming})
    assert not ({row["text"] for row in train} & {row["text"] for row in incoming})
    assert Counter(row["linguistic_level"] for row in incoming) == {
        "limited": 240,
        "informal": 240,
        "standard": 240,
        "advanced": 240,
        "technical": 240,
    }
    assert sum(row["flagprofanity"] == "1" for row in incoming) == 300
    assert sum(row["goldtest"] == "1" for row in incoming) == 240
    assert manifest["generation"] == 0
    assert manifest["record_counts"] == {"incoming.csv": 1_200, "train.csv": 1_200}


def test_advance_promotes_goldtest_and_replaces_entire_incoming_batch(tmp_path: Path) -> None:
    agent = SyntheticDataAgent(_config())
    agent.initialize(tmp_path, TIMESTAMP_0)
    first_incoming = _rows(tmp_path / "incoming.csv")
    promoted_ids = {row["ID"] for row in first_incoming if row["goldtest"] == "1"}
    first_incoming_ids = {row["ID"] for row in first_incoming}

    manifest_path = agent.advance(tmp_path, TIMESTAMP_1)
    train = _rows(tmp_path / "train.csv")
    next_incoming = _rows(tmp_path / "incoming.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(train) == 1_440
    assert len(next_incoming) == 1_200
    assert promoted_ids.issubset({row["ID"] for row in train})
    assert not (first_incoming_ids & {row["ID"] for row in next_incoming})
    assert Counter(row["source"] for row in train)["goldtest"] == 240
    assert {row["training_generation"] for row in train if row["source"] == "goldtest"} == {"1"}
    assert manifest["generation"] == 1
    assert manifest["promoted_from_previous_incoming"] == 240
