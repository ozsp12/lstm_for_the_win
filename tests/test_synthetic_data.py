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
    return SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, vary_counts=False)


def test_initialize_creates_nonuniform_disjoint_train_and_incoming(tmp_path: Path) -> None:
    agent = SyntheticDataAgent(_config())
    manifest_path = agent.initialize(tmp_path, TIMESTAMP_0)
    train = _rows(tmp_path / "train.csv")
    incoming = _rows(tmp_path / "incoming.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(train) == 1200 and len(incoming) == 1200
    assert len({row["ID"] for row in train}) == len(train)
    assert len({row["ID"] for row in incoming}) == len(incoming)
    assert not ({row["ID"] for row in train} & {row["ID"] for row in incoming})
    assert not ({row["text"] for row in train} & {row["text"] for row in incoming})
    levels = Counter(row["linguistic_level"] for row in incoming)
    assert set(levels) == {"limited", "informal", "standard", "advanced", "technical"}
    assert len(set(levels.values())) > 1
    strata = Counter((row["expected_sentiment"], row["expected_topic"], row["linguistic_level"]) for row in incoming)
    assert len(strata) == 60 and len(set(strata.values())) > 1
    assert all(row["template_family"] for row in train + incoming)
    assert manifest["generation"] == 0
    assert manifest["promoted_from_previous_incoming"] == 0
    assert manifest["record_counts"] == {"incoming.csv": 1200, "train.csv": 1200}
    assert manifest["template_family_metadata"]["origin"] == "generated_at_render_time"


def test_advance_promotes_actual_gold_rows_and_replaces_incoming(tmp_path: Path) -> None:
    agent = SyntheticDataAgent(_config())
    agent.initialize(tmp_path, TIMESTAMP_0)
    first_incoming = _rows(tmp_path / "incoming.csv")
    promoted_ids = {row["ID"] for row in first_incoming if row["goldtest"] == "1"}
    first_ids = {row["ID"] for row in first_incoming}
    manifest_path = agent.advance(tmp_path, TIMESTAMP_1)
    train = _rows(tmp_path / "train.csv")
    next_incoming = _rows(tmp_path / "incoming.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(train) == 1200 + len(promoted_ids)
    assert len(next_incoming) == 1200
    assert promoted_ids.issubset({row["ID"] for row in train})
    assert not (first_ids & {row["ID"] for row in next_incoming})
    assert Counter(row["source"] for row in train)["goldtest"] == len(promoted_ids)
    assert manifest["generation"] == 1
    assert manifest["promoted_from_previous_incoming"] == len(promoted_ids)
