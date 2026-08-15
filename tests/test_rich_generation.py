from __future__ import annotations

import csv
import json
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_generation_variability_is_reproducible_and_rich(tmp_path: Path) -> None:
    config = SyntheticDataConfig(
        initial_train_rows=120,
        incoming_rows=120,
        incoming_rows_jitter=60,
        profanity_fraction=0.25,
        goldtest_fraction=0.20,
        emoji_fraction=0.25,
        spelling_error_fraction=0.25,
        slang_fraction=0.25,
        mixed_sentiment_fraction=0.25,
        vary_counts=True,
    )
    agent = SyntheticDataAgent(config)
    first = config.effective_generation(3)
    second = config.effective_generation(3)
    assert first == second
    assert int(first["incoming_rows"]) % 60 == 0
    assert 0.01 <= float(first["goldtest_fraction"]) <= 0.95
    assert 0.01 <= float(first["validation_fraction"]) <= 0.95

    manifest_path = agent.initialize(tmp_path, "2026-08-15T12:00:00+00:00")
    incoming = _rows(tmp_path / "incoming.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = {
        "hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment"
    }
    assert required.issubset(incoming[0])
    assert any(row["hasemoji"] == "1" for row in incoming)
    assert any(any(ord(char) >= 0x1F000 for char in row["text"]) for row in incoming if row["hasemoji"] == "1")
    assert any(row["hasspellingerror"] == "1" for row in incoming)
    assert any(row["hasslang"] == "1" for row in incoming)
    assert any(row["mixed_sentiment"] == "1" for row in incoming)
    assert manifest["effective_generation"] == config.effective_generation(0)
    assert "incoming_emoji_counts" in manifest
    assert "incoming_spelling_error_counts" in manifest
    assert "incoming_slang_counts" in manifest
