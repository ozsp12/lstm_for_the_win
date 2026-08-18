from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_generation_variability_is_reproducible_and_rich(tmp_path: Path) -> None:
    config = SyntheticDataConfig(
        initial_train_rows=1200,
        incoming_rows=1200,
        incoming_rows_jitter=120,
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
    assert first == config.effective_generation(3)
    assert int(first["incoming_rows"]) >= 1000
    assert int(first["incoming_rows"]) % 60 == 0

    manifest_path = agent.initialize(tmp_path, "2026-08-15T12:00:00+00:00")
    train = _rows(tmp_path / "train.csv")
    incoming = _rows(tmp_path / "incoming.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = {
        "hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment", "template_family"
    }
    assert required.issubset(incoming[0])
    assert all(row["template_family"] for row in train + incoming)
    assert set(row["template_family"] for row in incoming) == {
        "noticed", "using", "stood_out", "context_component", "main_impression", "attention"
    }
    assert manifest["template_family_metadata"]["origin"] == "generated_at_render_time"
    assert manifest["generation"] == 0
    assert manifest["promoted_from_previous_incoming"] == 0

    strata = Counter((row["expected_sentiment"], row["expected_topic"], row["linguistic_level"]) for row in incoming)
    assert len(strata) == 60
    assert len(set(strata.values())) > 1, "Synthetic strata should not be perfectly balanced."

    assert any(row["hasemoji"] == "1" for row in incoming)
    assert any(any(ord(char) >= 0x1F000 for char in row["text"]) for row in incoming if row["hasemoji"] == "1")
    assert any(row["flagprofanity"] == "1" for row in incoming)
    assert any(row["hasspellingerror"] == "1" for row in incoming)
    assert any(row["hasslang"] == "1" for row in incoming)
    assert any(row["mixed_sentiment"] == "1" for row in incoming)
    assert any(
        phrase in row["text"]
        for row in incoming if row["linguistic_level"] == "advanced"
        for phrase in ("overall assessment", "first impression", "comparable conditions", "ownership experience", "ordinary use cases")
    )


def test_advance_preserves_explicit_template_family(tmp_path: Path) -> None:
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, vary_counts=False)
    agent = SyntheticDataAgent(config)
    agent.initialize(tmp_path, "2026-08-15T12:00:00+00:00")
    before = _rows(tmp_path / "incoming.csv")
    promoted = {row["ID"]: row["template_family"] for row in before if row["goldtest"] == "1"}
    agent.advance(tmp_path, "2026-08-16T12:00:00+00:00")
    train = _rows(tmp_path / "train.csv")
    promoted_train = {row["ID"]: row["template_family"] for row in train if row["source"] == "goldtest"}
    assert promoted_train == promoted
