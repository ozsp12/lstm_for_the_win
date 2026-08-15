from __future__ import annotations

import csv
from pathlib import Path

from lstm_for_the_win.classification.data import load_incoming


def test_legacy_incoming_schema_gets_safe_metadata_defaults(tmp_path: Path) -> None:
    path = tmp_path / "incoming.csv"
    fields = [
        "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
        "flagprofanity", "goldtest", "input_timestamp",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "ID": 1,
            "text": "ngl this phone works 😅",
            "expected_sentiment": "positive",
            "expected_topic": "smartphone",
            "linguistic_level": "informal",
            "flagprofanity": 0,
            "goldtest": 0,
            "input_timestamp": "2026-08-15T12:00:00+00:00",
        })
    record = load_incoming(path)[0]
    assert record.hasemoji == 1
    assert record.hasslang == 1
    assert record.hasspellingerror == 0
    assert record.mixed_sentiment == 0
