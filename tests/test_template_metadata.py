from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.template_metadata import TEMPLATE_FAMILIES, ensure_template_metadata


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generated_template_family_requires_no_backfill(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, vary_counts=False)
    SyntheticDataAgent(config).initialize(input_dir, "2026-08-15T12:00:00+00:00")

    changed = ensure_template_metadata(input_dir)
    assert changed == {"train.csv": False, "incoming.csv": False}
    for name in ("train.csv", "incoming.csv"):
        rows = _rows(input_dir / name)
        assert rows
        assert {row["template_family"] for row in rows}.issubset(set(TEMPLATE_FAMILIES))

    manifest = json.loads((input_dir / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["template_family_metadata"]["materialized"] is True
    assert manifest["template_family_metadata"]["origin"] == "generated_at_render_time"
    assert manifest["incoming_template_family_counts"]
    for name in ("train.csv", "incoming.csv"):
        digest = hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
        assert manifest["sha256"][name] == digest
