from __future__ import annotations

import csv
import json
from pathlib import Path

from lstm_for_the_win.agents import SyntheticDataAgent, SyntheticDataConfig
from lstm_for_the_win.benchmark import ensure_immutable_benchmark


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_benchmark_is_bootstrapped_once_and_never_promoted(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    config = SyntheticDataConfig(initial_train_rows=1200, incoming_rows=1200, incoming_rows_jitter=0, goldtest_fraction=0.50, vary_counts=False)
    agent = SyntheticDataAgent(config)
    agent.initialize(input_dir, "2026-08-15T12:00:00+00:00")
    initial_incoming = _rows(input_dir / "incoming.csv")
    expected_benchmark_rows = sum(row["goldtest"] == "0" for row in initial_incoming)

    benchmark_path, manifest = ensure_immutable_benchmark(input_dir)
    original = benchmark_path.read_bytes()
    benchmark = _rows(benchmark_path)
    assert len(benchmark) == expected_benchmark_rows
    assert len(benchmark) >= 500
    assert {row["goldtest"] for row in benchmark} == {"0"}
    assert manifest["source_generation"] == 0
    assert manifest["rows"] == expected_benchmark_rows

    agent.advance(input_dir, "2026-08-16T12:00:00+00:00")
    same_path, same_manifest = ensure_immutable_benchmark(input_dir)
    assert same_path.read_bytes() == original
    assert same_manifest == json.loads((input_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))

    train = _rows(input_dir / "train.csv")
    benchmark_ids = {row["ID"] for row in benchmark}
    benchmark_texts = {row["text"] for row in benchmark}
    assert not benchmark_ids.intersection(row["ID"] for row in train)
    assert not benchmark_texts.intersection(row["text"] for row in train)
