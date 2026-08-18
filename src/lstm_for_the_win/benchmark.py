"""Immutable synthetic benchmark bootstrap and validation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

BENCHMARK_FILE = "benchmark.csv"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _validate(rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("benchmark.csv cannot be empty.")
    ids = [row["ID"] for row in rows]
    texts = [row["text"] for row in rows]
    if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
        raise ValueError("benchmark.csv must contain unique IDs and text.")
    if any(row.get("goldtest") != "0" for row in rows):
        raise ValueError("benchmark.csv must contain only rows that are never promoted to training.")
    train_ids = {row["ID"] for row in train_rows}
    train_texts = {row["text"] for row in train_rows}
    if train_ids & set(ids) or train_texts & set(texts):
        raise ValueError("benchmark.csv must remain disjoint from train.csv.")


def ensure_immutable_benchmark(input_dir: str | Path) -> Path:
    """Create benchmark.csv once from non-gold incoming rows and never rewrite it."""

    root = Path(input_dir)
    train_path = root / "train.csv"
    incoming_path = root / "incoming.csv"
    benchmark_path = root / BENCHMARK_FILE
    train_rows = _read(train_path)

    if benchmark_path.is_file():
        _validate(_read(benchmark_path), train_rows)
        return benchmark_path

    with incoming_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("incoming.csv has no header.")
        rows = [row for row in reader if row.get("goldtest") == "0"]
        fieldnames = list(reader.fieldnames)
    if len(rows) < 1000:
        raise ValueError("At least 1000 non-gold incoming rows are required to bootstrap benchmark.csv.")
    _validate(rows, train_rows)
    _write(benchmark_path, rows, fieldnames)
    return benchmark_path
