"""Immutable synthetic benchmark bootstrap, provenance, and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

BENCHMARK_FILE = "benchmark.csv"
BENCHMARK_MANIFEST_FILE = "benchmark_manifest.json"
MIN_BENCHMARK_ROWS = 500


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> None:
    if len(rows) < MIN_BENCHMARK_ROWS:
        raise ValueError(f"benchmark.csv must contain at least {MIN_BENCHMARK_ROWS} rows.")
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


def _manifest(root: Path, benchmark_path: Path) -> dict[str, object]:
    input_manifest_path = root / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8")) if input_manifest_path.is_file() else {}
    return {
        "schema_version": "1.0.0",
        "immutable": True,
        "source": "non-gold rows from the incoming batch used to bootstrap benchmark support",
        "source_generation": int(input_manifest.get("generation", 0)),
        "created_from_agent_version": input_manifest.get("agent_version"),
        "rows": len(_read(benchmark_path)),
        "sha256": _sha256(benchmark_path),
    }


def ensure_immutable_benchmark(input_dir: str | Path) -> tuple[Path, dict[str, object]]:
    """Create benchmark.csv once from non-gold incoming rows and retain provenance."""

    root = Path(input_dir)
    train_path = root / "train.csv"
    incoming_path = root / "incoming.csv"
    benchmark_path = root / BENCHMARK_FILE
    manifest_path = root / BENCHMARK_MANIFEST_FILE
    train_rows = _read(train_path)

    if benchmark_path.is_file():
        _validate(_read(benchmark_path), train_rows)
        if not manifest_path.is_file():
            raise ValueError("benchmark.csv exists without benchmark_manifest.json; rebuild the benchmark explicitly.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != _sha256(benchmark_path):
            raise ValueError("benchmark.csv no longer matches benchmark_manifest.json.")
        return benchmark_path, manifest

    with incoming_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("incoming.csv has no header.")
        rows = [row for row in reader if row.get("goldtest") == "0"]
        fieldnames = list(reader.fieldnames)
    if len(rows) < MIN_BENCHMARK_ROWS:
        raise ValueError(
            f"At least {MIN_BENCHMARK_ROWS} non-gold incoming rows are required to bootstrap benchmark.csv."
        )
    _validate(rows, train_rows)
    _write(benchmark_path, rows, fieldnames)
    manifest = _manifest(root, benchmark_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return benchmark_path, manifest
