"""Text data loading, cleaning, and splitting."""

from __future__ import annotations

import csv
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable


Record = tuple[str, str]


def clean_text(text: str) -> str:
    """Normalize text without external language resources."""

    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", normalized)
    normalized = re.sub(r"[@#]\w+", " ", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def load_dataset(path: str | Path) -> list[Record]:
    """Load a CSV file with the required ``text`` and ``topic`` columns."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records: list[Record] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"text", "topic"}.issubset(reader.fieldnames):
            raise ValueError("The dataset must contain the 'text' and 'topic' columns.")

        for row_number, row in enumerate(reader, start=2):
            text = clean_text(row["text"])
            topic = row["topic"].strip()
            if not text or not topic:
                raise ValueError(f"Row {row_number} contains an empty text or topic value.")
            records.append((text, topic))

    validate_records(records)
    return records


def validate_records(records: Iterable[Record]) -> None:
    """Check minimum dataset size and class balance for training and testing."""

    counts: dict[str, int] = defaultdict(int)
    total = 0
    for _, topic in records:
        counts[topic] += 1
        total += 1

    if total == 0:
        raise ValueError("The dataset is empty.")
    if len(counts) < 2:
        raise ValueError("At least two topics are required.")
    if min(counts.values()) < 5:
        raise ValueError("Each topic requires at least five examples.")


def stratified_split(
    records: Iterable[Record],
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Split records while preserving every topic in train and test sets."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")

    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)

    random_generator = random.Random(seed)
    train_records: list[Record] = []
    test_records: list[Record] = []

    for topic in sorted(grouped):
        topic_records = grouped[topic][:]
        random_generator.shuffle(topic_records)
        test_size = max(1, round(len(topic_records) * test_fraction))
        test_records.extend(topic_records[:test_size])
        train_records.extend(topic_records[test_size:])

    random_generator.shuffle(train_records)
    random_generator.shuffle(test_records)
    return train_records, test_records
