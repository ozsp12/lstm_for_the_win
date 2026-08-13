"""Text data loading, cleaning, validation, and explicit splitting."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Record:
    """One traceable input row used by a classifier."""

    ID: int
    text: str
    label: str
    type: str
    input_timestamp: str


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


def _validate_timestamp(value: str, row_number: int) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Row {row_number} contains an invalid input_timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"Row {row_number} input_timestamp must include a timezone.")


def load_dataset(path: str | Path) -> list[Record]:
    """Load a versioned CSV with explicit train/test membership."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records: list[Record] = []
    required = {"ID", "text", "label", "type", "input_timestamp"}
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "The dataset must contain ID, text, label, type, and input_timestamp."
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                record_id = int(row["ID"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number} contains an invalid ID.") from error
            text = clean_text(row["text"])
            label = row["label"].strip()
            row_type = row["type"].strip()
            input_timestamp = row["input_timestamp"].strip()
            if record_id < 1 or not text or not label:
                raise ValueError(f"Row {row_number} contains an invalid ID, text, or label.")
            if row_type not in {"train", "test"}:
                raise ValueError(f"Row {row_number} type must be train or test.")
            _validate_timestamp(input_timestamp, row_number)
            records.append(
                Record(
                    ID=record_id,
                    text=text,
                    label=label,
                    type=row_type,
                    input_timestamp=input_timestamp,
                )
            )

    validate_records(records)
    return records


def validate_records(records: Iterable[Record]) -> None:
    """Check IDs, label coverage, and explicit train/test class balance."""

    rows = list(records)
    if not rows:
        raise ValueError("The dataset is empty.")

    ids = [record.ID for record in rows]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("IDs must be unique and monotonically increasing.")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in rows:
        counts[record.type][record.label] += 1
    if set(counts) != {"train", "test"}:
        raise ValueError("The dataset must contain both train and test rows.")
    labels = set(counts["train"]) | set(counts["test"])
    if len(labels) < 2:
        raise ValueError("At least two labels are required.")
    for row_type in ("train", "test"):
        if set(counts[row_type]) != labels:
            raise ValueError(f"Every label must be represented in {row_type} rows.")
        if min(counts[row_type].values()) < 5:
            raise ValueError(f"Each label requires at least five {row_type} examples.")


def split_by_type(records: Iterable[Record]) -> tuple[list[Record], list[Record]]:
    """Use the versioned type column instead of creating a random holdout split."""

    rows = list(records)
    train_records = [record for record in rows if record.type == "train"]
    test_records = [record for record in rows if record.type == "test"]
    if not train_records or not test_records:
        raise ValueError("Both train and test records are required.")
    return train_records, test_records
