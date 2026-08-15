"""Loading, normalization, validation, and deterministic validation splitting."""

from __future__ import annotations

import csv
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


VALID_SENTIMENTS = {"positive", "neutral", "negative"}
VALID_TOPICS = {"smartphone", "television", "refrigerator", "washing_machine"}
VALID_LEVELS = {"limited", "informal", "standard", "advanced", "technical"}
VALID_TASKS = {"sentiment", "topic"}


@dataclass(frozen=True)
class ReviewRecord:
    """A normalized review used either for training or incoming evaluation."""

    ID: int
    text: str
    sentiment: str
    topic: str
    linguistic_level: str
    flagprofanity: int
    input_timestamp: str
    goldtest: int = 0
    source: str = "incoming"
    training_generation: int | None = None


def clean_text(text: str) -> str:
    """Normalize text while preserving lexical signals such as profanity and misspellings."""

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


def _parse_binary(value: str, field: str, row_number: int) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"Row {row_number} {field} must be 0 or 1.")
    return int(value)


def _validate_common(record: ReviewRecord, row_number: int) -> None:
    if record.ID < 1 or not record.text:
        raise ValueError(f"Row {row_number} contains an invalid ID or empty text.")
    if record.sentiment not in VALID_SENTIMENTS:
        raise ValueError(f"Row {row_number} contains an invalid sentiment.")
    if record.topic not in VALID_TOPICS:
        raise ValueError(f"Row {row_number} contains an invalid topic.")
    if record.linguistic_level not in VALID_LEVELS:
        raise ValueError(f"Row {row_number} contains an invalid linguistic_level.")
    _validate_timestamp(record.input_timestamp, row_number)


def _validate_sequence(records: Sequence[ReviewRecord], name: str) -> None:
    if not records:
        raise ValueError(f"{name} is empty.")
    ids = [record.ID for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} IDs must be unique and monotonically increasing.")
    texts = [record.text for record in records]
    if len(texts) != len(set(texts)):
        raise ValueError(f"{name} text must be unique.")


def load_train(path: str | Path) -> list[ReviewRecord]:
    """Load train.csv using the current training schema."""

    dataset_path = Path(path)
    required = {
        "ID",
        "text",
        "sentiment",
        "topic",
        "linguistic_level",
        "flagprofanity",
        "source",
        "training_generation",
        "input_timestamp",
    }
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    records: list[ReviewRecord] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError("train.csv does not use the expected schema.")
        for row_number, row in enumerate(reader, start=2):
            try:
                record_id = int(row["ID"])
                generation = int(row["training_generation"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number} contains an invalid numeric field.") from error
            source = row["source"].strip()
            if source not in {"initial", "goldtest"}:
                raise ValueError(f"Row {row_number} source must be initial or goldtest.")
            record = ReviewRecord(
                ID=record_id,
                text=clean_text(row["text"]),
                sentiment=row["sentiment"].strip(),
                topic=row["topic"].strip(),
                linguistic_level=row["linguistic_level"].strip(),
                flagprofanity=_parse_binary(row["flagprofanity"], "flagprofanity", row_number),
                source=source,
                training_generation=generation,
                input_timestamp=row["input_timestamp"].strip(),
            )
            _validate_common(record, row_number)
            records.append(record)
    _validate_sequence(records, "train.csv")
    return records


def load_incoming(path: str | Path) -> list[ReviewRecord]:
    """Load incoming.csv and map its gold labels onto the common record representation."""

    dataset_path = Path(path)
    required = {
        "ID",
        "text",
        "expected_sentiment",
        "expected_topic",
        "linguistic_level",
        "flagprofanity",
        "goldtest",
        "input_timestamp",
    }
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Incoming dataset not found: {dataset_path}")

    records: list[ReviewRecord] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError("incoming.csv does not use the expected schema.")
        for row_number, row in enumerate(reader, start=2):
            try:
                record_id = int(row["ID"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {row_number} contains an invalid ID.") from error
            record = ReviewRecord(
                ID=record_id,
                text=clean_text(row["text"]),
                sentiment=row["expected_sentiment"].strip(),
                topic=row["expected_topic"].strip(),
                linguistic_level=row["linguistic_level"].strip(),
                flagprofanity=_parse_binary(row["flagprofanity"], "flagprofanity", row_number),
                goldtest=_parse_binary(row["goldtest"], "goldtest", row_number),
                source="incoming",
                input_timestamp=row["input_timestamp"].strip(),
            )
            _validate_common(record, row_number)
            records.append(record)
    _validate_sequence(records, "incoming.csv")
    return records


def label_for(record: ReviewRecord, task: str) -> str:
    if task not in VALID_TASKS:
        raise ValueError("task must be sentiment or topic.")
    return record.sentiment if task == "sentiment" else record.topic


def stratified_validation_split(
    records: Iterable[ReviewRecord],
    task: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ReviewRecord], list[ReviewRecord]]:
    """Create a deterministic label-stratified fit/validation split from train.csv."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    groups: dict[str, list[ReviewRecord]] = defaultdict(list)
    for record in records:
        groups[label_for(record, task)].append(record)
    if len(groups) < 2:
        raise ValueError("At least two labels are required.")

    fit: list[ReviewRecord] = []
    validation: list[ReviewRecord] = []
    for group_index, label in enumerate(sorted(groups)):
        group = list(groups[label])
        if len(group) < 3:
            raise ValueError(f"Label {label} requires at least three training examples.")
        random.Random(seed + group_index * 10_007).shuffle(group)
        validation_size = max(1, int(round(len(group) * validation_fraction)))
        validation_size = min(validation_size, len(group) - 2)
        validation.extend(group[:validation_size])
        fit.extend(group[validation_size:])

    fit.sort(key=lambda record: record.ID)
    validation.sort(key=lambda record: record.ID)
    return fit, validation
