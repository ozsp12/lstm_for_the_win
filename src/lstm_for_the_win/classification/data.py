"""Review loading, normalization, validation, and leakage-aware validation splitting."""

from __future__ import annotations

import csv
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from ..template_metadata import TEMPLATE_FAMILIES, infer_template_family

VALID_SENTIMENTS = {"positive", "neutral", "negative"}
VALID_TOPICS = {"smartphone", "television", "refrigerator", "washing_machine"}
VALID_LEVELS = {"limited", "informal", "standard", "advanced", "technical"}
VALID_LENGTHS = {"short", "medium", "long"}
VALID_TASKS = {"sentiment", "topic"}
SLANG = {"ngl", "tbh", "idk", "imo", "kinda", "lowkey", "fr", "lol", "wtf", "gonna"}


@dataclass(frozen=True)
class ReviewRecord:
    ID: int
    text: str
    sentiment: str
    topic: str
    linguistic_level: str
    flagprofanity: int
    input_timestamp: str
    hasemoji: int = 0
    hasspellingerror: int = 0
    hasslang: int = 0
    length_class: str = "short"
    mixed_sentiment: int = 0
    goldtest: int = 0
    source: str = "incoming"
    training_generation: int | None = None
    template_family: str = "context_component"


def _hasemoji(text: str) -> bool:
    return any(ord(char) >= 0x1F000 for char in text)


def _length(text: str) -> str:
    count = len(text.split())
    return "short" if count < 14 else "medium" if count < 30 else "long"


def _hasslang(text: str) -> bool:
    return bool(set(re.findall(r"[a-z']+", text.lower())) & SLANG)


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char)).lower()
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", normalized)
    normalized = re.sub(r"[@#]\w+", " ", normalized)
    normalized = "".join(
        char if char.isalpha() or char.isspace() or char == "'" or ord(char) >= 0x1F000 else " "
        for char in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _bin(value: str | None, field: str, row: int, default: int = 0) -> int:
    if value in (None, ""):
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"Row {row} {field} must be 0 or 1.")
    return int(value)


def _validate_timestamp(value: str, row: int) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Row {row} contains an invalid input_timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError(f"Row {row} input_timestamp must include a timezone.")


def _template_family(row: dict[str, str], raw: str) -> str:
    family = row.get("template_family", "").strip() or infer_template_family(raw)
    if family not in TEMPLATE_FAMILIES:
        raise ValueError(f"Unsupported template_family: {family}")
    return family


def _metadata(row: dict[str, str], raw: str, number: int) -> dict[str, int | str]:
    return {
        "hasemoji": _bin(row.get("hasemoji"), "hasemoji", number, int(_hasemoji(raw))),
        "hasspellingerror": _bin(row.get("hasspellingerror"), "hasspellingerror", number),
        "hasslang": _bin(row.get("hasslang"), "hasslang", number, int(_hasslang(raw))),
        "length_class": row.get("length_class", "").strip() or _length(raw),
        "mixed_sentiment": _bin(row.get("mixed_sentiment"), "mixed_sentiment", number),
        "template_family": _template_family(row, raw),
    }


def _validate_record(record: ReviewRecord, row: int) -> None:
    if record.ID < 1 or not record.text:
        raise ValueError(f"Row {row} contains an invalid ID or empty text.")
    if (
        record.sentiment not in VALID_SENTIMENTS
        or record.topic not in VALID_TOPICS
        or record.linguistic_level not in VALID_LEVELS
        or record.length_class not in VALID_LENGTHS
        or record.template_family not in TEMPLATE_FAMILIES
    ):
        raise ValueError(f"Row {row} contains invalid review metadata.")
    _validate_timestamp(record.input_timestamp, row)


def _validate_sequence(records: Sequence[ReviewRecord], name: str) -> None:
    if not records:
        raise ValueError(f"{name} is empty.")
    ids = [record.ID for record in records]
    texts = [record.text for record in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} IDs must be unique and monotonically increasing.")
    if len(texts) != len(set(texts)):
        raise ValueError(f"{name} text must be unique.")


def load_train(path: str | Path) -> list[ReviewRecord]:
    source_path = Path(path)
    required = {
        "ID", "text", "sentiment", "topic", "linguistic_level", "flagprofanity",
        "source", "training_generation", "input_timestamp",
    }
    if not source_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {source_path}")

    records: list[ReviewRecord] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("train.csv does not use the expected schema.")
        for number, row in enumerate(reader, start=2):
            raw = row["text"]
            source = row["source"].strip()
            if source not in {"initial", "goldtest"}:
                raise ValueError(f"Row {number} source must be initial or goldtest.")
            try:
                review_id = int(row["ID"])
                generation = int(row["training_generation"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {number} contains an invalid numeric field.") from error
            record = ReviewRecord(
                review_id,
                clean_text(raw),
                row["sentiment"].strip(),
                row["topic"].strip(),
                row["linguistic_level"].strip(),
                _bin(row["flagprofanity"], "flagprofanity", number),
                row["input_timestamp"].strip(),
                source=source,
                training_generation=generation,
                **_metadata(row, raw, number),
            )
            _validate_record(record, number)
            records.append(record)
    _validate_sequence(records, "train.csv")
    return records


def load_incoming(path: str | Path) -> list[ReviewRecord]:
    source_path = Path(path)
    required = {
        "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
        "flagprofanity", "goldtest", "input_timestamp",
    }
    if not source_path.is_file():
        raise FileNotFoundError(f"Incoming dataset not found: {source_path}")

    records: list[ReviewRecord] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("incoming.csv does not use the expected schema.")
        for number, row in enumerate(reader, start=2):
            raw = row["text"]
            try:
                review_id = int(row["ID"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Row {number} contains an invalid ID.") from error
            record = ReviewRecord(
                review_id,
                clean_text(raw),
                row["expected_sentiment"].strip(),
                row["expected_topic"].strip(),
                row["linguistic_level"].strip(),
                _bin(row["flagprofanity"], "flagprofanity", number),
                row["input_timestamp"].strip(),
                goldtest=_bin(row["goldtest"], "goldtest", number),
                **_metadata(row, raw, number),
            )
            _validate_record(record, number)
            records.append(record)
    _validate_sequence(records, "incoming.csv")
    return records


def label_for(record: ReviewRecord, task: str) -> str:
    if task not in VALID_TASKS:
        raise ValueError("task must be sentiment or topic.")
    return record.sentiment if task == "sentiment" else record.topic


def _random_stratified_split(
    records: Sequence[ReviewRecord],
    task: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ReviewRecord], list[ReviewRecord]]:
    groups: dict[str, list[ReviewRecord]] = defaultdict(list)
    for record in records:
        groups[label_for(record, task)].append(record)
    if len(groups) < 2:
        raise ValueError("At least two labels are required.")

    fit: list[ReviewRecord] = []
    validation: list[ReviewRecord] = []
    for index, label in enumerate(sorted(groups)):
        group = list(groups[label])
        if len(group) < 3:
            raise ValueError(f"Label {label} requires at least three training examples.")
        random.Random(seed + index * 10_007).shuffle(group)
        size = min(max(1, int(round(len(group) * validation_fraction))), len(group) - 2)
        validation.extend(group[:size])
        fit.extend(group[size:])
    fit.sort(key=lambda record: record.ID)
    validation.sort(key=lambda record: record.ID)
    return fit, validation


def validation_split(
    records: Iterable[ReviewRecord],
    task: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ReviewRecord], list[ReviewRecord], dict[str, object]]:
    """Prefer a persisted whole-template holdout; fall back to label-stratified random splitting."""

    if task not in VALID_TASKS:
        raise ValueError("task must be sentiment or topic.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    materialized = list(records)
    if not materialized:
        raise ValueError("At least one training record is required.")
    all_labels = set(label_for(record, task) for record in materialized)
    total_by_label = Counter(label_for(record, task) for record in materialized)
    families: dict[str, list[ReviewRecord]] = defaultdict(list)
    for record in materialized:
        families[record.template_family].append(record)

    target = len(materialized) * validation_fraction
    candidates: list[tuple[float, str, list[ReviewRecord]]] = []
    for family, group in families.items():
        validation_labels = Counter(label_for(record, task) for record in group)
        if set(validation_labels) != all_labels:
            continue
        if any(total_by_label[label] - validation_labels[label] < 2 for label in all_labels):
            continue
        candidates.append((abs(len(group) - target), family, group))

    if len(families) >= 3 and candidates:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[0])
        _, heldout_family, heldout = candidates[0]
        validation_ids = {record.ID for record in heldout}
        fit = sorted((record for record in materialized if record.ID not in validation_ids), key=lambda record: record.ID)
        validation = sorted(heldout, key=lambda record: record.ID)
        return fit, validation, {
            "method": "template_family_grouped",
            "family_source": "persisted_metadata",
            "heldout_families": [heldout_family],
            "requested_fraction": validation_fraction,
            "actual_fraction": len(validation) / len(materialized),
        }

    fit, validation = _random_stratified_split(materialized, task, validation_fraction, seed)
    return fit, validation, {
        "method": "stratified_random_fallback",
        "family_source": "persisted_metadata",
        "heldout_families": [],
        "requested_fraction": validation_fraction,
        "actual_fraction": len(validation) / len(materialized),
    }


def stratified_validation_split(
    records: Iterable[ReviewRecord],
    task: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ReviewRecord], list[ReviewRecord]]:
    """Backward-compatible wrapper returning only fit and validation records."""

    fit, validation, _ = validation_split(records, task, validation_fraction, seed)
    return fit, validation
