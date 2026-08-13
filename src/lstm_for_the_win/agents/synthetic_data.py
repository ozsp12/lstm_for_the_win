"""Deterministic synthetic product-review data generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


REVIEW_FIELDS = (
    "ID",
    "text",
    "expected_sentiment",
    "expected_topic",
    "type",
    "input_timestamp",
)
SAMPLE_FIELDS = ("ID", "text", "label", "type", "input_timestamp")
VALID_TYPES = ("train", "test")


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for the synthetic review agent."""

    agent_name: str = "synthetic-review-generator"
    agent_version: str = "2.0.0"
    language: str = "en"
    seed: int = 42
    initial_train_rows: int = 500
    initial_test_rows: int = 500
    append_train_rows: int = 100
    append_test_rows: int = 100
    synthetic_only: bool = True
    allow_personal_data: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "SyntheticDataConfig":
        """Load and validate agent configuration from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**payload)
        if config.language != "en":
            raise ValueError("The current phrase library supports English only.")
        if not config.synthetic_only or config.allow_personal_data:
            raise ValueError("The agent must remain synthetic-only and PII-free.")
        counts = (
            config.initial_train_rows,
            config.initial_test_rows,
            config.append_train_rows,
            config.append_test_rows,
        )
        if min(counts) < 1:
            raise ValueError("All requested row counts must be positive.")
        if config.initial_train_rows + config.initial_test_rows != 1_000:
            raise ValueError("The initial review dataset must contain exactly 1,000 rows.")
        return config


TOPIC_COMPONENTS = {
    "smartphone": {
        "aliases": ("smartphone", "phone", "mobile device", "handset"),
        "components": (
            "battery",
            "touchscreen",
            "camera",
            "mobile signal",
            "charging port",
            "speaker",
            "app storage",
            "fingerprint reader",
        ),
    },
    "television": {
        "aliases": ("television", "tv", "smart tv", "display"),
        "components": (
            "screen",
            "remote control",
            "hdmi input",
            "sound output",
            "streaming menu",
            "wifi connection",
            "picture panel",
            "channel tuner",
        ),
    },
    "refrigerator": {
        "aliases": ("refrigerator", "fridge", "freezer appliance", "cooling unit"),
        "components": (
            "temperature control",
            "freezer compartment",
            "door seal",
            "ice maker",
            "cooling fan",
            "interior light",
            "water dispenser",
            "storage drawer",
        ),
    },
    "washing_machine": {
        "aliases": ("washing machine", "washer", "laundry machine", "laundry appliance"),
        "components": (
            "spin cycle",
            "water inlet",
            "detergent drawer",
            "drain pump",
            "control panel",
            "drum",
            "door lock",
            "rinse program",
        ),
    },
}

REVIEW_OUTCOMES = {
    "positive": (
        "is excellent and works perfectly",
        "is reliable and easy to use",
        "performs better than expected",
        "has delivered consistently good results",
    ),
    "neutral": (
        "matches the standard specification",
        "performs the expected basic function",
        "arrived in ordinary packaging",
        "is typical for this product category",
    ),
    "negative": (
        "is unreliable and fails during use",
        "is disappointing and causes repeated problems",
        "stopped working and needs support",
        "performs far below expectations",
    ),
}

CONTEXTS = (
    "during the initial setup",
    "after a week of regular use",
    "during everyday household use",
    "after following the supplied instructions",
    "while testing the main functions",
    "during a routine product check",
    "after several normal operating cycles",
    "while comparing it with the specification",
    "during a typical weekday",
    "after the latest usage session",
)

DETAILS = (
    "the experience has remained consistent",
    "the observation was easy to reproduce",
    "the result was clear during the review",
    "the same behavior appeared more than once",
    "the reviewer noted it during normal operation",
    "the outcome matched the overall impression",
    "the finding was recorded for product monitoring",
    "the behavior was visible without special testing",
    "the review reflects an ordinary customer scenario",
    "the result was included in the product assessment",
)

REVIEW_TEMPLATES = (
    "{context}, the {component} on this {alias} {outcome}; {detail}.",
    "{context}, this {alias}'s {component} {outcome}; {detail}.",
    "{context}, I found that the {component} of the {alias} {outcome}; {detail}.",
    "{context}, the review showed that this {alias} {component} {outcome}; {detail}.",
    "{context}, my assessment is that the {alias} {component} {outcome}; {detail}.",
)


def _validate_timestamp(value: str) -> str:
    """Require an ISO-8601 timestamp with an explicit timezone."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("input_timestamp must be a valid ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise ValueError("input_timestamp must include a timezone.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class SyntheticDataAgent:
    """Generate balanced, reproducible, and PII-free product-review datasets."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config

    @staticmethod
    def _review_for_id(review_id: int, row_type: str, input_timestamp: str) -> dict[str, str]:
        """Map a monotonic ID to one unique deterministic review."""

        if review_id < 1:
            raise ValueError("Review IDs must be positive.")
        if row_type not in VALID_TYPES:
            raise ValueError("Review type must be train or test.")

        index = review_id - 1
        sentiments = tuple(REVIEW_OUTCOMES)
        sentiment = sentiments[index % len(sentiments)]
        index //= len(sentiments)

        topics = tuple(TOPIC_COMPONENTS)
        topic = topics[index % len(topics)]
        index //= len(topics)

        topic_phrases = TOPIC_COMPONENTS[topic]
        aliases = topic_phrases["aliases"]
        alias = aliases[index % len(aliases)]
        index //= len(aliases)

        components = topic_phrases["components"]
        component = components[index % len(components)]
        index //= len(components)

        outcomes = REVIEW_OUTCOMES[sentiment]
        outcome = outcomes[index % len(outcomes)]
        index //= len(outcomes)

        context = CONTEXTS[index % len(CONTEXTS)]
        index //= len(CONTEXTS)
        detail = DETAILS[index % len(DETAILS)]
        index //= len(DETAILS)
        template = REVIEW_TEMPLATES[index % len(REVIEW_TEMPLATES)]
        index //= len(REVIEW_TEMPLATES)
        if index:
            raise ValueError("The deterministic phrase library has been exhausted.")

        return {
            "ID": str(review_id),
            "text": template.format(
                context=context,
                component=component,
                alias=alias,
                outcome=outcome,
                detail=detail,
            ),
            "expected_sentiment": sentiment,
            "expected_topic": topic,
            "type": row_type,
            "input_timestamp": input_timestamp,
        }

    @staticmethod
    def _validate_existing(rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        if set(rows[0]) != set(REVIEW_FIELDS):
            raise ValueError("reviews.csv does not use the current versioned schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Existing review IDs must be unique and monotonically increasing.")
        if ids != list(range(1, ids[-1] + 1)):
            raise ValueError("Existing review IDs must be contiguous.")
        if any(row["type"] not in VALID_TYPES for row in rows):
            raise ValueError("Existing review types must be train or test.")
        if len({row["text"] for row in rows}) != len(rows):
            raise ValueError("Existing review text must be unique.")
        for row in rows:
            _validate_timestamp(row["input_timestamp"])

    def initialize(
        self,
        output_dir: str | Path,
        input_timestamp: str,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Create the initial 1,000-row versioned dataset."""

        timestamp = _validate_timestamp(input_timestamp)
        destination = Path(output_dir)
        reviews_path = destination / "reviews.csv"
        if reviews_path.exists() and not overwrite:
            raise FileExistsError("Input data already exists. Use overwrite=True to initialize it.")

        rows: list[dict[str, str]] = []
        next_id = 1
        for row_type, count in (
            ("train", self.config.initial_train_rows),
            ("test", self.config.initial_test_rows),
        ):
            rows.extend(
                self._review_for_id(review_id, row_type, timestamp)
                for review_id in range(next_id, next_id + count)
            )
            next_id += count
        return self._write_all(destination, rows)

    def append(
        self,
        output_dir: str | Path,
        input_timestamp: str,
    ) -> Path:
        """Append one configured train/test batch with contiguous IDs."""

        timestamp = _validate_timestamp(input_timestamp)
        destination = Path(output_dir)
        reviews_path = destination / "reviews.csv"
        if not reviews_path.is_file():
            raise FileNotFoundError("Initialize the input dataset before appending a batch.")

        rows = _read_csv(reviews_path)
        self._validate_existing(rows)
        next_id = int(rows[-1]["ID"]) + 1 if rows else 1
        for row_type, count in (
            ("train", self.config.append_train_rows),
            ("test", self.config.append_test_rows),
        ):
            rows.extend(
                self._review_for_id(review_id, row_type, timestamp)
                for review_id in range(next_id, next_id + count)
            )
            next_id += count
        return self._write_all(destination, rows)

    def _write_all(self, destination: Path, reviews: list[dict[str, str]]) -> Path:
        """Publish the review table and its two aligned label projections."""

        self._validate_existing(reviews)
        destination.mkdir(parents=True, exist_ok=True)
        sentiment_rows = [
            {
                "ID": row["ID"],
                "text": row["text"],
                "label": row["expected_sentiment"],
                "type": row["type"],
                "input_timestamp": row["input_timestamp"],
            }
            for row in reviews
        ]
        topic_rows = [
            {
                "ID": row["ID"],
                "text": row["text"],
                "label": row["expected_topic"],
                "type": row["type"],
                "input_timestamp": row["input_timestamp"],
            }
            for row in reviews
        ]
        files: dict[str, tuple[list[dict[str, str]], Sequence[str]]] = {
            "reviews.csv": (reviews, REVIEW_FIELDS),
            "sentiment_samples.csv": (sentiment_rows, SAMPLE_FIELDS),
            "topic_samples.csv": (topic_rows, SAMPLE_FIELDS),
        }
        for filename, (rows, fieldnames) in files.items():
            _write_csv(destination / filename, rows, fieldnames)

        counts = Counter(row["type"] for row in reviews)
        manifest = {
            "generated_by": self.config.agent_name,
            "agent_version": self.config.agent_version,
            "config": asdict(self.config),
            "record_counts": {name: len(rows) for name, (rows, _) in files.items()},
            "review_type_counts": dict(sorted(counts.items())),
            "id_range": {"first": int(reviews[0]["ID"]), "last": int(reviews[-1]["ID"])},
            "latest_input_timestamp": reviews[-1]["input_timestamp"],
            "sha256": {
                filename: hashlib.sha256((destination / filename).read_bytes()).hexdigest()
                for filename in files
            },
        }
        manifest_path = destination / "input_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path
