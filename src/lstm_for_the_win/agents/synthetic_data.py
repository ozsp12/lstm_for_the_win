"""Deterministic synthetic product-review data generation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Configuration for the synthetic review agent."""

    agent_name: str = "synthetic-review-generator"
    agent_version: str = "1.0.0"
    language: str = "en"
    seed: int = 42
    sentiment_samples_per_label: int = 36
    topic_samples_per_label: int = 32
    reviews_per_sentiment_topic_pair: int = 4
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
        if min(
            config.sentiment_samples_per_label,
            config.topic_samples_per_label,
            config.reviews_per_sentiment_topic_pair,
        ) < 1:
            raise ValueError("All requested sample counts must be positive.")
        return config


SENTIMENT_PHRASES = {
    "positive": {
        "attributes": (
            "excellent",
            "reliable",
            "impressive",
            "well designed",
            "easy to use",
            "high quality",
        ),
        "outcomes": (
            "works exactly as expected",
            "delivers consistently good results",
            "made the purchase worthwhile",
            "performed perfectly during daily use",
            "exceeded my expectations",
            "has been a pleasure to use",
        ),
    },
    "neutral": {
        "attributes": (
            "standard",
            "ordinary",
            "as described",
            "typical for its category",
            "neither exceptional nor poor",
            "consistent with the specification",
        ),
        "outcomes": (
            "arrived on the scheduled date",
            "includes the listed accessories",
            "performs the basic functions",
            "matches the product description",
            "has an average everyday performance",
            "was delivered in regular packaging",
        ),
    },
    "negative": {
        "attributes": (
            "disappointing",
            "unreliable",
            "poorly designed",
            "difficult to use",
            "low quality",
            "far below expectations",
        ),
        "outcomes": (
            "stopped working during normal use",
            "caused repeated problems",
            "made the purchase frustrating",
            "failed when it was needed",
            "requires constant troubleshooting",
            "did not deliver the promised result",
        ),
    },
}

SENTIMENT_SUBJECTS = (
    "the product",
    "this device",
    "my purchase",
    "the delivered item",
    "the customer experience",
    "this order",
)

SENTIMENT_TEMPLATES = (
    "{subject} is {attribute} and {outcome}",
    "overall {subject} felt {attribute} and {outcome}",
    "I found {subject} {attribute}; it {outcome}",
)

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

TOPIC_TEMPLATES = (
    "the {component} on the {alias} stopped working during normal use",
    "the {alias} reports a problem related to its {component}",
    "the {component} of this {alias} is unreliable",
    "this {alias} needs attention because of the {component}",
    "the {alias} has an issue with the {component}",
)

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


class SyntheticDataAgent:
    """Generate balanced, reproducible, and PII-free product-review datasets."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config

    def _select(self, rows: Iterable[dict[str, str]], count: int, salt: int) -> list[dict[str, str]]:
        candidates = list(rows)
        if count > len(candidates):
            raise ValueError(f"Requested {count} samples from a pool of {len(candidates)}.")
        random.Random(self.config.seed + salt).shuffle(candidates)
        return candidates[:count]

    def sentiment_records(self) -> list[dict[str, str]]:
        """Create a balanced labeled sentiment dataset."""

        output: list[dict[str, str]] = []
        for label_index, (label, phrases) in enumerate(SENTIMENT_PHRASES.items()):
            generic_combinations = itertools.product(
                SENTIMENT_SUBJECTS,
                phrases["attributes"],
                phrases["outcomes"],
                SENTIMENT_TEMPLATES,
            )
            generic_rows = (
                {
                    "text": template.format(
                        subject=subject,
                        attribute=attribute,
                        outcome=outcome,
                    ),
                    "label": label,
                }
                for subject, attribute, outcome, template in generic_combinations
            )
            contextual_rows = (
                {
                    "text": f"the {component} on this {alias} {outcome}",
                    "label": label,
                }
                for topic_phrases in TOPIC_COMPONENTS.values()
                for alias, component, outcome in itertools.product(
                    topic_phrases["aliases"],
                    topic_phrases["components"],
                    REVIEW_OUTCOMES[label],
                )
            )
            generic_count = self.config.sentiment_samples_per_label // 2
            contextual_count = self.config.sentiment_samples_per_label - generic_count
            output.extend(
                self._select(generic_rows, generic_count, salt=100 + label_index)
            )
            output.extend(
                self._select(
                    contextual_rows,
                    contextual_count,
                    salt=150 + label_index,
                )
            )
        random.Random(self.config.seed + 199).shuffle(output)
        return output

    def topic_records(self) -> list[dict[str, str]]:
        """Create a balanced labeled product-topic dataset."""

        output: list[dict[str, str]] = []
        for label_index, (label, phrases) in enumerate(TOPIC_COMPONENTS.items()):
            combinations = itertools.product(
                phrases["aliases"],
                phrases["components"],
                TOPIC_TEMPLATES,
            )
            rows = (
                {
                    "text": template.format(alias=alias, component=component),
                    "label": label,
                }
                for alias, component, template in combinations
            )
            output.extend(
                self._select(
                    rows,
                    self.config.topic_samples_per_label,
                    salt=200 + label_index,
                )
            )
        random.Random(self.config.seed + 299).shuffle(output)
        return output

    def review_records(self) -> list[dict[str, str]]:
        """Create stakeholder-facing reviews for dual-model inference."""

        output: list[dict[str, str]] = []
        pair_index = 0
        for topic, phrases in TOPIC_COMPONENTS.items():
            for sentiment, outcomes in REVIEW_OUTCOMES.items():
                combinations = itertools.product(
                    phrases["aliases"],
                    phrases["components"],
                    outcomes,
                )
                rows = (
                    {
                        "text": f"the {component} on this {alias} {outcome}",
                        "expected_sentiment": sentiment,
                        "expected_topic": topic,
                    }
                    for alias, component, outcome in combinations
                )
                output.extend(
                    self._select(
                        rows,
                        self.config.reviews_per_sentiment_topic_pair,
                        salt=300 + pair_index,
                    )
                )
                pair_index += 1
        random.Random(self.config.seed + 399).shuffle(output)
        return output

    def write(self, output_dir: str | Path, overwrite: bool = False) -> Path:
        """Write all input datasets and a deterministic provenance manifest."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        files = {
            "sentiment_samples.csv": (
                self.sentiment_records(),
                ("text", "label"),
            ),
            "topic_samples.csv": (
                self.topic_records(),
                ("text", "label"),
            ),
            "reviews.csv": (
                self.review_records(),
                ("text", "expected_sentiment", "expected_topic"),
            ),
        }

        existing = [name for name in files if (destination / name).exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "Synthetic input already exists. Use overwrite=True to regenerate it."
            )

        counts: dict[str, int] = {}
        hashes: dict[str, str] = {}
        for filename, (rows, fieldnames) in files.items():
            path = destination / filename
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            counts[filename] = len(rows)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

        manifest = {
            "generated_by": self.config.agent_name,
            "agent_version": self.config.agent_version,
            "config": asdict(self.config),
            "record_counts": counts,
            "sha256": hashes,
        }
        manifest_path = destination / "input_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path
