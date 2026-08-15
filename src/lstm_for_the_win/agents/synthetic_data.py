"""Balanced synthetic review generation for continual-learning experiments."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

SENTIMENTS = ("positive", "neutral", "negative")
TOPICS = ("smartphone", "television", "refrigerator", "washing_machine")
LINGUISTIC_LEVELS = ("limited", "informal", "standard", "advanced", "technical")

TRAIN_FIELDS = (
    "ID", "text", "sentiment", "topic", "linguistic_level", "flagprofanity",
    "source", "training_generation", "input_timestamp",
)
INCOMING_FIELDS = (
    "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
    "flagprofanity", "goldtest", "input_timestamp",
)


@dataclass(frozen=True)
class SyntheticDataConfig:
    agent_name: str = "synthetic-review-generator"
    agent_version: str = "3.0.0"
    language: str = "en"
    seed: int = 42
    initial_train_rows: int = 6_000
    incoming_rows: int = 1_200
    profanity_fraction: float = 0.25
    goldtest_fraction: float = 0.20
    validation_fraction: float = 0.15
    synthetic_only: bool = True
    allow_personal_data: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "SyntheticDataConfig":
        config = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    def validate(self) -> None:
        if self.language != "en":
            raise ValueError("The current language library supports English only.")
        if not self.synthetic_only or self.allow_personal_data:
            raise ValueError("The generator must remain synthetic-only and PII-free.")
        strata = len(SENTIMENTS) * len(TOPICS) * len(LINGUISTIC_LEVELS)
        if any(count < strata or count % strata for count in (self.initial_train_rows, self.incoming_rows)):
            raise ValueError(f"Row counts must be positive multiples of {strata}.")
        if any(not 0.0 < value < 1.0 for value in (
            self.profanity_fraction, self.goldtest_fraction, self.validation_fraction
        )):
            raise ValueError("Fractions must be strictly between 0 and 1.")


TOPIC_LANGUAGE = {
    "smartphone": {
        "train": (("smartphone", "phone", "handset"), ("battery", "camera", "touchscreen", "charging port")),
        "incoming": (("cell phone", "mobile", "pocket device"), ("power cell", "rear lens", "touch panel", "usb port")),
    },
    "television": {
        "train": (("television", "tv", "smart tv"), ("screen", "remote control", "hdmi input", "sound output")),
        "incoming": (("television set", "screen unit", "video panel"), ("picture panel", "controller", "video input", "built in audio")),
    },
    "refrigerator": {
        "train": (("refrigerator", "fridge", "cooling unit"), ("temperature control", "door seal", "ice maker", "cooling fan")),
        "incoming": (("cold storage unit", "kitchen fridge", "food cooler"), ("thermostat", "gasket", "ice tray system", "compressor fan")),
    },
    "washing_machine": {
        "train": (("washing machine", "washer", "laundry machine"), ("spin cycle", "water inlet", "detergent drawer", "drain pump")),
        "incoming": (("clothes washer", "wash unit", "front loader"), ("spin program", "fill valve", "soap tray", "drainage motor")),
    },
}

ASSESSMENTS = {
    "train": {
        "positive": ("works reliably", "is better than expected", "has been consistently good", "does its job very well"),
        "neutral": ("works as expected", "is fairly ordinary", "meets the basic specification", "does the job and little more"),
        "negative": ("keeps failing", "works far below expectations", "has become unreliable", "causes repeated problems"),
    },
    "incoming": {
        "positive": ("still performs well under different use", "has not let me down", "is surprisingly dependable", "remains better than I expected"),
        "neutral": ("is neither impressive nor bad", "remains basically average", "behaves like a normal unit", "shows no meaningful change"),
        "negative": ("fails under ordinary use", "has turned into a recurring problem", "is increasingly unreliable", "breaks down when conditions change"),
    },
}

CONTEXTS = {
    "train": ("after a week of use", "during routine use", "after setup", "after several normal cycles", "during a basic check"),
    "incoming": ("after a trip", "during a heavy day of use", "after changing settings", "when another person used it", "outside my usual routine"),
}

DETAILS = {
    "train": ("the result was repeatable", "the behavior stayed consistent", "I noticed it more than once", "nothing unusual happened around it"),
    "incoming": ("this was not part of my first impression", "the change appeared in a new situation", "I checked it again before writing this", "the behavior persisted across repeated attempts"),
}

PATTERNS = {
    "train": {
        "limited": ("my {alias} {component} {assessment} {profanity}", "{alias} {component} {assessment} {profanity}"),
        "informal": ("honestly the {component} on this {alias} {assessment} {profanity}", "been using this {alias} and the {component} {assessment} {profanity}"),
        "standard": ("{context}, the {component} on this {alias} {assessment}; {detail}. {profanity}", "The {alias}'s {component} {assessment} after regular use; {detail}. {profanity}"),
        "advanced": ("{context}, I found that the {component} of the {alias} {assessment}; {detail}. {profanity}", "After extended observation, the {alias}'s {component} {assessment}; {detail}. {profanity}"),
        "technical": ("Under a representative operating profile, the {component} subsystem of the {alias} {assessment}; {detail}. {profanity}", "Repeated observation indicates that the {alias} {component} {assessment}; {detail}. {profanity}"),
    },
    "incoming": {
        "limited": ("got this {alias} {component} now {assessment} {profanity}", "used {alias} today {component} {assessment} {profanity}"),
        "informal": ("quick update the {component} on my {alias} {assessment} {profanity}", "not gonna overthink it the {alias} {component} {assessment} {profanity}"),
        "standard": ("{context}, I noticed that the {alias} {component} {assessment}; {detail}. {profanity}", "A later check showed that the {component} in the {alias} {assessment}; {detail}. {profanity}"),
        "advanced": ("Viewed across several different situations, the {component} of the {alias} {assessment}; {detail}. {profanity}", "My subsequent experience changed the initial impression because the {alias} {component} {assessment}; {detail}. {profanity}"),
        "technical": ("Under conditions outside the original usage profile, the {component} subsystem of the {alias} {assessment}; {detail}. {profanity}", "Across a shifted usage regime, the observed behavior of the {alias} {component} {assessment}; {detail}. {profanity}"),
    },
}

PROFANITY = {
    "positive": ("damn this is good", "this thing is fucking impressive", "good as hell"),
    "neutral": ("the damn thing is basically average", "nothing fucking special", "plain as shit"),
    "negative": ("this shit is frustrating", "the damn thing keeps causing trouble", "this is fucking annoying", "what a piece of crap"),
}


def _validate_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("input_timestamp must be valid ISO-8601.") from error
    if parsed.tzinfo is None:
        raise ValueError("input_timestamp must include a timezone.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _text_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", text.lower())).strip()


def _alpha_code(value: int) -> str:
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(97 + remainder))
    return "".join(reversed(chars)) or "a"


def _flags(size: int, fraction: float, rng: random.Random) -> list[int]:
    flags = [1] * int(round(size * fraction))
    flags += [0] * (size - len(flags))
    rng.shuffle(flags)
    return flags


def _degrade(text: str, rng: random.Random) -> str:
    replacements = {"battery": "batery", "works": "work", "using": "usin", "nothing": "nothin", "again": "agin"}
    words = text.lower().replace("'", "").split()
    for index, word in enumerate(words):
        clean = re.sub(r"[^a-z]", "", word)
        if clean in replacements and rng.random() < 0.6:
            words[index] = replacements[clean]
    if len(words) > 6 and rng.random() < 0.4:
        words.pop(rng.randrange(1, len(words) - 1))
    return re.sub(r"[^a-z\s]", "", " ".join(words)).strip()


class SyntheticDataAgent:
    """Generate balanced train and incoming streams with controlled linguistic shift."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        config.validate()
        self.config = config

    def _specs(self, count: int, generation: int, incoming: bool) -> list[dict[str, Any]]:
        strata = list(itertools.product(SENTIMENTS, TOPICS, LINGUISTIC_LEVELS))
        per_stratum = count // len(strata)
        rng = random.Random(self.config.seed + generation * 10_007 + (97 if incoming else 0))
        specs: list[dict[str, Any]] = []
        for sentiment, topic, level in strata:
            profanity = _flags(per_stratum, self.config.profanity_fraction, rng)
            gold = _flags(per_stratum, self.config.goldtest_fraction, rng) if incoming else [0] * per_stratum
            for repetition in range(per_stratum):
                specs.append({
                    "sentiment": sentiment, "topic": topic, "linguistic_level": level,
                    "flagprofanity": profanity[repetition], "goldtest": gold[repetition],
                    "repetition": repetition,
                })
        rng.shuffle(specs)
        return specs

    def _render(self, review_id: int, generation: int, split: str, spec: dict[str, Any], variant: int) -> str:
        rng = random.Random(self.config.seed * 1_000_033 + review_id * 97 + generation * 9_973 + variant * 7_919)
        aliases, components = TOPIC_LANGUAGE[spec["topic"]][split]
        text = rng.choice(PATTERNS[split][spec["linguistic_level"]]).format(
            alias=rng.choice(aliases), component=rng.choice(components),
            assessment=rng.choice(ASSESSMENTS[split][spec["sentiment"]]),
            context=rng.choice(CONTEXTS[split]), detail=rng.choice(DETAILS[split]),
            profanity=rng.choice(PROFANITY[spec["sentiment"]]) if spec["flagprofanity"] else "",
        )
        text = re.sub(r"\s+", " ", text).strip(" ,;.")
        if spec["linguistic_level"] == "limited":
            return _degrade(text, rng)
        if spec["linguistic_level"] == "informal":
            return text.lower()
        return text[0].upper() + text[1:] + ("" if text.endswith((".", "!", "?")) else ".")

    def _generate(self, start_id: int, count: int, generation: int, split: str, timestamp: str, used: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset, spec in enumerate(self._specs(count, generation, split == "incoming")):
            review_id = start_id + offset
            for variant in range(30):
                text = self._render(review_id, generation, split, spec, variant)
                key = _text_key(text)
                if key not in used:
                    break
            else:
                text = f"{text} reference {_alpha_code(review_id)}"
                key = _text_key(text)
            used.add(key)
            common = {
                "ID": str(review_id), "text": text, "linguistic_level": spec["linguistic_level"],
                "flagprofanity": str(spec["flagprofanity"]), "input_timestamp": timestamp,
            }
            if split == "incoming":
                rows.append({**common, "expected_sentiment": spec["sentiment"], "expected_topic": spec["topic"], "goldtest": str(spec["goldtest"])})
            else:
                rows.append({**common, "sentiment": spec["sentiment"], "topic": spec["topic"], "source": "initial", "training_generation": str(generation)})
        return rows

    @staticmethod
    def _validate_train(rows: list[dict[str, str]]) -> None:
        if not rows or set(rows[0]) != set(TRAIN_FIELDS):
            raise ValueError("train.csv does not use the current schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Training IDs must be unique and increasing.")
        if any(row["linguistic_level"] not in LINGUISTIC_LEVELS or row["flagprofanity"] not in {"0", "1"} for row in rows):
            raise ValueError("Invalid training metadata.")
        if any(row["source"] not in {"initial", "goldtest"} or row["sentiment"] not in SENTIMENTS or row["topic"] not in TOPICS for row in rows):
            raise ValueError("Invalid training labels or source.")
        for row in rows:
            _validate_timestamp(row["input_timestamp"])

    @staticmethod
    def _validate_incoming(rows: list[dict[str, str]]) -> None:
        if not rows or set(rows[0]) != set(INCOMING_FIELDS):
            raise ValueError("incoming.csv does not use the current schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Incoming IDs must be unique and increasing.")
        if any(row["linguistic_level"] not in LINGUISTIC_LEVELS or row["flagprofanity"] not in {"0", "1"} or row["goldtest"] not in {"0", "1"} for row in rows):
            raise ValueError("Invalid incoming metadata.")
        if any(row["expected_sentiment"] not in SENTIMENTS or row["expected_topic"] not in TOPICS for row in rows):
            raise ValueError("Invalid incoming labels.")
        for row in rows:
            _validate_timestamp(row["input_timestamp"])

    def initialize(self, output_dir: str | Path, input_timestamp: str, *, overwrite: bool = False) -> Path:
        timestamp = _validate_timestamp(input_timestamp)
        destination = Path(output_dir)
        if any((destination / name).exists() for name in ("train.csv", "incoming.csv")) and not overwrite:
            raise FileExistsError("Input data already exists. Use overwrite=True.")
        used: set[str] = set()
        train = self._generate(1, self.config.initial_train_rows, 0, "train", timestamp, used)
        start = self.config.initial_train_rows + 1
        incoming = self._generate(start, self.config.incoming_rows, 0, "incoming", timestamp, used)
        return self._write_state(destination, train, incoming, 0, start + self.config.incoming_rows - 1)

    def advance(self, output_dir: str | Path, input_timestamp: str) -> Path:
        timestamp = _validate_timestamp(input_timestamp)
        destination = Path(output_dir)
        manifest_path = destination / "input_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Initialize input data before advancing it.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train = _read_csv(destination / "train.csv")
        incoming = _read_csv(destination / "incoming.csv")
        self._validate_train(train)
        self._validate_incoming(incoming)
        generation = int(manifest["generation"]) + 1
        promoted = [row for row in incoming if row["goldtest"] == "1"]
        train.extend({
            "ID": row["ID"], "text": row["text"], "sentiment": row["expected_sentiment"],
            "topic": row["expected_topic"], "linguistic_level": row["linguistic_level"],
            "flagprofanity": row["flagprofanity"], "source": "goldtest",
            "training_generation": str(generation), "input_timestamp": row["input_timestamp"],
        } for row in promoted)
        train.sort(key=lambda row: int(row["ID"]))
        last_id = int(manifest["last_issued_id"])
        used = {_text_key(row["text"]) for row in train}
        next_incoming = self._generate(last_id + 1, self.config.incoming_rows, generation, "incoming", timestamp, used)
        return self._write_state(destination, train, next_incoming, generation, last_id + self.config.incoming_rows, len(promoted))

    def _write_state(self, destination: Path, train: list[dict[str, Any]], incoming: list[dict[str, Any]], generation: int, last_id: int, promoted: int = 0) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        _write_csv(destination / "train.csv", train, TRAIN_FIELDS)
        _write_csv(destination / "incoming.csv", incoming, INCOMING_FIELDS)
        train_rows, incoming_rows = _read_csv(destination / "train.csv"), _read_csv(destination / "incoming.csv")
        self._validate_train(train_rows)
        self._validate_incoming(incoming_rows)
        if {row["ID"] for row in train_rows} & {row["ID"] for row in incoming_rows}:
            raise ValueError("train.csv and incoming.csv must have disjoint IDs.")
        if {_text_key(row["text"]) for row in train_rows} & {_text_key(row["text"]) for row in incoming_rows}:
            raise ValueError("train.csv and incoming.csv must have disjoint text.")
        files = (destination / "train.csv", destination / "incoming.csv")
        manifest = {
            "generated_by": self.config.agent_name, "agent_version": self.config.agent_version,
            "generation": generation, "last_issued_id": last_id,
            "promoted_from_previous_incoming": promoted, "config": asdict(self.config),
            "record_counts": {"train.csv": len(train_rows), "incoming.csv": len(incoming_rows)},
            "incoming_goldtest_count": sum(row["goldtest"] == "1" for row in incoming_rows),
            "train_source_counts": dict(sorted(Counter(row["source"] for row in train_rows).items())),
            "incoming_linguistic_level_counts": dict(sorted(Counter(row["linguistic_level"] for row in incoming_rows).items())),
            "incoming_profanity_counts": dict(sorted(Counter(row["flagprofanity"] for row in incoming_rows).items())),
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        }
        manifest_path = destination / "input_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path
