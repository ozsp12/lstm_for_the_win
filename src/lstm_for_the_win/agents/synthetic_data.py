"""Rich synthetic review generation for continual-learning experiments."""

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
LENGTH_CLASSES = ("short", "medium", "long")
STYLE_FIELDS = ("hasemoji", "hasspellingerror", "hasslang", "mixed_sentiment")
MIN_CORPUS_ROWS = 1_000

TRAIN_FIELDS = (
    "ID", "text", "sentiment", "topic", "linguistic_level", "flagprofanity",
    "hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment",
    "source", "training_generation", "input_timestamp",
)
INCOMING_FIELDS = (
    "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
    "flagprofanity", "hasemoji", "hasspellingerror", "hasslang", "length_class",
    "mixed_sentiment", "goldtest", "input_timestamp",
)


@dataclass(frozen=True)
class SyntheticDataConfig:
    agent_name: str = "synthetic-review-generator"
    agent_version: str = "4.1.0"
    language: str = "en"
    seed: int = 42
    initial_train_rows: int = 12_000
    incoming_rows: int = 1_800
    incoming_rows_jitter: int = 300
    profanity_fraction: float = 0.25
    profanity_fraction_jitter: float = 0.08
    goldtest_fraction: float = 0.20
    goldtest_fraction_jitter: float = 0.05
    emoji_fraction: float = 0.18
    emoji_fraction_jitter: float = 0.07
    spelling_error_fraction: float = 0.22
    slang_fraction: float = 0.22
    mixed_sentiment_fraction: float = 0.12
    validation_fraction: float = 0.15
    validation_fraction_jitter: float = 0.03
    vary_counts: bool = True
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
        if self.initial_train_rows < MIN_CORPUS_ROWS or self.initial_train_rows % strata:
            raise ValueError(f"initial_train_rows must be at least {MIN_CORPUS_ROWS} and a multiple of {strata}.")
        if self.incoming_rows < MIN_CORPUS_ROWS or self.incoming_rows % strata:
            raise ValueError(f"incoming_rows must be at least {MIN_CORPUS_ROWS} and a multiple of {strata}.")
        if self.incoming_rows_jitter < 0:
            raise ValueError("incoming_rows_jitter cannot be negative.")
        if self.incoming_rows - self.incoming_rows_jitter < MIN_CORPUS_ROWS:
            raise ValueError(f"incoming_rows - incoming_rows_jitter must remain at least {MIN_CORPUS_ROWS}.")
        fractions = (
            self.profanity_fraction, self.goldtest_fraction, self.emoji_fraction,
            self.spelling_error_fraction, self.slang_fraction, self.mixed_sentiment_fraction,
            self.validation_fraction,
        )
        if any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError("Fractions must be strictly between 0 and 1.")
        jitters = (
            self.profanity_fraction_jitter, self.goldtest_fraction_jitter,
            self.emoji_fraction_jitter, self.validation_fraction_jitter,
        )
        if any(value < 0.0 for value in jitters):
            raise ValueError("Fraction jitters cannot be negative.")

    def effective_generation(self, generation: int) -> dict[str, float | int]:
        """Return deterministic generation-level quantities derived from seed + generation."""
        rng = random.Random(self.seed + generation * 104_729)
        strata = len(SENTIMENTS) * len(TOPICS) * len(LINGUISTIC_LEVELS)
        incoming = self.incoming_rows
        if self.vary_counts and self.incoming_rows_jitter:
            low = max(MIN_CORPUS_ROWS, self.incoming_rows - self.incoming_rows_jitter)
            high = self.incoming_rows + self.incoming_rows_jitter
            candidates = [n for n in range(low, high + 1) if n % strata == 0]
            incoming = rng.choice(candidates) if candidates else self.incoming_rows

        def vary(center: float, jitter: float) -> float:
            if not self.vary_counts or jitter == 0:
                return center
            return min(0.95, max(0.01, center + rng.uniform(-jitter, jitter)))

        return {
            "incoming_rows": incoming,
            "profanity_fraction": vary(self.profanity_fraction, self.profanity_fraction_jitter),
            "goldtest_fraction": vary(self.goldtest_fraction, self.goldtest_fraction_jitter),
            "emoji_fraction": vary(self.emoji_fraction, self.emoji_fraction_jitter),
            "validation_fraction": vary(self.validation_fraction, self.validation_fraction_jitter),
        }


TOPIC_LANGUAGE = {
    "smartphone": {
        "train": (
            ("smartphone", "phone", "handset", "mobile phone", "device", "daily phone"),
            ("battery", "camera", "touchscreen", "charging port", "speaker", "signal reception", "face unlock", "software", "microphone"),
        ),
        "incoming": (
            ("cell phone", "mobile", "pocket device", "daily driver", "new phone", "work phone"),
            ("battery life", "rear camera", "touch panel", "USB port", "earpiece", "network reception", "fingerprint reader", "apps", "call quality"),
        ),
    },
    "television": {
        "train": (
            ("television", "TV", "smart TV", "display", "living-room TV", "screen"),
            ("screen", "remote control", "HDMI input", "sound output", "backlight", "menu system", "Wi-Fi", "streaming apps", "motion handling"),
        ),
        "incoming": (
            ("television set", "screen unit", "video panel", "living room display", "TV set", "bedroom TV"),
            ("picture quality", "controller", "video input", "built-in audio", "panel lighting", "interface", "wireless connection", "streaming menu", "sports motion"),
        ),
    },
    "refrigerator": {
        "train": (
            ("refrigerator", "fridge", "cooling unit", "kitchen refrigerator", "family fridge", "appliance"),
            ("temperature control", "door seal", "ice maker", "cooling fan", "shelves", "compressor", "freezer drawer", "interior light", "door alarm"),
        ),
        "incoming": (
            ("cold-storage unit", "kitchen fridge", "food cooler", "refrigeration unit", "main fridge", "new refrigerator"),
            ("thermostat", "gasket", "ice system", "compressor fan", "shelf layout", "cooling motor", "freezer compartment", "inside lighting", "door warning"),
        ),
    },
    "washing_machine": {
        "train": (
            ("washing machine", "washer", "laundry machine", "front loader", "washing unit", "laundry appliance"),
            ("spin cycle", "water inlet", "detergent drawer", "drain pump", "drum", "control panel", "door lock", "rinse cycle", "noise level"),
        ),
        "incoming": (
            ("clothes washer", "wash unit", "front-loading washer", "laundry unit", "new washer", "machine"),
            ("spin program", "fill valve", "soap tray", "drainage motor", "wash drum", "cycle controls", "door latch", "rinse program", "vibration"),
        ),
    },
}

ASSESSMENTS = {
    "train": {
        "positive": (
            "has been reliable", "works better than I expected", "still feels solid", "does its job without fuss",
            "has held up well", "performs consistently", "has been easy to live with", "keeps doing what I need",
        ),
        "neutral": (
            "works about as expected", "is pretty ordinary", "does the basic job", "is fine but forgettable",
            "has no major strengths or weaknesses", "feels average", "is usable without standing out", "does what the specification says",
        ),
        "negative": (
            "has become unreliable", "keeps causing problems", "works worse than I expected", "has been frustrating to use",
            "fails too often", "is inconsistent", "has not held up", "makes routine use harder than it should be",
        ),
    },
    "incoming": {
        "positive": (
            "has stayed dependable", "works well even outside my usual routine", "has been a pleasant surprise", "still performs consistently",
            "holds up better than expected", "has not given me trouble", "feels dependable in daily use", "continues to work without drama",
        ),
        "neutral": (
            "is neither impressive nor bad", "remains basically average", "works normally", "has not changed my opinion much",
            "is acceptable but unremarkable", "does the job", "feels ordinary after more use", "is fine for what it is",
        ),
        "negative": (
            "has turned into a recurring problem", "fails in normal use", "has become increasingly unreliable", "keeps acting up",
            "breaks down when conditions change", "has been a disappointment", "is difficult to trust", "keeps getting in the way",
        ),
    },
}

CONTEXTS = {
    "train": (
        "after a week of everyday use", "during normal weekday use", "after the initial setup", "after several routine cycles",
        "while using it at home", "after comparing it with my previous unit", "after a month of ownership", "during a typical weekend",
    ),
    "incoming": (
        "after taking it on a trip", "during a heavier-than-usual day", "after changing a few settings", "when someone else in the house used it",
        "outside my normal routine", "after a few weeks of mixed use", "after using it for work and at home", "after coming back to it after a few days",
    ),
}

DETAILS = {
    "train": (
        "I got the same result more than once", "the behavior was consistent from day to day", "I checked it again before making up my mind",
        "nothing else unusual was happening at the time", "the same pattern showed up in a second test", "it behaved similarly under a few different conditions",
        "I noticed the pattern gradually rather than all at once", "the result matched what I saw earlier in the week",
    ),
    "incoming": (
        "this was different from my first impression", "I noticed the change only in the new situation", "I repeated the check before writing this",
        "the behavior persisted across several attempts", "a second check gave me the same result", "the difference became clearer after a few more days",
        "someone else in the house noticed it too", "I tried the same thing again later and got a similar result",
    ),
}

FOLLOWUPS = {
    "positive": (
        "I would buy it again", "overall I am happy with it", "I plan to keep using it", "it has earned my confidence",
        "for my use case, this is a keeper", "I have stopped worrying about this part of the product",
    ),
    "neutral": (
        "I can live with it", "I do not feel strongly either way", "there is not much else to say", "it is adequate for now",
        "I would neither recommend it nor warn people away", "my opinion is still basically in the middle",
    ),
    "negative": (
        "I would not rely on it", "I am considering a replacement", "this needs to improve", "I would hesitate to buy it again",
        "I am already looking at alternatives", "this is the part that makes me regret the purchase",
    ),
}

SLANG_OPENERS = ("ngl,", "tbh,", "imo,", "lowkey,", "fr,", "not gonna lie,", "honestly tho,", "real talk,")
SLANG_TAILS = ("lol", "idk", "for real", "kinda wild", "not great tbh", "pretty solid ngl")
EMOJIS = ("🙂", "😅", "🤷", "😐", "🙃", "🔥", "💀", "🤔", "👍", "👎", "😂", "😬")

PROFANITY_CLAUSES = {
    "positive": (
        "Damn, I did not expect this part to be this solid.",
        "No bullshit, this has been one of the better parts of the product.",
        "I expected some crap here, but it has actually held up.",
        "The damn thing surprised me in a good way.",
    ),
    "neutral": (
        "The damn thing is basically fine.",
        "It is not amazing, but it is not bullshit either.",
        "There is no major crap to complain about here.",
        "Honestly, the damn thing is just average.",
    ),
    "negative": (
        "This is damn annoying.",
        "I am tired of this crap.",
        "The damn thing keeps getting in the way.",
        "This shit should not be happening in normal use.",
    ),
}

TECHNICAL_OPENERS = (
    "Under a representative operating profile,", "Across repeated observations,", "Under ordinary operating conditions,",
    "After repeated use under comparable conditions,", "From a practical reliability standpoint,",
)

COMMON_TYPOS = {
    "battery": "batery", "really": "realy", "received": "recieved", "separate": "seperate", "because": "becuase",
    "different": "diferent", "reliable": "relaiable", "quality": "quailty", "using": "useing", "problem": "probelm",
    "frustrating": "frustating", "performance": "perfomance", "connection": "conection", "temperature": "temprature",
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
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
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


def _length_class(text: str) -> str:
    words = len(text.split())
    return "short" if words < 14 else "medium" if words < 34 else "long"


def _has_emoji(text: str) -> bool:
    return any(ord(char) >= 0x1F000 for char in text)


def _inject_typo(text: str, rng: random.Random) -> str:
    words = text.split()
    lowered = [re.sub(r"[^A-Za-z]", "", word).lower() for word in words]
    common = [i for i, word in enumerate(lowered) if word in COMMON_TYPOS]
    if common and rng.random() < 0.65:
        index = rng.choice(common)
        original = words[index]
        clean = lowered[index]
        replacement = COMMON_TYPOS[clean]
        suffix = "".join(char for char in original if not char.isalpha())
        words[index] = replacement + suffix
        return " ".join(words)

    candidates = [i for i, word in enumerate(words) if len(re.sub(r"[^A-Za-z]", "", word)) >= 5]
    if not candidates:
        return text + " agin"
    index = rng.choice(candidates)
    word = words[index]
    letters = list(word)
    positions = [i for i, char in enumerate(letters[:-1]) if char.isalpha() and letters[i + 1].isalpha()]
    if positions:
        pos = rng.choice(positions)
        letters[pos], letters[pos + 1] = letters[pos + 1], letters[pos]
    else:
        letters.pop(max(1, len(letters) // 2))
    words[index] = "".join(letters)
    return " ".join(words)


def _legacy_defaults(row: dict[str, str]) -> None:
    text = row.get("text", "")
    row.setdefault("hasemoji", "1" if _has_emoji(text) else "0")
    row.setdefault("hasspellingerror", "0")
    row.setdefault("hasslang", "0")
    row.setdefault("mixed_sentiment", "0")
    row.setdefault("length_class", _length_class(text))


class SyntheticDataAgent:
    """Generate reproducible, compositionally varied train and incoming review streams."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        config.validate()
        self.config = config

    def _specs(self, count: int, generation: int, incoming: bool) -> list[dict[str, Any]]:
        strata = list(itertools.product(SENTIMENTS, TOPICS, LINGUISTIC_LEVELS))
        per_stratum = count // len(strata)
        effective = self.config.effective_generation(generation)
        rng = random.Random(self.config.seed + generation * 10_007 + (97 if incoming else 0))
        specs: list[dict[str, Any]] = []
        for sentiment, topic, level in strata:
            profanity = _flags(per_stratum, float(effective["profanity_fraction"]), rng)
            emoji = _flags(per_stratum, float(effective["emoji_fraction"]), rng)
            spelling = _flags(per_stratum, self.config.spelling_error_fraction, rng)
            slang = _flags(per_stratum, self.config.slang_fraction, rng)
            mixed = _flags(per_stratum, self.config.mixed_sentiment_fraction, rng)
            gold = _flags(per_stratum, float(effective["goldtest_fraction"]), rng) if incoming else [0] * per_stratum
            for repetition in range(per_stratum):
                specs.append({
                    "sentiment": sentiment, "topic": topic, "linguistic_level": level,
                    "flagprofanity": profanity[repetition], "hasemoji": emoji[repetition],
                    "hasspellingerror": spelling[repetition], "hasslang": slang[repetition],
                    "mixed_sentiment": mixed[repetition], "goldtest": gold[repetition],
                    "repetition": repetition,
                })
        rng.shuffle(specs)
        return specs

    @staticmethod
    def _base_review(
        alias: str,
        component: str,
        assessment: str,
        context: str,
        detail: str,
        rng: random.Random,
    ) -> str:
        patterns = (
            f"{context.capitalize()}, I noticed that the {component} on this {alias} {assessment}.",
            f"I have been using this {alias} {context}, and the {component} {assessment}.",
            f"The {component} is what stood out {context}; it {assessment}.",
            f"{context.capitalize()}, the {component} {assessment}.",
            f"My main impression of this {alias} comes from the {component}: it {assessment} {context}.",
            f"I did not pay much attention to the {component} at first, but {context} it {assessment}.",
        )
        text = rng.choice(patterns)
        if rng.random() < 0.76:
            text += f" {detail}."
        return text

    def _render(self, review_id: int, generation: int, split: str, spec: dict[str, Any], variant: int) -> str:
        rng = random.Random(self.config.seed * 1_000_033 + review_id * 97 + generation * 9_973 + variant * 7_919)
        aliases, components = TOPIC_LANGUAGE[spec["topic"]][split]
        alias = rng.choice(aliases)
        component = rng.choice(components)
        assessment = rng.choice(ASSESSMENTS[split][spec["sentiment"]])
        context = rng.choice(CONTEXTS[split])
        detail = rng.choice(DETAILS[split])
        sentence = self._base_review(alias, component, assessment, context, detail, rng)

        if rng.random() < 0.36:
            sentence += f" {rng.choice(FOLLOWUPS[spec['sentiment']])}."

        if spec["mixed_sentiment"]:
            other = rng.choice([value for value in SENTIMENTS if value != spec["sentiment"]])
            secondary = rng.choice(ASSESSMENTS[split][other])
            other_component = rng.choice([value for value in components if value != component] or components)
            contrasts = (
                f"That said, the {other_component} {secondary}, so the experience is not completely one-sided.",
                f"On the other hand, the {other_component} {secondary}, which makes the overall picture more mixed.",
                f"The {other_component} tells a different story because it {secondary}.",
            )
            sentence += f" {rng.choice(contrasts)}"

        if spec["flagprofanity"]:
            sentence += f" {rng.choice(PROFANITY_CLAUSES[spec['sentiment']])}"

        if spec["hasslang"]:
            if rng.random() < 0.7:
                sentence = f"{rng.choice(SLANG_OPENERS)} {sentence[0].lower() + sentence[1:]}"
            else:
                sentence += f" {rng.choice(SLANG_TAILS)}."

        if spec["hasspellingerror"]:
            sentence = _inject_typo(sentence, rng)
            if len(sentence.split()) > 38 and rng.random() < 0.35:
                sentence = _inject_typo(sentence, rng)

        level = spec["linguistic_level"]
        if level == "limited":
            sentence = sentence.lower().replace("'", "")
            words = sentence.split()
            if len(words) > 10 and rng.random() < 0.65:
                words.pop(rng.randrange(2, len(words) - 2))
            if len(words) > 18 and rng.random() < 0.35:
                words.pop(rng.randrange(2, len(words) - 2))
            sentence = " ".join(words)
        elif level == "informal":
            sentence = sentence.lower().replace("do not", "don't").replace("going to", "gonna").replace("I have", "I've")
        elif level == "advanced":
            sentence = sentence.replace("overall", "on balance").replace("normal use", "routine use")
        elif level == "technical":
            sentence = f"{rng.choice(TECHNICAL_OPENERS)} {sentence[0].lower() + sentence[1:]}"

        if spec["hasemoji"]:
            placements = (
                f"{sentence} {rng.choice(EMOJIS)}",
                f"{sentence} {rng.choice(EMOJIS)}",
                f"{sentence} Honestly {rng.choice(EMOJIS)}",
            )
            sentence = rng.choice(placements)

        return re.sub(r"\s+", " ", sentence).strip()

    def _generate(self, start_id: int, count: int, generation: int, split: str, timestamp: str, used: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset, spec in enumerate(self._specs(count, generation, split == "incoming")):
            review_id = start_id + offset
            for variant in range(80):
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
                "flagprofanity": str(spec["flagprofanity"]), "hasemoji": str(spec["hasemoji"]),
                "hasspellingerror": str(spec["hasspellingerror"]), "hasslang": str(spec["hasslang"]),
                "length_class": _length_class(text), "mixed_sentiment": str(spec["mixed_sentiment"]),
                "input_timestamp": timestamp,
            }
            if split == "incoming":
                rows.append({**common, "expected_sentiment": spec["sentiment"], "expected_topic": spec["topic"], "goldtest": str(spec["goldtest"])})
            else:
                rows.append({**common, "sentiment": spec["sentiment"], "topic": spec["topic"], "source": "initial", "training_generation": str(generation)})
        return rows

    @staticmethod
    def _upgrade_train(rows: list[dict[str, str]]) -> None:
        for row in rows:
            _legacy_defaults(row)

    @staticmethod
    def _upgrade_incoming(rows: list[dict[str, str]]) -> None:
        for row in rows:
            _legacy_defaults(row)

    @staticmethod
    def _validate_train(rows: list[dict[str, str]]) -> None:
        SyntheticDataAgent._upgrade_train(rows)
        if len(rows) < MIN_CORPUS_ROWS:
            raise ValueError(f"train.csv must contain at least {MIN_CORPUS_ROWS} reviews.")
        if not set(TRAIN_FIELDS).issubset(rows[0]):
            raise ValueError("train.csv does not use the current schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Training IDs must be unique and increasing.")
        if any(row["linguistic_level"] not in LINGUISTIC_LEVELS or row["flagprofanity"] not in {"0", "1"} for row in rows):
            raise ValueError("Invalid training metadata.")
        if any(row[field] not in {"0", "1"} for row in rows for field in STYLE_FIELDS):
            raise ValueError("Invalid training style metadata.")
        if any(row["length_class"] not in LENGTH_CLASSES for row in rows):
            raise ValueError("Invalid training length_class.")
        if any(row["source"] not in {"initial", "goldtest"} or row["sentiment"] not in SENTIMENTS or row["topic"] not in TOPICS for row in rows):
            raise ValueError("Invalid training labels or source.")
        for row in rows:
            _validate_timestamp(row["input_timestamp"])

    @staticmethod
    def _validate_incoming(rows: list[dict[str, str]]) -> None:
        SyntheticDataAgent._upgrade_incoming(rows)
        if len(rows) < MIN_CORPUS_ROWS:
            raise ValueError(f"incoming.csv must contain at least {MIN_CORPUS_ROWS} reviews.")
        if not set(INCOMING_FIELDS).issubset(rows[0]):
            raise ValueError("incoming.csv does not use the current schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Incoming IDs must be unique and increasing.")
        if any(row["linguistic_level"] not in LINGUISTIC_LEVELS or row["flagprofanity"] not in {"0", "1"} or row["goldtest"] not in {"0", "1"} for row in rows):
            raise ValueError("Invalid incoming metadata.")
        if any(row[field] not in {"0", "1"} for row in rows for field in STYLE_FIELDS):
            raise ValueError("Invalid incoming style metadata.")
        if any(row["length_class"] not in LENGTH_CLASSES for row in rows):
            raise ValueError("Invalid incoming length_class.")
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
        count = int(self.config.effective_generation(0)["incoming_rows"])
        incoming = self._generate(start, count, 0, "incoming", timestamp, used)
        return self._write_state(destination, train, incoming, 0, start + count - 1)

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
            "flagprofanity": row["flagprofanity"], "hasemoji": row["hasemoji"],
            "hasspellingerror": row["hasspellingerror"], "hasslang": row["hasslang"],
            "length_class": row["length_class"], "mixed_sentiment": row["mixed_sentiment"],
            "source": "goldtest", "training_generation": str(generation),
            "input_timestamp": row["input_timestamp"],
        } for row in promoted)
        train.sort(key=lambda row: int(row["ID"]))
        last_id = int(manifest["last_issued_id"])
        used = {_text_key(row["text"]) for row in train}
        count = int(self.config.effective_generation(generation)["incoming_rows"])
        next_incoming = self._generate(last_id + 1, count, generation, "incoming", timestamp, used)
        return self._write_state(destination, train, next_incoming, generation, last_id + count, len(promoted))

    def _write_state(self, destination: Path, train: list[dict[str, Any]], incoming: list[dict[str, Any]], generation: int, last_id: int, promoted: int = 0) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        self._upgrade_train(train)
        self._upgrade_incoming(incoming)
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
        effective = self.config.effective_generation(generation)
        manifest = {
            "generated_by": self.config.agent_name, "agent_version": self.config.agent_version,
            "generation": generation, "last_issued_id": last_id,
            "promoted_from_previous_incoming": promoted, "config": asdict(self.config),
            "effective_generation": effective,
            "record_counts": {"train.csv": len(train_rows), "incoming.csv": len(incoming_rows)},
            "incoming_goldtest_count": sum(row["goldtest"] == "1" for row in incoming_rows),
            "train_source_counts": dict(sorted(Counter(row["source"] for row in train_rows).items())),
            "incoming_linguistic_level_counts": dict(sorted(Counter(row["linguistic_level"] for row in incoming_rows).items())),
            "incoming_profanity_counts": dict(sorted(Counter(row["flagprofanity"] for row in incoming_rows).items())),
            "incoming_emoji_counts": dict(sorted(Counter(row["hasemoji"] for row in incoming_rows).items())),
            "incoming_spelling_error_counts": dict(sorted(Counter(row["hasspellingerror"] for row in incoming_rows).items())),
            "incoming_slang_counts": dict(sorted(Counter(row["hasslang"] for row in incoming_rows).items())),
            "incoming_length_counts": dict(sorted(Counter(row["length_class"] for row in incoming_rows).items())),
            "incoming_mixed_sentiment_counts": dict(sorted(Counter(row["mixed_sentiment"] for row in incoming_rows).items())),
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        }
        manifest_path = destination / "input_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path
