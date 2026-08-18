"""Improved synthetic generator with explicit template provenance and less rigid strata."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .synthetic_data import (
    ASSESSMENTS,
    CONTEXTS,
    DETAILS,
    EMOJIS,
    FOLLOWUPS,
    LENGTH_CLASSES,
    LINGUISTIC_LEVELS,
    PROFANITY_CLAUSES,
    SENTIMENTS,
    SLANG_OPENERS,
    SLANG_TAILS,
    STYLE_FIELDS,
    TECHNICAL_OPENERS,
    TOPICS,
    TOPIC_LANGUAGE,
    SyntheticDataAgent as _BaseSyntheticDataAgent,
    SyntheticDataConfig,
    _alpha_code,
    _flags,
    _has_emoji,
    _length_class,
    _read_csv,
    _text_key,
    _validate_timestamp,
    _write_csv,
)

TEMPLATE_FAMILIES = (
    "noticed",
    "using",
    "stood_out",
    "context_component",
    "main_impression",
    "attention",
)

TRAIN_FIELDS = (
    "ID", "text", "sentiment", "topic", "linguistic_level", "flagprofanity",
    "hasemoji", "hasspellingerror", "hasslang", "length_class", "mixed_sentiment",
    "template_family", "source", "training_generation", "input_timestamp",
)
INCOMING_FIELDS = (
    "ID", "text", "expected_sentiment", "expected_topic", "linguistic_level",
    "flagprofanity", "hasemoji", "hasspellingerror", "hasslang", "length_class",
    "mixed_sentiment", "template_family", "goldtest", "input_timestamp",
)

RICH_TYPOS = {
    "battery": "batery", "because": "becuase", "camera": "camra", "charger": "chargar",
    "connection": "conection", "control": "contorl", "different": "diferent", "display": "dispaly",
    "excellent": "excelent", "frustrating": "frustating", "interface": "interfase", "microphone": "microfone",
    "performance": "perfomance", "quality": "quailty", "received": "recieved", "refrigerator": "refridgerator",
    "reliable": "relaiable", "separate": "seperate", "software": "sofware", "speaker": "speeker",
    "temperature": "temprature", "television": "telivision", "touchscreen": "touchsreen", "using": "useing",
    "washing": "waching", "wireless": "wirless", "problem": "probelm", "comfortable": "comfotable",
    "recommended": "recomended", "experience": "experiance", "ordinary": "ordnary", "consistent": "consistant",
}

ADVANCED_CLAUSES = (
    "The pattern remained consistent enough to influence my overall assessment",
    "That repeated behavior matters more to me than the first impression",
    "The result is consistent with what I observed under comparable conditions",
    "This makes the broader ownership experience easier to evaluate",
    "The difference became clearer once I compared several ordinary use cases",
)


def _inject_rich_typo(text: str, rng: random.Random) -> str:
    words = text.split()
    normalized = [re.sub(r"[^A-Za-z]", "", word).lower() for word in words]
    candidates = [index for index, word in enumerate(normalized) if word in RICH_TYPOS]
    if candidates and rng.random() < 0.78:
        index = rng.choice(candidates)
        clean = normalized[index]
        suffix = "".join(char for char in words[index] if not char.isalpha())
        words[index] = RICH_TYPOS[clean] + suffix
        return " ".join(words)
    long_words = [index for index, word in enumerate(normalized) if len(word) >= 6]
    if not long_words:
        return text + " agin"
    index = rng.choice(long_words)
    original = words[index]
    letters = list(original)
    alphabetic = [i for i in range(len(letters) - 1) if letters[i].isalpha() and letters[i + 1].isalpha()]
    if alphabetic:
        pos = rng.choice(alphabetic)
        letters[pos], letters[pos + 1] = letters[pos + 1], letters[pos]
    words[index] = "".join(letters)
    return " ".join(words)


class SyntheticDataAgent(_BaseSyntheticDataAgent):
    """Generate deterministic reviews while preserving explicit generator provenance."""

    @staticmethod
    def _allocate_strata(count: int, size: int, rng: random.Random) -> list[int]:
        weights = [rng.uniform(0.82, 1.18) for _ in range(size)]
        scale = count / sum(weights)
        raw = [weight * scale for weight in weights]
        allocation = [max(1, int(value)) for value in raw]
        difference = count - sum(allocation)
        fractions = sorted(range(size), key=lambda i: raw[i] - int(raw[i]), reverse=difference > 0)
        cursor = 0
        while difference != 0:
            index = fractions[cursor % size]
            if difference > 0:
                allocation[index] += 1
                difference -= 1
            elif allocation[index] > 1:
                allocation[index] -= 1
                difference += 1
            cursor += 1
        return allocation

    def _specs(self, count: int, generation: int, incoming: bool) -> list[dict[str, Any]]:
        strata = list(itertools.product(SENTIMENTS, TOPICS, LINGUISTIC_LEVELS))
        effective = self.config.effective_generation(generation)
        rng = random.Random(self.config.seed + generation * 10_007 + (97 if incoming else 0))
        counts = self._allocate_strata(count, len(strata), rng)
        specs: list[dict[str, Any]] = []
        for (sentiment, topic, level), stratum_count in zip(strata, counts, strict=True):
            profanity = _flags(stratum_count, float(effective["profanity_fraction"]), rng)
            emoji = _flags(stratum_count, float(effective["emoji_fraction"]), rng)
            spelling = _flags(stratum_count, self.config.spelling_error_fraction, rng)
            slang = _flags(stratum_count, self.config.slang_fraction, rng)
            mixed = _flags(stratum_count, self.config.mixed_sentiment_fraction, rng)
            gold = _flags(stratum_count, float(effective["goldtest_fraction"]), rng) if incoming else [0] * stratum_count
            for repetition in range(stratum_count):
                specs.append({
                    "sentiment": sentiment,
                    "topic": topic,
                    "linguistic_level": level,
                    "flagprofanity": profanity[repetition],
                    "hasemoji": emoji[repetition],
                    "hasspellingerror": spelling[repetition],
                    "hasslang": slang[repetition],
                    "mixed_sentiment": mixed[repetition],
                    "goldtest": gold[repetition],
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
    ) -> tuple[str, str]:
        patterns = (
            ("noticed", f"{context.capitalize()}, I noticed that the {component} on this {alias} {assessment}."),
            ("using", f"I have been using this {alias} {context}, and the {component} {assessment}."),
            ("stood_out", f"The {component} is what stood out {context}; it {assessment}."),
            ("context_component", f"{context.capitalize()}, the {component} {assessment}."),
            ("main_impression", f"My main impression of this {alias} comes from the {component}: it {assessment} {context}."),
            ("attention", f"I did not pay much attention to the {component} at first, but {context} it {assessment}."),
        )
        family, text = rng.choice(patterns)
        if rng.random() < 0.76:
            text += f" {detail}."
        return text, family

    def _render(
        self,
        review_id: int,
        generation: int,
        split: str,
        spec: dict[str, Any],
        variant: int,
    ) -> tuple[str, str]:
        rng = random.Random(self.config.seed * 1_000_033 + review_id * 97 + generation * 9_973 + variant * 7_919)
        aliases, components = TOPIC_LANGUAGE[spec["topic"]][split]
        alias = rng.choice(aliases)
        component = rng.choice(components)
        assessment = rng.choice(ASSESSMENTS[split][spec["sentiment"]])
        context = rng.choice(CONTEXTS[split])
        detail = rng.choice(DETAILS[split])
        sentence, family = self._base_review(alias, component, assessment, context, detail, rng)

        if rng.random() < 0.36:
            sentence += f" {rng.choice(FOLLOWUPS[spec['sentiment']])}."

        if spec["mixed_sentiment"]:
            other = rng.choice([value for value in SENTIMENTS if value != spec["sentiment"]])
            secondary = rng.choice(ASSESSMENTS[split][other])
            other_component = rng.choice([value for value in components if value != component] or components)
            contrast = rng.choice((
                f"That said, the {other_component} {secondary}, so the experience is not completely one-sided.",
                f"On the other hand, the {other_component} {secondary}, which makes the overall picture more mixed.",
                f"The {other_component} tells a different story because it {secondary}.",
                f"A separate point is the {other_component}, which {secondary}.",
                f"By contrast, the {other_component} {secondary}.",
            ))
            placement = rng.choice(("before", "middle", "after"))
            if placement == "before":
                sentence = f"{contrast} {sentence}"
            elif placement == "middle" and ". " in sentence:
                first, rest = sentence.split(". ", 1)
                sentence = f"{first}. {contrast} {rest}"
            else:
                sentence += f" {contrast}"

        if spec["flagprofanity"]:
            sentence += f" {rng.choice(PROFANITY_CLAUSES[spec['sentiment']])}"

        if spec["hasslang"]:
            if rng.random() < 0.7:
                sentence = f"{rng.choice(SLANG_OPENERS)} {sentence[0].lower() + sentence[1:]}"
            else:
                sentence += f" {rng.choice(SLANG_TAILS)}."

        if spec["hasspellingerror"]:
            sentence = _inject_rich_typo(sentence, rng)
            if len(sentence.split()) > 38 and rng.random() < 0.35:
                sentence = _inject_rich_typo(sentence, rng)

        level = spec["linguistic_level"]
        if level == "limited":
            sentence = sentence.lower().replace("'", "")
            words = sentence.split()
            removals = 1 + int(len(words) > 22 and rng.random() < 0.55)
            for _ in range(removals):
                if len(words) > 10:
                    words.pop(rng.randrange(2, len(words) - 2))
            sentence = " ".join(words)
        elif level == "informal":
            sentence = sentence.lower().replace("do not", "don't").replace("going to", "gonna").replace("I have", "I've")
        elif level == "advanced":
            sentence = sentence.replace("I noticed", "I observed").replace("works", "performs")
            sentence += f" {rng.choice(ADVANCED_CLAUSES)}."
        elif level == "technical":
            sentence = f"{rng.choice(TECHNICAL_OPENERS)} {sentence[0].lower() + sentence[1:]}"

        if spec["hasemoji"]:
            sentence = rng.choice((
                f"{sentence} {rng.choice(EMOJIS)}",
                f"{sentence} {rng.choice(EMOJIS)}",
                f"{sentence} Honestly {rng.choice(EMOJIS)}",
            ))

        return re.sub(r"\s+", " ", sentence).strip(), family

    def _generate(
        self,
        start_id: int,
        count: int,
        generation: int,
        split: str,
        timestamp: str,
        used: set[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset, spec in enumerate(self._specs(count, generation, split == "incoming")):
            review_id = start_id + offset
            for variant in range(80):
                text, family = self._render(review_id, generation, split, spec, variant)
                key = _text_key(text)
                if key not in used:
                    break
            else:
                text = f"{text} reference {_alpha_code(review_id)}"
                key = _text_key(text)
            used.add(key)
            common = {
                "ID": str(review_id),
                "text": text,
                "linguistic_level": spec["linguistic_level"],
                "flagprofanity": str(spec["flagprofanity"]),
                "hasemoji": str(spec["hasemoji"]),
                "hasspellingerror": str(spec["hasspellingerror"]),
                "hasslang": str(spec["hasslang"]),
                "length_class": _length_class(text),
                "mixed_sentiment": str(spec["mixed_sentiment"]),
                "template_family": family,
                "input_timestamp": timestamp,
            }
            if split == "incoming":
                rows.append({
                    **common,
                    "expected_sentiment": spec["sentiment"],
                    "expected_topic": spec["topic"],
                    "goldtest": str(spec["goldtest"]),
                })
            else:
                rows.append({
                    **common,
                    "sentiment": spec["sentiment"],
                    "topic": spec["topic"],
                    "source": "initial",
                    "training_generation": str(generation),
                })
        return rows

    @staticmethod
    def _upgrade_train(rows: list[dict[str, str]]) -> None:
        for row in rows:
            text = row.get("text", "")
            row.setdefault("hasemoji", "1" if _has_emoji(text) else "0")
            row.setdefault("hasspellingerror", "0")
            row.setdefault("hasslang", "0")
            row.setdefault("mixed_sentiment", "0")
            row.setdefault("length_class", _length_class(text))

    @staticmethod
    def _upgrade_incoming(rows: list[dict[str, str]]) -> None:
        SyntheticDataAgent._upgrade_train(rows)

    @staticmethod
    def _validate_train(rows: list[dict[str, str]]) -> None:
        if len(rows) < 1_000 or not set(TRAIN_FIELDS).issubset(rows[0]):
            raise ValueError("train.csv does not use the current generated schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Training IDs must be unique and increasing.")
        if any(row["template_family"] not in TEMPLATE_FAMILIES for row in rows):
            raise ValueError("Training rows must carry a generated template_family.")
        if any(row["linguistic_level"] not in LINGUISTIC_LEVELS for row in rows):
            raise ValueError("Invalid linguistic level.")
        if any(row[field] not in {"0", "1"} for row in rows for field in STYLE_FIELDS):
            raise ValueError("Invalid training style metadata.")
        if any(row["length_class"] not in LENGTH_CLASSES for row in rows):
            raise ValueError("Invalid training length_class.")
        for row in rows:
            _validate_timestamp(row["input_timestamp"])

    @staticmethod
    def _validate_incoming(rows: list[dict[str, str]]) -> None:
        if len(rows) < 1_000 or not set(INCOMING_FIELDS).issubset(rows[0]):
            raise ValueError("incoming.csv does not use the current generated schema.")
        ids = [int(row["ID"]) for row in rows]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Incoming IDs must be unique and increasing.")
        if any(row["template_family"] not in TEMPLATE_FAMILIES for row in rows):
            raise ValueError("Incoming rows must carry a generated template_family.")
        if any(row["linguistic_level"] not in LINGUISTIC_LEVELS or row["goldtest"] not in {"0", "1"} for row in rows):
            raise ValueError("Invalid incoming metadata.")
        if any(row[field] not in {"0", "1"} for row in rows for field in STYLE_FIELDS):
            raise ValueError("Invalid incoming style metadata.")
        if any(row["length_class"] not in LENGTH_CLASSES for row in rows):
            raise ValueError("Invalid incoming length_class.")
        for row in rows:
            _validate_timestamp(row["input_timestamp"])

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
            "ID": row["ID"],
            "text": row["text"],
            "sentiment": row["expected_sentiment"],
            "topic": row["expected_topic"],
            "linguistic_level": row["linguistic_level"],
            "flagprofanity": row["flagprofanity"],
            "hasemoji": row["hasemoji"],
            "hasspellingerror": row["hasspellingerror"],
            "hasslang": row["hasslang"],
            "length_class": row["length_class"],
            "mixed_sentiment": row["mixed_sentiment"],
            "template_family": row["template_family"],
            "source": "goldtest",
            "training_generation": str(generation),
            "input_timestamp": row["input_timestamp"],
        } for row in promoted)
        train.sort(key=lambda row: int(row["ID"]))
        last_id = int(manifest["last_issued_id"])
        used = {_text_key(row["text"]) for row in train}
        count = int(self.config.effective_generation(generation)["incoming_rows"])
        next_incoming = self._generate(last_id + 1, count, generation, "incoming", timestamp, used)
        return self._write_state(destination, train, next_incoming, generation, last_id + count, len(promoted))

    def _write_state(
        self,
        destination: Path,
        train: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
        generation: int,
        last_id: int,
        promoted: int = 0,
    ) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        _write_csv(destination / "train.csv", train, TRAIN_FIELDS)
        _write_csv(destination / "incoming.csv", incoming, INCOMING_FIELDS)
        train_rows = _read_csv(destination / "train.csv")
        incoming_rows = _read_csv(destination / "incoming.csv")
        self._validate_train(train_rows)
        self._validate_incoming(incoming_rows)
        if {row["ID"] for row in train_rows} & {row["ID"] for row in incoming_rows}:
            raise ValueError("train.csv and incoming.csv must have disjoint IDs.")
        if {_text_key(row["text"]) for row in train_rows} & {_text_key(row["text"]) for row in incoming_rows}:
            raise ValueError("train.csv and incoming.csv must have disjoint text.")
        files = (destination / "train.csv", destination / "incoming.csv")
        manifest = {
            "generated_by": self.config.agent_name,
            "agent_version": self.config.agent_version,
            "generation": generation,
            "last_issued_id": last_id,
            "promoted_from_previous_incoming": promoted,
            "config": asdict(self.config),
            "effective_generation": self.config.effective_generation(generation),
            "record_counts": {"train.csv": len(train_rows), "incoming.csv": len(incoming_rows)},
            "incoming_goldtest_count": sum(row["goldtest"] == "1" for row in incoming_rows),
            "train_source_counts": dict(sorted(Counter(row["source"] for row in train_rows).items())),
            "incoming_linguistic_level_counts": dict(sorted(Counter(row["linguistic_level"] for row in incoming_rows).items())),
            "incoming_template_family_counts": dict(sorted(Counter(row["template_family"] for row in incoming_rows).items())),
            "incoming_profanity_counts": dict(sorted(Counter(row["flagprofanity"] for row in incoming_rows).items())),
            "incoming_emoji_counts": dict(sorted(Counter(row["hasemoji"] for row in incoming_rows).items())),
            "incoming_spelling_error_counts": dict(sorted(Counter(row["hasspellingerror"] for row in incoming_rows).items())),
            "incoming_slang_counts": dict(sorted(Counter(row["hasslang"] for row in incoming_rows).items())),
            "incoming_length_counts": dict(sorted(Counter(row["length_class"] for row in incoming_rows).items())),
            "incoming_mixed_sentiment_counts": dict(sorted(Counter(row["mixed_sentiment"] for row in incoming_rows).items())),
            "template_family_metadata": {
                "materialized": True,
                "origin": "generated_at_render_time",
                "families": list(TEMPLATE_FAMILIES),
            },
            "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        }
        manifest_path = destination / "input_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path


__all__ = ["SyntheticDataAgent", "SyntheticDataConfig", "TEMPLATE_FAMILIES"]
