"""Leitura, limpeza e divisão dos dados de texto."""

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
    """Normaliza um texto sem depender de recursos externos."""

    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " ", normalized)
    normalized = re.sub(r"[@#]\w+", " ", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def load_dataset(path: str | Path) -> list[Record]:
    """Carrega um CSV com as colunas obrigatórias ``text`` e ``topic``."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    records: list[Record] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"text", "topic"}.issubset(reader.fieldnames):
            raise ValueError("O dataset deve conter as colunas 'text' e 'topic'.")

        for row_number, row in enumerate(reader, start=2):
            text = clean_text(row["text"])
            topic = row["topic"].strip()
            if not text or not topic:
                raise ValueError(f"Linha {row_number} possui texto ou tópico vazio.")
            records.append((text, topic))

    validate_records(records)
    return records


def validate_records(records: Iterable[Record]) -> None:
    """Verifica volume e balanceamento mínimo para treino e teste."""

    counts: dict[str, int] = defaultdict(int)
    total = 0
    for _, topic in records:
        counts[topic] += 1
        total += 1

    if total == 0:
        raise ValueError("O dataset está vazio.")
    if len(counts) < 2:
        raise ValueError("São necessários pelo menos dois tópicos.")
    if min(counts.values()) < 5:
        raise ValueError("Cada tópico precisa de pelo menos cinco exemplos.")


def stratified_split(
    records: Iterable[Record],
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Divide os registros mantendo todos os tópicos em treino e teste."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction deve estar entre zero e um.")

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
