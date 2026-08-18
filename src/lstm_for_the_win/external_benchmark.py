"""Immutable real-world sentiment benchmark sourced from the UCI repository."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

DATASET_NAME = "Sentiment Labelled Sentences"
DATASET_DOI = "10.24432/C57604"
DATASET_LICENSE = "CC BY 4.0"
DATASET_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
SOURCE_MEMBER = "sentiment labelled sentences/amazon_cells_labelled.txt"
EXTERNAL_SUBDIR = "uci_sentiment_labelled_sentences"
DATA_FILE = "amazon_cells_labelled.tsv"
MANIFEST_FILE = "manifest.json"
EXPECTED_ROWS = 1_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("dataset_doi") != DATASET_DOI or manifest.get("license") != DATASET_LICENSE:
        raise ValueError("External benchmark provenance does not match the configured UCI dataset.")
    if manifest.get("sha256") != sha256(path):
        raise ValueError("External benchmark SHA-256 does not match its immutable manifest.")
    rows = load_external_sentiment(path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"External benchmark must contain exactly {EXPECTED_ROWS} Amazon review sentences.")
    if {row["expected_sentiment"] for row in rows} != {"negative", "positive"}:
        raise ValueError("External benchmark must contain both binary sentiment labels.")


def load_external_sentiment(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for number, row in enumerate(reader, start=1):
            if len(row) != 2 or row[1] not in {"0", "1"} or not row[0].strip():
                raise ValueError(f"Invalid external benchmark row {number}.")
            rows.append(
                {
                    "ID": f"uci-amazon-{number:04d}",
                    "text": row[0].strip(),
                    "expected_sentiment": "positive" if row[1] == "1" else "negative",
                }
            )
    return rows


def ensure_external_sentiment_benchmark(root: str | Path = "data/external") -> tuple[Path, dict[str, Any]]:
    """Download the CC-BY UCI Amazon sentiment subset once, then enforce immutability."""

    destination = Path(root) / EXTERNAL_SUBDIR
    data_path = destination / DATA_FILE
    manifest_path = destination / MANIFEST_FILE

    if data_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate(data_path, manifest)
        return data_path, manifest
    if data_path.exists() or manifest_path.exists():
        raise ValueError("External benchmark state is incomplete; refusing to overwrite it.")

    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:  # nosec B310: fixed HTTPS UCI URL
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        try:
            source = archive.read(SOURCE_MEMBER)
        except KeyError as error:
            raise ValueError("UCI archive does not contain the expected Amazon sentiment file.") from error

    text = source.decode("utf-8").replace("\r\n", "\n")
    data_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    rows = load_external_sentiment(data_path)
    if len(rows) != EXPECTED_ROWS:
        data_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded UCI Amazon subset contains {len(rows)} rows; expected {EXPECTED_ROWS}.")

    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "source_url": DATASET_URL,
        "source_member": SOURCE_MEMBER,
        "subset": "Amazon cell-phone review sentences",
        "task": "sentiment",
        "labels": ["negative", "positive"],
        "rows": len(rows),
        "sha256": sha256(data_path),
        "immutable": True,
        "redistribution_note": "UCI lists this dataset under Creative Commons Attribution 4.0 International.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate(data_path, manifest)
    return data_path, manifest
