"""Persist stable sentence-template family metadata for generated review corpora."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

TEMPLATE_FAMILIES = (
    "noticed",
    "using",
    "stood_out",
    "context_component",
    "main_impression",
    "attention",
)


def infer_template_family(text: str) -> str:
    """Infer the generator template once so the value can be persisted with the record."""

    normalized = " ".join(str(text).lower().split())
    markers = (
        ("noticed", "i noticed that the"),
        ("using", "i have been using this"),
        ("stood_out", "is what stood out"),
        ("main_impression", "my main impression of this"),
        ("attention", "i did not pay much attention"),
    )
    for family, marker in markers:
        if marker in normalized:
            return family
    return "context_component"


def _materialize(path: Path) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name} has no header.")
        rows = list(reader)
        fields = list(reader.fieldnames)
    if "template_family" in fields and all(row.get("template_family") in TEMPLATE_FAMILIES for row in rows):
        return False

    if "template_family" not in fields:
        fields.append("template_family")
    for row in rows:
        family = row.get("template_family") or infer_template_family(row.get("text", ""))
        if family not in TEMPLATE_FAMILIES:
            raise ValueError(f"Unsupported template family in {path.name}: {family}")
        row["template_family"] = family

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _refresh_manifest(root: Path) -> None:
    manifest_path = root / "input_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [root / "train.csv", root / "incoming.csv"]
    manifest["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest["record_counts"] = {}
    for path in files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        manifest["record_counts"][path.name] = len(rows)
        if path.name == "incoming.csv":
            manifest["incoming_template_family_counts"] = dict(
                sorted(Counter(row["template_family"] for row in rows).items())
            )
    manifest["template_family_metadata"] = {
        "materialized": True,
        "families": list(TEMPLATE_FAMILIES),
        "legacy_rows": "backfilled_once_from_rendered_text; new pipeline states persist the value",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_template_metadata(input_dir: str | Path) -> dict[str, bool]:
    """Materialize template families in train/incoming CSVs before any split is made."""

    root = Path(input_dir)
    changed = {
        name: _materialize(root / name)
        for name in ("train.csv", "incoming.csv")
    }
    if any(changed.values()):
        _refresh_manifest(root)
    return changed
