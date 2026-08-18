from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import lstm_for_the_win.external_benchmark as external_module
from lstm_for_the_win.external_benchmark import ensure_external_sentiment_benchmark, load_external_sentiment


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.payload


def _archive() -> bytes:
    content = "".join(
        f"review {index}\t{1 if index % 2 else 0}\n"
        for index in range(1, 1001)
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(external_module.SOURCE_MEMBER, content)
    return buffer.getvalue()


def test_external_benchmark_is_downloaded_once_and_hash_validated(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def fake_urlopen(url, timeout):
        nonlocal calls
        calls += 1
        assert url == external_module.DATASET_URL
        assert timeout == 60
        return _Response(_archive())

    monkeypatch.setattr(external_module.urllib.request, "urlopen", fake_urlopen)
    data_path, manifest = ensure_external_sentiment_benchmark(tmp_path / "external")
    assert calls == 1
    assert len(load_external_sentiment(data_path)) == 1000
    assert manifest["dataset_doi"] == "10.24432/C57604"
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["task"] == "sentiment"

    same_path, same_manifest = ensure_external_sentiment_benchmark(tmp_path / "external")
    assert same_path == data_path
    assert same_manifest == manifest
    assert calls == 1

    data_path.write_text(data_path.read_text(encoding="utf-8") + "tampered\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_external_sentiment_benchmark(tmp_path / "external")


def test_external_manifest_is_machine_readable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(external_module.urllib.request, "urlopen", lambda url, timeout: _Response(_archive()))
    data_path, _ = ensure_external_sentiment_benchmark(tmp_path / "external")
    manifest_path = data_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["immutable"] is True
    assert manifest["rows"] == 1000
