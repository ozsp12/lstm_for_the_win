from __future__ import annotations

import json
from pathlib import Path

import lstm_for_the_win.cli as cli


class _FakeHandler:
    calls: list[tuple[str, dict[str, object]]] = []

    def generate_inputs(self, config, input_dir, **kwargs):
        self.calls.append(("generate", {"config": config, "input_dir": input_dir, **kwargs}))
        path = Path(input_dir) / "input_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path

    def train_and_publish(self, input_dir, output_root, **kwargs):
        self.calls.append(("train", {"input_dir": input_dir, "output_root": output_root, **kwargs}))
        path = Path(output_root) / str(kwargs.get("run_id") or "generated")
        path.mkdir(parents=True, exist_ok=True)
        return path


def test_generate_data_command(monkeypatch, tmp_path: Path, capsys) -> None:
    _FakeHandler.calls.clear()
    monkeypatch.setattr(cli, "PipelineHandler", _FakeHandler)
    result = cli.main([
        "generate-data",
        "--config", str(tmp_path / "config.json"),
        "--input-dir", str(tmp_path / "input"),
        "--mode", "initialize",
        "--data-timestamp", "2026-08-18T12:00:00+00:00",
        "--overwrite",
    ])
    assert result == 0
    assert _FakeHandler.calls[0][0] == "generate"
    assert _FakeHandler.calls[0][1]["mode"] == "initialize"
    assert _FakeHandler.calls[0][1]["overwrite"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_train_command_with_explicit_validation_fraction(monkeypatch, tmp_path: Path, capsys) -> None:
    _FakeHandler.calls.clear()
    monkeypatch.setattr(cli, "PipelineHandler", _FakeHandler)
    result = cli.main([
        "train",
        "--input-dir", str(tmp_path / "input"),
        "--output-root", str(tmp_path / "output"),
        "--run-id", "r1",
        "--epochs", "2",
        "--validation-fraction", "0.25",
        "--patience", "1",
        "--seed", "7",
        "--split-seed", "8",
        "--replicate-seeds", "7,9",
    ])
    assert result == 0
    call = _FakeHandler.calls[0][1]
    assert call["validation_fraction"] == 0.25
    assert call["epochs"] == 2
    assert call["seed"] == 7
    assert call["split_seed"] == 8
    assert call["replicate_seeds"] == "7,9"
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation_fraction"] == 0.25


def test_train_uses_environment_validation_fraction(monkeypatch, tmp_path: Path) -> None:
    _FakeHandler.calls.clear()
    monkeypatch.setattr(cli, "PipelineHandler", _FakeHandler)
    monkeypatch.setenv("PIPELINE_VALIDATION_FRACTION", "0.22")
    assert cli.main([
        "train", "--input-dir", str(tmp_path / "input"), "--output-root", str(tmp_path / "output")
    ]) == 0
    assert _FakeHandler.calls[0][1]["validation_fraction"] == 0.22


def test_train_default_validation_fraction(monkeypatch, tmp_path: Path) -> None:
    _FakeHandler.calls.clear()
    monkeypatch.setattr(cli, "PipelineHandler", _FakeHandler)
    monkeypatch.delenv("PIPELINE_VALIDATION_FRACTION", raising=False)
    assert cli.main([
        "train", "--input-dir", str(tmp_path / "input"), "--output-root", str(tmp_path / "output")
    ]) == 0
    assert _FakeHandler.calls[0][1]["validation_fraction"] == 0.15


def test_run_uses_generation_config_and_advances(monkeypatch, tmp_path: Path, capsys) -> None:
    _FakeHandler.calls.clear()
    monkeypatch.setattr(cli, "PipelineHandler", _FakeHandler)
    monkeypatch.delenv("PIPELINE_VALIDATION_FRACTION", raising=False)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "input_manifest.json").write_text(json.dumps({"generation": 3}), encoding="utf-8")

    class _Config:
        def effective_generation(self, generation):
            assert generation == 3
            return {"validation_fraction": 0.19}

    monkeypatch.setattr(cli.SyntheticDataConfig, "from_json", lambda path: _Config())
    result = cli.main([
        "run",
        "--config", str(tmp_path / "config.json"),
        "--input-dir", str(input_dir),
        "--output-root", str(tmp_path / "output"),
        "--run-id", "r2",
        "--advance-data",
        "--data-timestamp", "2026-08-18T12:00:00+00:00",
    ])
    assert result == 0
    assert [kind for kind, _ in _FakeHandler.calls] == ["train", "generate"]
    assert _FakeHandler.calls[0][1]["validation_fraction"] == 0.19
    assert _FakeHandler.calls[1][1]["mode"] == "advance"
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation_fraction"] == 0.19
    assert "next_input_manifest" in payload


def test_run_without_manifest_uses_generation_zero(monkeypatch, tmp_path: Path) -> None:
    _FakeHandler.calls.clear()
    monkeypatch.setattr(cli, "PipelineHandler", _FakeHandler)
    monkeypatch.delenv("PIPELINE_VALIDATION_FRACTION", raising=False)

    class _Config:
        def effective_generation(self, generation):
            assert generation == 0
            return {"validation_fraction": 0.17}

    monkeypatch.setattr(cli.SyntheticDataConfig, "from_json", lambda path: _Config())
    assert cli.main([
        "run",
        "--config", str(tmp_path / "config.json"),
        "--input-dir", str(tmp_path / "input"),
        "--output-root", str(tmp_path / "output"),
    ]) == 0
    assert _FakeHandler.calls[0][1]["validation_fraction"] == 0.17
