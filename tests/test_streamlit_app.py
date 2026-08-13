from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_and_runs_live_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    run_path = os.getenv("PIPELINE_TEST_RUN_PATH")
    if not run_path:
        pytest.skip("PIPELINE_TEST_RUN_PATH is not configured")
    monkeypatch.setenv("PIPELINE_OUTPUT_ROOT", str(Path(run_path).parent))

    app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()

    assert not app.exception
    assert len(app.tabs) == 4
    assert len(app.metric) >= 6
    assert app.table

    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert len(app.metric) >= 8
