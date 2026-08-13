from __future__ import annotations

import os
import csv
from pathlib import Path

import pytest

from lstm_for_the_win.dashboard.charts import (
    confidence_chart,
    sentiment_distribution_chart,
    topic_distribution_chart,
)
from lstm_for_the_win.dashboard.data import load_run
from lstm_for_the_win.dashboard.text_insights import term_counts, wordcloud_image


def _integration_run() -> Path:
    value = os.getenv("PIPELINE_TEST_RUN_PATH")
    if not value:
        pytest.skip("PIPELINE_TEST_RUN_PATH is not configured")
    return Path(value)


def test_persisted_run_supports_all_dashboard_visuals() -> None:
    run_path = _integration_run()
    bundle = load_run(run_path)
    rows = bundle.inference_predictions

    assert rows
    assert sentiment_distribution_chart(rows).data
    assert topic_distribution_chart(rows).data
    assert confidence_chart(rows).data
    assert term_counts(row["text"] for row in rows)
    assert wordcloud_image(row["text"] for row in rows) is not None

    with (run_path / "predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)
        predictions = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames == [
        "ID",
        "text",
        "expected_sentiment",
        "expected_topic",
        "predicted_sentiment",
        "predicted_topic",
        "type",
        "input_timestamp",
        "model_timestamp",
    ]
    assert predictions
    assert {row["type"] for row in predictions} == {"test"}
    assert len({row["ID"] for row in predictions}) == len(predictions)
    assert all(row["input_timestamp"] and row["model_timestamp"] for row in predictions)
