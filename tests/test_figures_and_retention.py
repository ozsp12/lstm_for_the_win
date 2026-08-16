from __future__ import annotations

import csv
import json
from pathlib import Path

from lstm_for_the_win.analysis.article_analysis import ANALYSIS_COLUMNS
from lstm_for_the_win.analysis.dashboard_figures import export_dashboard_figures
from lstm_for_the_win.analysis.figures import export_figures
from lstm_for_the_win.analysis.retention import prune_output_runs


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in ANALYSIS_COLUMNS}
    row.update({"run_id": "20260816T160000Z_github-1", "input_generation": 1})
    row.update(updates)
    return row


def test_export_figures_creates_png_svg_and_manifest(tmp_path: Path) -> None:
    run = tmp_path / "20260816T160000Z_github-1"
    run.mkdir()
    rows: list[dict[str, object]] = []
    for task, labels in (("sentiment", ["negative", "positive"]), ("topic", ["smartphone", "television"])):
        for model, score in (("lstm", 0.9), ("tfidf_logistic", 0.8)):
            for metric, value in (
                ("accuracy", score), ("macro_f1", score - 0.01), ("weighted_f1", score - 0.01),
                ("log_loss", 1.0 - score), ("brier_score", (1.0 - score) / 2),
                ("expected_calibration_error", (1.0 - score) / 3),
            ):
                rows.append(_row(analysis_group="aggregate_metrics", task=task, model=model, metric=metric, value=value, support=10))
        for expected in labels:
            for predicted in labels:
                rows.append(_row(
                    analysis_group="confusion_matrix", task=task, model="lstm",
                    expected_label=expected, predicted_label=predicted, metric="count",
                    value=4 if expected == predicted else 1, support=5,
                ))
        for label in labels:
            for metric, value in (("precision", 0.9), ("recall", 0.9), ("specificity", 0.95), ("f1", 0.9), ("support", 5)):
                rows.append(_row(analysis_group="classwise_metrics", task=task, model="lstm", class_label=label, metric=metric, value=value, support=5))
            rows.append(_row(analysis_group="class_distribution", task=task, model="lstm", class_label=label, metric="expected_proportion", value=0.5, support=10))
            rows.append(_row(analysis_group="class_distribution", task=task, model="lstm", class_label=label, metric="predicted_proportion", value=0.5, support=10))
        for bin_value, confidence, accuracy in (("[0.7,0.8)", 0.75, 0.7), ("[0.9,1.0]", 0.95, 1.0)):
            rows.append(_row(analysis_group="calibration_bins", task=task, model="lstm", segment_dimension="confidence_bin", segment_value=bin_value, metric="mean_confidence", value=confidence, support=5))
            rows.append(_row(analysis_group="calibration_bins", task=task, model="lstm", segment_dimension="confidence_bin", segment_value=bin_value, metric="accuracy", value=accuracy, support=5))
        for epoch in (1, 2):
            rows.append(_row(analysis_group="training_history", task=task, model="lstm", segment_dimension="epoch", segment_value=epoch, metric="accuracy", value=0.7 + 0.1 * epoch, support=20))
            rows.append(_row(analysis_group="training_history", task=task, model="lstm", segment_dimension="epoch", segment_value=epoch, metric="val_accuracy", value=0.65 + 0.1 * epoch, support=5))
        for value, score in (("0", 0.88), ("1", 0.92)):
            rows.append(_row(analysis_group="segment_metrics", task=task, model="lstm", segment_dimension="hasemoji", segment_value=value, metric="accuracy", value=score, support=5))
            rows.append(_row(analysis_group="segment_metrics", task=task, model="lstm", segment_dimension="hasemoji", segment_value=value, metric="macro_f1", value=score - 0.01, support=5))
        rows.append(_row(analysis_group="confidence_accuracy", task=task, model="lstm", metric="accuracy", value=0.9, support=10))
        rows.append(_row(analysis_group="confidence_accuracy", task=task, model="lstm", metric="mean_selected_class_confidence", value=0.93, support=10))
        rows.append(_row(analysis_group="error_record", task=task, model="lstm", metric="confidence", value=0.82, record_id="9", expected_label=labels[0], predicted_label=labels[1]))

    with (run / "article_analysis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (run / "run_manifest.json").write_text(json.dumps({"run_id": run.name, "outputs": {}}), encoding="utf-8")

    export_figures(run)
    manifest_path = export_dashboard_figures(run)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (run / "figures" / "confusion_matrix_sentiment.png").is_file()
    assert (run / "figures" / "confusion_matrix_topic.svg").is_file()
    assert (run / "figures" / "segment_hasemoji_sentiment.png").is_file()
    assert (run / "figures" / "dashboard_model_accuracy.png").is_file()
    assert (run / "figures" / "dashboard_sentiment_class_recall.svg").is_file()
    assert (run / "figures" / "dashboard_topic_volume_accuracy.png").is_file()
    assert (run / "figures" / "dashboard_confidence_vs_accuracy.png").is_file()
    assert {item["format"] for item in manifest["figure_files"]} == {"png", "svg"}
    updated = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert updated["figures"]["logical_figures"] > 0
    assert updated["outputs"]["figures_manifest"] == "figures/figures_manifest.json"


def test_prune_output_runs_keeps_only_requested_run(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    keep = "20260816T160000Z_github-3"
    for name in ("github-1", "github-2", keep, "local-streamlit-old"):
        directory = root / name
        directory.mkdir()
        (directory / "artifact.txt").write_text(name, encoding="utf-8")
    (root / "latest.json").write_text(json.dumps({"run_id": keep}), encoding="utf-8")
    (root / ".gitkeep").write_text("\n", encoding="utf-8")

    removed = prune_output_runs(root, keep)
    assert set(removed) == {"github-1", "github-2", "local-streamlit-old"}
    assert (root / keep).is_dir()
    assert (root / "latest.json").is_file()
    assert not (root / "github-1").exists()
