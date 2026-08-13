"""Stakeholder-facing Streamlit application for persisted pipeline runs."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import streamlit as st

from lstm_for_the_win.dashboard.charts import (
    confidence_chart,
    confusion_matrix_chart,
    probability_chart,
    sentiment_distribution_chart,
    top_terms_chart,
    topic_distribution_chart,
    training_accuracy_chart,
)
from lstm_for_the_win.dashboard.data import (
    RunBundle,
    discover_runs,
    filter_predictions,
    load_run,
)
from lstm_for_the_win.dashboard.inference import classify_text, load_models
from lstm_for_the_win.dashboard.text_insights import term_counts, wordcloud_image


st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="💬",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_load_run(path: str) -> RunBundle:
    """Cache deterministic run-artifact reads across UI reruns."""

    return load_run(path)


@st.cache_resource(show_spinner="Loading trained models...")
def cached_load_models(path: str):
    """Load each pair of trained models once per server process."""

    return load_models(path)


def _human_label(value: str) -> str:
    return value.replace("_", " ").title()


def _share(rows: list[dict[str, object]], field: str, value: str) -> float:
    if not rows:
        return 0.0
    return sum(str(row[field]) == value for row in rows) / len(rows)


def _top_value(rows: list[dict[str, object]], field: str) -> str:
    if not rows:
        return "—"
    return _human_label(Counter(str(row[field]) for row in rows).most_common(1)[0][0])


def _render_header(bundle: RunBundle) -> None:
    st.title("Product Review Intelligence")
    st.markdown(
        "Sentiment and product-topic classification for support triage, monitoring, "
        "and insight discovery."
    )
    st.warning(
        "Demonstration scope: all displayed reviews are synthetic. Model quality on "
        "representative production data has not been established."
    )
    st.caption(
        f"Run `{bundle.manifest['run_id']}` · created {bundle.manifest['created_at']} · "
        f"TensorFlow {bundle.manifest['tensorflow_version']}"
    )


def _render_overview(rows: list[dict[str, object]]) -> None:
    st.subheader("Current review overview")
    metric_columns = st.columns([0.7, 0.9, 1.6, 0.9])
    metric_columns[0].metric("Reviews", f"{len(rows):,}", border=True)
    metric_columns[1].metric(
        "Negative share",
        f"{_share(rows, 'sentiment', 'negative'):.1%}",
        border=True,
    )
    metric_columns[2].metric("Leading topic", _top_value(rows, "topic"), border=True)
    low_confidence = sum(
        min(float(row["sentiment_confidence"]), float(row["topic_confidence"])) < 0.70
        for row in rows
    )
    metric_columns[3].metric(
        "Low confidence",
        f"{low_confidence:,}",
        help="At least one model is below 70% confidence.",
        border=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            sentiment_distribution_chart(rows),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            topic_distribution_chart(rows),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )
    st.plotly_chart(
        confidence_chart(rows),
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
    )


def _render_text_insights(rows: list[dict[str, object]]) -> None:
    st.subheader("Language patterns")
    st.caption(
        "Word size represents frequency after normalization and stop-word removal; "
        "it does not measure causal importance."
    )
    texts = [str(row["text"]) for row in rows]
    cloud, terms = st.columns([3, 2])
    with cloud:
        st.markdown("#### Word cloud")
        image = wordcloud_image(texts)
        if image is None:
            st.info("No terms are available for the selected filters.")
        else:
            st.image(image, width="stretch")
    with terms:
        st.plotly_chart(
            top_terms_chart(term_counts(texts)),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )

    st.markdown("#### Review explorer")
    display_rows = [
        {
            "Review": row["text"],
            "Sentiment": _human_label(str(row["sentiment"])),
            "Sentiment confidence": f"{float(row['sentiment_confidence']):.1%}",
            "Topic": _human_label(str(row["topic"])),
            "Topic confidence": f"{float(row['topic_confidence']):.1%}",
            "Suggested action": row["suggested_action"],
        }
        for row in rows[:20]
    ]
    st.table(display_rows)


def _render_model_quality(bundle: RunBundle) -> None:
    st.subheader("Model quality")
    sentiment = bundle.results["sentiment"]
    topic = bundle.results["topic"]
    left, right = st.columns(2)
    left.metric(
        "Sentiment holdout accuracy",
        f"{float(sentiment['metrics']['accuracy']):.1%}",
        help=f"n={sentiment['test_size']} holdout reviews",
        border=True,
    )
    right.metric(
        "Topic holdout accuracy",
        f"{float(topic['metrics']['accuracy']):.1%}",
        help=f"n={topic['test_size']} holdout reviews",
        border=True,
    )
    st.plotly_chart(
        training_accuracy_chart(bundle.results),
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            confusion_matrix_chart("sentiment", sentiment),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            confusion_matrix_chart("topic", topic),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )


def _render_live_prediction(bundle: RunBundle) -> None:
    st.subheader("Try both models")
    review = st.text_area(
        "Product review",
        value="The phone battery is unreliable and drains too quickly.",
        height=110,
    )
    if not st.button("Classify review", type="primary"):
        st.caption("The review is processed only when you select Classify review.")
        return

    try:
        sentiment_model, topic_model = cached_load_models(str(bundle.path))
        sentiment = classify_text(
            sentiment_model,
            list(bundle.results["sentiment"]["labels"]),
            review,
        )
        topic = classify_text(
            topic_model,
            list(bundle.results["topic"]["labels"]),
            review,
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    left, right = st.columns(2)
    left.metric(
        "Predicted sentiment",
        _human_label(str(sentiment["predicted"])),
        f"{float(sentiment['confidence']):.1%} confidence",
        delta_color="off",
        border=True,
    )
    right.metric(
        "Predicted topic",
        _human_label(str(topic["predicted"])),
        f"{float(topic['confidence']):.1%} confidence",
        delta_color="off",
        border=True,
    )
    if sentiment["predicted"] == "negative":
        action = f"Prioritize and route to the {_human_label(str(topic['predicted']))} support team."
    elif sentiment["predicted"] == "positive":
        action = f"Route to the {_human_label(str(topic['predicted']))} insights queue."
    else:
        action = f"Add to the {_human_label(str(topic['predicted']))} monitoring queue."
    st.info(f"Illustrative action: {action}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            probability_chart(dict(sentiment["probabilities"]), "Sentiment probabilities"),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )
    with right:
        st.plotly_chart(
            probability_chart(dict(topic["probabilities"]), "Topic probabilities"),
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
        )


def main() -> None:
    """Render the application from persisted output artifacts."""

    output_root = Path(os.getenv("PIPELINE_OUTPUT_ROOT", "data/output"))
    runs = discover_runs(output_root)
    if not runs:
        st.title("Product Review Intelligence")
        st.warning("No completed pipeline run was found.")
        st.code("lstm-pipeline run --epochs 20", language="bash")
        st.stop()

    run_options = {path.name: path for path in runs}
    selected_run = st.sidebar.selectbox("Pipeline run", list(run_options))
    bundle = cached_load_run(str(run_options[selected_run]))
    all_rows = bundle.inference_predictions

    sentiment_options = sorted({str(row["sentiment"]) for row in all_rows})
    topic_options = sorted({str(row["topic"]) for row in all_rows})
    selected_sentiment = st.sidebar.selectbox(
        "Sentiment",
        ["All", *map(_human_label, sentiment_options)],
    )
    selected_topic = st.sidebar.selectbox(
        "Topic",
        ["All", *map(_human_label, topic_options)],
    )
    sentiment_filter = (
        None
        if selected_sentiment == "All"
        else sentiment_options[[*map(_human_label, sentiment_options)].index(selected_sentiment)]
    )
    topic_filter = (
        None
        if selected_topic == "All"
        else topic_options[[*map(_human_label, topic_options)].index(selected_topic)]
    )
    filtered_rows = filter_predictions(
        all_rows,
        sentiment=sentiment_filter,
        topic=topic_filter,
    )

    _render_header(bundle)
    if not filtered_rows:
        st.info("No reviews match the selected filters.")
        return

    overview, insights, quality, live = st.tabs(
        ["Overview", "Text insights", "Model quality", "Try the model"]
    )
    with overview:
        _render_overview(filtered_rows)
    with insights:
        _render_text_insights(filtered_rows)
    with quality:
        _render_model_quality(bundle)
    with live:
        _render_live_prediction(bundle)


if __name__ == "__main__":
    main()
