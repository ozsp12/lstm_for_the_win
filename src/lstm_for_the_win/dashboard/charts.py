"""Plotly chart construction for the Streamlit dashboard."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import plotly.graph_objects as go


BLUE = "#2563eb"
BLUE_DARK = "#1e3a8a"
BLUE_LIGHT = "#93c5fd"
GOLD = "#d97706"
ORANGE = "#ea580c"
SLATE = "#64748b"
GRID = "#e2e8f0"

SENTIMENT_COLORS = {
    "negative": ORANGE,
    "neutral": SLATE,
    "positive": BLUE,
}


def _layout(figure: go.Figure, title: str, subtitle: str, *, height: int = 350) -> go.Figure:
    figure.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.01, "xanchor": "left"},
        height=height,
        margin={"l": 24, "r": 24, "t": 78, "b": 38},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, Segoe UI, Arial", "color": "#172033"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return figure


def sentiment_distribution_chart(rows: Sequence[dict[str, Any]]) -> go.Figure:
    """Show sentiment share as a single 100% stacked bar."""

    counts = Counter(str(row["sentiment"]) for row in rows)
    total = max(1, sum(counts.values()))
    figure = go.Figure()
    for label in ("negative", "neutral", "positive"):
        share = counts[label] / total
        figure.add_trace(
            go.Bar(
                name=label.title(),
                x=[share],
                y=["Reviews"],
                orientation="h",
                marker_color=SENTIMENT_COLORS[label],
                text=[f"{share:.1%}" if share >= 0.08 else ""],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"{label.title()}: %{{x:.1%}} ({counts[label]} reviews)<extra></extra>",
            )
        )
    figure.update_layout(barmode="stack", showlegend=True)
    figure.update_xaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID, title="Share of reviews")
    figure.update_yaxes(showgrid=False, title=None)
    return _layout(
        figure,
        "Sentiment distribution",
        f"Predicted sentiment, n={sum(counts.values())}",
        height=280,
    )


def topic_distribution_chart(rows: Sequence[dict[str, Any]]) -> go.Figure:
    """Compare predicted topic volumes with sorted horizontal bars."""

    counts = Counter(str(row["topic"]) for row in rows)
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    labels = [label.replace("_", " ").title() for label, _ in ordered]
    values = [value for _, value in ordered]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=BLUE,
            text=values,
            textposition="outside",
            hovertemplate="%{y}: %{x} reviews<extra></extra>",
        )
    )
    figure.update_xaxes(rangemode="tozero", gridcolor=GRID, title="Reviews")
    figure.update_yaxes(showgrid=False, title=None)
    return _layout(
        figure,
        "Topic distribution",
        f"Predicted product category, n={sum(values)}",
    )


def confidence_chart(rows: Sequence[dict[str, Any]]) -> go.Figure:
    """Compare prediction-confidence distributions for both models."""

    figure = go.Figure()
    for label, field, color in (
        ("Sentiment", "sentiment_confidence", BLUE),
        ("Topic", "topic_confidence", GOLD),
    ):
        figure.add_trace(
            go.Box(
                y=[float(row[field]) for row in rows],
                name=label,
                marker_color=color,
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                hovertemplate=f"{label}: %{{y:.1%}}<extra></extra>",
            )
        )
    figure.update_yaxes(tickformat=".0%", range=[0, 1.02], gridcolor=GRID, title="Confidence")
    figure.update_xaxes(showgrid=False, title=None)
    figure.add_hline(y=0.70, line_dash="dash", line_color=SLATE)
    return _layout(
        figure,
        "Prediction confidence",
        f"Each point is one model prediction, n={len(rows)} reviews",
    )


def training_accuracy_chart(results: dict[str, Any]) -> go.Figure:
    """Show training accuracy by epoch for the two classifiers."""

    figure = go.Figure()
    for task, color in (("sentiment", BLUE), ("topic", GOLD)):
        values = results[task]["history"].get("accuracy", [])
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                name=task.title(),
                mode="lines+markers",
                line={"color": color, "width": 3},
                marker={"size": 6},
                hovertemplate=f"{task.title()} · epoch %{{x}}: %{{y:.1%}}<extra></extra>",
            )
        )
    figure.update_xaxes(rangemode="tozero", gridcolor=GRID, title="Epoch")
    figure.update_yaxes(tickformat=".0%", range=[0, 1.02], gridcolor=GRID, title="Accuracy")
    return _layout(
        figure,
        "Training accuracy",
        "Optimization history; holdout accuracy is reported separately",
        height=390,
    )


def confusion_matrix_chart(task: str, result: dict[str, Any]) -> go.Figure:
    """Render one holdout confusion matrix."""

    labels = [label.replace("_", " ").title() for label in result["labels"]]
    figure = go.Figure(
        go.Heatmap(
            z=result["confusion_matrix"],
            x=labels,
            y=labels,
            colorscale=[[0, "#eff6ff"], [1, BLUE_DARK]],
            showscale=False,
            text=result["confusion_matrix"],
            texttemplate="%{text}",
            hovertemplate="Expected %{y}<br>Predicted %{x}<br>Reviews %{z}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Predicted", side="bottom")
    figure.update_yaxes(title="Expected", autorange="reversed")
    return _layout(
        figure,
        f"{task.title()} confusion matrix",
        f"Holdout evaluation, n={result['test_size']}",
        height=410,
    )


def top_terms_chart(term_counts: Iterable[tuple[str, int]]) -> go.Figure:
    """Render top word frequencies as a precise companion to the word cloud."""

    ordered = list(reversed(list(term_counts)))
    figure = go.Figure(
        go.Bar(
            x=[count for _, count in ordered],
            y=[term for term, _ in ordered],
            orientation="h",
            marker_color=BLUE,
            text=[count for _, count in ordered],
            textposition="outside",
            hovertemplate="%{y}: %{x} occurrences<extra></extra>",
        )
    )
    figure.update_xaxes(rangemode="tozero", gridcolor=GRID, title="Occurrences")
    figure.update_yaxes(showgrid=False, title=None)
    return _layout(
        figure,
        "Most frequent terms",
        "After normalization and stop-word removal",
        height=430,
    )


def probability_chart(probabilities: dict[str, float], title: str) -> go.Figure:
    """Show all class probabilities for one live prediction."""

    ordered = sorted(probabilities.items(), key=lambda item: item[1])
    figure = go.Figure(
        go.Bar(
            x=[value for _, value in ordered],
            y=[label.replace("_", " ").title() for label, _ in ordered],
            orientation="h",
            marker_color=BLUE,
            text=[f"{value:.1%}" for _, value in ordered],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID, title="Probability")
    figure.update_yaxes(showgrid=False, title=None)
    return _layout(figure, title, "Model probability by class", height=320)
