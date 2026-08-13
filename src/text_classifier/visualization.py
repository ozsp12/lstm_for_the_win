"""Stakeholder-oriented HTML and SVG views for pipeline results."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Sequence

from .pipeline import PipelineResult


BLUE = "#2563eb"
BLUE_DARK = "#1e3a8a"
BLUE_LIGHT = "#dbeafe"
GOLD = "#d97706"
GOLD_LIGHT = "#fef3c7"
INK = "#172033"
MUTED = "#64748b"
BORDER = "#d8e0ea"
SURFACE = "#f8fafc"
WHITE = "#ffffff"
SUCCESS = "#166534"
SUCCESS_LIGHT = "#dcfce7"
ERROR = "#9f1239"
ERROR_LIGHT = "#ffe4e6"


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _polyline_points(values: Sequence[float], width: int, height: int) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"0,{height - (values[0] * height):.1f}"
    return " ".join(
        f"{index * width / (len(values) - 1):.1f},{height - (value * height):.1f}"
        for index, value in enumerate(values)
    )


def _training_chart(title: str, result: PipelineResult) -> str:
    values = result.history.get("accuracy", [])
    points = _polyline_points(values, width=560, height=170)
    final_value = values[-1] if values else 0.0
    return f"""
    <section class="stake-chart">
      <div class="stake-chart-heading">
        <div><strong>{escape(title)}</strong><span>Training accuracy by epoch</span></div>
        <b>{_percentage(final_value)}</b>
      </div>
      <svg viewBox="0 0 620 230" role="img" aria-label="{escape(title)} training accuracy curve">
        <line x1="45" y1="20" x2="45" y2="190" class="stake-axis" />
        <line x1="45" y1="190" x2="605" y2="190" class="stake-axis" />
        <line x1="45" y1="105" x2="605" y2="105" class="stake-grid" />
        <text x="36" y="25" text-anchor="end">100%</text>
        <text x="36" y="110" text-anchor="end">50%</text>
        <text x="36" y="195" text-anchor="end">0%</text>
        <text x="45" y="215">1</text>
        <text x="605" y="215" text-anchor="end">{len(values)}</text>
        <polyline points="{points}" transform="translate(45 20)" class="stake-line" />
      </svg>
    </section>
    """


def _distribution_chart(title: str, result: PipelineResult) -> str:
    maximum = max(result.label_counts.values(), default=1)
    rows = "".join(
        f"""
        <div class="stake-bar-row">
          <span>{escape(label.replace('_', ' ').title())}</span>
          <div class="stake-bar-track"><i style="width:{count / maximum * 100:.1f}%"></i></div>
          <b>{count}</b>
        </div>
        """
        for label, count in result.label_counts.items()
    )
    return f"""
    <section class="stake-chart stake-distribution">
      <div class="stake-chart-heading">
        <div><strong>{escape(title)}</strong><span>Examples per class, n={result.dataset_size}</span></div>
      </div>
      {rows}
    </section>
    """


def _confusion_matrix(title: str, result: PipelineResult) -> str:
    maximum = max((max(row) for row in result.confusion_matrix), default=1)
    headers = "".join(
        f"<th>{escape(label.replace('_', ' ').title())}</th>" for label in result.labels
    )
    rows = []
    for expected_label, values in zip(result.labels, result.confusion_matrix, strict=True):
        cells = "".join(
            f'<td style="--heat:{value / maximum:.3f}"><span>{value}</span></td>'
            for value in values
        )
        rows.append(
            f"<tr><th>{escape(expected_label.replace('_', ' ').title())}</th>{cells}</tr>"
        )
    return f"""
    <section class="stake-chart stake-matrix">
      <div class="stake-chart-heading">
        <div><strong>{escape(title)}</strong><span>Rows: expected · Columns: predicted</span></div>
      </div>
      <div class="stake-table-scroll">
        <table><thead><tr><th></th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
      </div>
    </section>
    """


def _prediction_review(title: str, result: PipelineResult, limit: int = 6) -> str:
    ordered = sorted(result.predictions, key=lambda item: (item["correct"], item["text"]))
    rows = []
    for item in ordered[:limit]:
        status = "Correct" if item["correct"] else "Review"
        status_class = "is-correct" if item["correct"] else "is-review"
        rows.append(
            f"""
            <tr>
              <td>{escape(str(item['text']))}</td>
              <td>{escape(str(item['expected']).replace('_', ' '))}</td>
              <td>{escape(str(item['predicted']).replace('_', ' '))}</td>
              <td>{_percentage(float(item['confidence']))}</td>
              <td><span class="stake-status {status_class}">{status}</span></td>
            </tr>
            """
        )
    return f"""
    <section class="stake-chart stake-review">
      <div class="stake-chart-heading">
        <div><strong>{escape(title)}</strong><span>Errors first, then representative correct predictions</span></div>
      </div>
      <div class="stake-table-scroll">
        <table>
          <thead><tr><th>Review</th><th>Expected</th><th>Predicted</th><th>Confidence</th><th>Status</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def _demo_table(sentiment: PipelineResult, topic: PipelineResult) -> str:
    sentiment_by_text = {item["text"]: item for item in sentiment.demo_predictions}
    topic_by_text = {item["text"]: item for item in topic.demo_predictions}
    shared_texts = [text for text in sentiment_by_text if text in topic_by_text]
    rows = []
    for text in shared_texts:
        sentiment_item = sentiment_by_text[text]
        topic_item = topic_by_text[text]
        action = _recommended_action(
            str(sentiment_item["predicted"]),
            str(topic_item["predicted"]),
        )
        rows.append(
            f"""
            <tr>
              <td>{escape(text)}</td>
              <td><strong>{escape(str(sentiment_item['predicted']).title())}</strong><br><small>{_percentage(float(sentiment_item['confidence']))} confidence</small></td>
              <td><strong>{escape(str(topic_item['predicted']).replace('_', ' ').title())}</strong><br><small>{_percentage(float(topic_item['confidence']))} confidence</small></td>
              <td>{escape(action)}</td>
            </tr>
            """
        )
    probability_detail = ""
    if shared_texts:
        selected_text = shared_texts[0]
        probability_detail = f"""
        <div class="stake-probability-detail">
          <p><strong>Confidence profile:</strong> {escape(selected_text)}</p>
          <div class="stake-grid-two">
            <div><h4>Sentiment probabilities</h4>{render_probability_bars(sentiment_by_text[selected_text])}</div>
            <div><h4>Topic probabilities</h4>{render_probability_bars(topic_by_text[selected_text])}</div>
          </div>
        </div>
        """
    return f"""
    <section class="stake-chart stake-demo">
      <div class="stake-chart-heading">
        <div><strong>Business demonstration</strong><span>One review, two model outputs, one suggested routing decision</span></div>
      </div>
      <div class="stake-table-scroll">
        <table>
          <thead><tr><th>Incoming review</th><th>Sentiment</th><th>Topic</th><th>Suggested action</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      {probability_detail}
    </section>
    """


def _recommended_action(sentiment: str, topic: str) -> str:
    team = topic.replace("_", " ").title()
    if sentiment == "negative":
        return f"Prioritize and route to the {team} support team."
    if sentiment == "positive":
        return f"Route to the {team} insights queue for advocacy analysis."
    return f"Add to the {team} monitoring queue."


def render_stakeholder_dashboard(
    sentiment: PipelineResult,
    topic: PipelineResult,
) -> str:
    """Render a self-contained stakeholder dashboard as an HTML fragment."""

    return f"""
<div class="stakeholder-dashboard">
  <style>
    .stakeholder-dashboard {{font-family:Inter,Segoe UI,Arial,sans-serif;color:{INK};max-width:1180px;margin:0 auto}}
    .stakeholder-dashboard * {{box-sizing:border-box}}
    .stake-header {{padding:24px 26px;background:{BLUE_DARK};color:{WHITE};border-radius:18px}}
    .stake-header h2 {{margin:0 0 6px;font-size:26px}}
    .stake-header p {{margin:0;opacity:.86}}
    .stake-caveat {{margin:14px 0 0;padding:10px 12px;background:rgba(255,255,255,.12);border-radius:9px;font-size:13px}}
    .stake-kpis {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}}
    .stake-kpi {{padding:16px;background:{SURFACE};border:1px solid {BORDER};border-radius:14px}}
    .stake-kpi span {{display:block;color:{MUTED};font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
    .stake-kpi strong {{display:block;font-size:28px;margin-top:6px}}
    .stake-grid-two {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:14px 0}}
    .stake-chart {{padding:18px;border:1px solid {BORDER};border-radius:14px;background:{WHITE}}}
    .stake-chart-heading {{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:14px}}
    .stake-chart-heading strong {{display:block;font-size:16px}}
    .stake-chart-heading span {{display:block;color:{MUTED};font-size:12px;margin-top:3px}}
    .stake-chart-heading b {{font-size:20px;color:{BLUE_DARK}}}
    .stake-chart svg {{width:100%;height:auto}}
    .stake-chart svg text {{fill:{MUTED};font-size:12px}}
    .stake-axis {{stroke:{MUTED};stroke-width:1}}
    .stake-grid {{stroke:{BORDER};stroke-width:1}}
    .stake-line {{fill:none;stroke:{BLUE};stroke-width:4;stroke-linejoin:round;stroke-linecap:round}}
    .stake-bar-row {{display:grid;grid-template-columns:132px 1fr 28px;gap:10px;align-items:center;margin:10px 0;font-size:13px}}
    .stake-bar-track {{height:14px;background:{BLUE_LIGHT};border-radius:8px;overflow:hidden}}
    .stake-bar-track i {{display:block;height:100%;background:{BLUE}}}
    .stake-table-scroll {{overflow-x:auto}}
    .stakeholder-dashboard table {{width:100%;border-collapse:collapse;font-size:13px}}
    .stakeholder-dashboard th,.stakeholder-dashboard td {{padding:10px;border-bottom:1px solid {BORDER};text-align:left;vertical-align:top}}
    .stakeholder-dashboard thead th {{color:{MUTED};font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
    .stake-matrix td {{text-align:center;background:color-mix(in srgb, {BLUE} calc(var(--heat)*78%), {WHITE})}}
    .stake-matrix td span {{font-weight:700}}
    .stake-status {{display:inline-block;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:700}}
    .stake-status.is-correct {{background:{SUCCESS_LIGHT};color:{SUCCESS}}}
    .stake-status.is-review {{background:{ERROR_LIGHT};color:{ERROR}}}
    .stake-review,.stake-demo {{margin:14px 0}}
    .stake-demo small {{color:{MUTED}}}
    .stake-probability-detail {{margin-top:18px;padding-top:16px;border-top:1px solid {BORDER}}}
    .stake-probability-detail p,.stake-probability-detail h4 {{margin:0 0 10px}}
    .stake-probability-detail h4 {{font-size:13px;color:{MUTED}}}
    .stake-probability-detail .stake-grid-two {{margin:0}}
    .stake-probability-detail .stake-grid-two>div>div {{display:grid;grid-template-columns:120px 1fr 54px;gap:8px;align-items:center;margin:8px 0;font-size:12px}}
    .stake-probability-detail progress {{width:100%;accent-color:{BLUE}}}
    @media(max-width:760px) {{.stake-kpis,.stake-grid-two {{grid-template-columns:1fr 1fr}}}}
    @media(max-width:520px) {{.stake-kpis,.stake-grid-two {{grid-template-columns:1fr}}.stake-bar-row {{grid-template-columns:110px 1fr 24px}}}}
  </style>
  <header class="stake-header">
    <h2>Product Review Intelligence</h2>
    <p>Sentiment and product-topic classification for triage, monitoring, and insight discovery.</p>
    <div class="stake-caveat"><strong>Demonstration scope:</strong> results use small, balanced synthetic datasets. Real-world readiness requires representative data, confidence calibration, and business acceptance thresholds.</div>
  </header>
  <div class="stake-kpis">
    <div class="stake-kpi"><span>Sentiment accuracy</span><strong>{_percentage(sentiment.metrics['accuracy'])}</strong></div>
    <div class="stake-kpi"><span>Topic accuracy</span><strong>{_percentage(topic.metrics['accuracy'])}</strong></div>
    <div class="stake-kpi"><span>Evaluated reviews</span><strong>{sentiment.test_size + topic.test_size}</strong></div>
    <div class="stake-kpi"><span>Supported classes</span><strong>{len(sentiment.labels) + len(topic.labels)}</strong></div>
  </div>
  <div class="stake-grid-two">
    {_training_chart('Sentiment model', sentiment)}
    {_training_chart('Topic model', topic)}
  </div>
  <div class="stake-grid-two">
    {_distribution_chart('Sentiment class balance', sentiment)}
    {_distribution_chart('Topic class balance', topic)}
  </div>
  <div class="stake-grid-two">
    {_confusion_matrix('Sentiment confusion matrix', sentiment)}
    {_confusion_matrix('Topic confusion matrix', topic)}
  </div>
  {_prediction_review('Topic prediction review', topic)}
  {_demo_table(sentiment, topic)}
</div>
"""


def build_stakeholder_overview_svg(
    sentiment: PipelineResult,
    topic: PipelineResult,
) -> str:
    """Build a static executive overview suitable for the repository README."""

    sentiment_accuracy = _percentage(sentiment.metrics["accuracy"])
    topic_accuracy = _percentage(topic.metrics["accuracy"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560" role="img" aria-labelledby="title desc">
  <title id="title">Stakeholder overview for product review intelligence</title>
  <desc id="desc">Sentiment and topic model performance, supported outputs, and business routing demonstration.</desc>
  <rect width="1200" height="560" fill="{WHITE}"/>
  <rect x="30" y="25" width="1140" height="100" rx="18" fill="{BLUE_DARK}"/>
  <text x="65" y="70" font-family="Segoe UI,Arial" font-size="28" font-weight="700" fill="{WHITE}">Product Review Intelligence</text>
  <text x="65" y="99" font-family="Segoe UI,Arial" font-size="16" fill="{BLUE_LIGHT}">One review → sentiment + product topic → business routing</text>
  <rect x="30" y="145" width="270" height="130" rx="16" fill="{SURFACE}" stroke="{BORDER}"/>
  <text x="55" y="180" font-family="Segoe UI,Arial" font-size="14" fill="{MUTED}">SENTIMENT ACCURACY</text>
  <text x="55" y="235" font-family="Segoe UI,Arial" font-size="42" font-weight="700" fill="{BLUE_DARK}">{sentiment_accuracy}</text>
  <rect x="320" y="145" width="270" height="130" rx="16" fill="{SURFACE}" stroke="{BORDER}"/>
  <text x="345" y="180" font-family="Segoe UI,Arial" font-size="14" fill="{MUTED}">TOPIC ACCURACY</text>
  <text x="345" y="235" font-family="Segoe UI,Arial" font-size="42" font-weight="700" fill="{BLUE_DARK}">{topic_accuracy}</text>
  <rect x="610" y="145" width="270" height="130" rx="16" fill="{SURFACE}" stroke="{BORDER}"/>
  <text x="635" y="180" font-family="Segoe UI,Arial" font-size="14" fill="{MUTED}">SUPPORTED OUTPUTS</text>
  <text x="635" y="224" font-family="Segoe UI,Arial" font-size="34" font-weight="700" fill="{BLUE_DARK}">{len(sentiment.labels) + len(topic.labels)} classes</text>
  <rect x="900" y="145" width="270" height="130" rx="16" fill="{GOLD_LIGHT}" stroke="{GOLD}"/>
  <text x="925" y="180" font-family="Segoe UI,Arial" font-size="14" fill="{GOLD}">DEMONSTRATION DATA</text>
  <text x="925" y="220" font-family="Segoe UI,Arial" font-size="24" font-weight="700" fill="{INK}">Synthetic &amp; balanced</text>
  <text x="925" y="248" font-family="Segoe UI,Arial" font-size="14" fill="{MUTED}">Not a production benchmark</text>
  <text x="30" y="325" font-family="Segoe UI,Arial" font-size="20" font-weight="700" fill="{INK}">Stakeholder demonstration</text>
  <rect x="30" y="350" width="250" height="145" rx="16" fill="{SURFACE}" stroke="{BORDER}"/>
  <text x="55" y="385" font-family="Segoe UI,Arial" font-size="14" fill="{MUTED}">1 · INCOMING REVIEW</text>
  <text x="55" y="423" font-family="Segoe UI,Arial" font-size="17" font-weight="700" fill="{INK}">“The phone battery is</text>
  <text x="55" y="448" font-family="Segoe UI,Arial" font-size="17" font-weight="700" fill="{INK}">terrible and drains fast.”</text>
  <path d="M285 422 H330" stroke="{MUTED}" stroke-width="3" marker-end="url(#arrow)"/>
  <rect x="340" y="350" width="210" height="145" rx="16" fill="{BLUE_LIGHT}" stroke="{BLUE}"/>
  <text x="365" y="385" font-family="Segoe UI,Arial" font-size="14" fill="{BLUE_DARK}">2 · SENTIMENT</text>
  <text x="365" y="440" font-family="Segoe UI,Arial" font-size="30" font-weight="700" fill="{BLUE_DARK}">Negative</text>
  <path d="M555 422 H600" stroke="{MUTED}" stroke-width="3" marker-end="url(#arrow)"/>
  <rect x="610" y="350" width="210" height="145" rx="16" fill="{BLUE_LIGHT}" stroke="{BLUE}"/>
  <text x="635" y="385" font-family="Segoe UI,Arial" font-size="14" fill="{BLUE_DARK}">3 · TOPIC</text>
  <text x="635" y="440" font-family="Segoe UI,Arial" font-size="30" font-weight="700" fill="{BLUE_DARK}">Smartphone</text>
  <path d="M825 422 H870" stroke="{MUTED}" stroke-width="3" marker-end="url(#arrow)"/>
  <rect x="880" y="350" width="290" height="145" rx="16" fill="{SUCCESS_LIGHT}" stroke="{SUCCESS}"/>
  <text x="905" y="385" font-family="Segoe UI,Arial" font-size="14" fill="{SUCCESS}">4 · SUGGESTED ACTION</text>
  <text x="905" y="425" font-family="Segoe UI,Arial" font-size="18" font-weight="700" fill="{INK}">Prioritize and route to the</text>
  <text x="905" y="452" font-family="Segoe UI,Arial" font-size="18" font-weight="700" fill="{INK}">Smartphone support team</text>
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{MUTED}"/></marker></defs>
</svg>"""


def write_stakeholder_overview(
    path: str | Path,
    sentiment: PipelineResult,
    topic: PipelineResult,
) -> Path:
    """Write the README overview SVG and return its resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_stakeholder_overview_svg(sentiment, topic),
        encoding="utf-8",
    )
    return output_path.resolve()


def render_probability_bars(prediction: dict[str, object]) -> str:
    """Render class probabilities for a single demonstration prediction."""

    probabilities = prediction.get("probabilities", {})
    if not isinstance(probabilities, dict):
        return ""
    rows: Iterable[str] = (
        f'<div><span>{escape(str(label).replace("_", " ").title())}</span>'
        f'<progress max="1" value="{float(value):.6f}"></progress>'
        f'<b>{_percentage(float(value))}</b></div>'
        for label, value in probabilities.items()
    )
    return "".join(rows)
