"""Word-frequency and word-cloud helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from wordcloud import STOPWORDS, WordCloud

from ..classification.data import clean_text


PROJECT_STOPWORDS = set(STOPWORDS) | {
    "product",
    "device",
    "item",
    "overall",
    "use",
    "used",
}


def term_counts(texts: Iterable[str], limit: int = 12) -> list[tuple[str, int]]:
    """Return normalized non-stopword frequencies."""

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(
            word
            for word in clean_text(text).split()
            if len(word) > 2 and word not in PROJECT_STOPWORDS
        )
    return counter.most_common(limit)


def wordcloud_image(texts: Iterable[str]) -> object | None:
    """Generate an in-memory word-cloud image without writing files."""

    frequencies = dict(term_counts(texts, limit=100))
    if not frequencies:
        return None
    return WordCloud(
        width=1200,
        height=520,
        background_color="white",
        colormap="Blues",
        stopwords=PROJECT_STOPWORDS,
        collocations=False,
        random_state=42,
    ).generate_from_frequencies(frequencies).to_array()
