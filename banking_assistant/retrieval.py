"""Dependency-free TF-IDF retrieval for the demo knowledge base."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
    "do", "for", "from", "how", "i", "in", "is", "it", "of", "on",
    "or", "the", "to", "what", "when", "where", "with", "you",
}
_KNOWLEDGE_BASE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "banking_faq.json"
)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOP_WORDS
    ]


def load_knowledge_base(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the bundled educational articles."""
    source = path or _KNOWLEDGE_BASE_PATH
    with source.open(encoding="utf-8") as file:
        articles = json.load(file)

    if not isinstance(articles, list) or not articles:
        raise ValueError("The banking knowledge base must be a non-empty list.")
    for article in articles:
        if not isinstance(article, dict) or not all(
            isinstance(article.get(field), str) and article[field].strip()
            for field in ("id", "title", "question", "answer", "source")
        ):
            raise ValueError(
                "Each knowledge article needs id, title, question, answer, and source."
            )
    return articles


def retrieve(
    query: str,
    knowledge_base: list[dict[str, Any]] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return best matching articles using TF-IDF cosine similarity."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    query_terms = _tokens(query)
    if not query_terms:
        return []

    articles = knowledge_base if knowledge_base is not None else load_knowledge_base()
    documents = [
        _tokens(
            " ".join(
                str(article.get(field, ""))
                for field in ("title", "question", "answer")
            )
        )
        for article in articles
    ]
    document_frequency = Counter(
        term for terms in documents for term in set(terms)
    )
    document_count = len(documents)

    def weight(term: str, count: int) -> float:
        inverse_document_frequency = math.log(
            1 + (document_count + 1) / (document_frequency[term] + 1)
        )
        return (1 + math.log(count)) * inverse_document_frequency

    query_vector = {
        term: weight(term, count) for term, count in Counter(query_terms).items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    ranked: list[tuple[float, int]] = []

    for index, terms in enumerate(documents):
        document_vector = {
            term: weight(term, count) for term, count in Counter(terms).items()
        }
        dot_product = sum(
            query_vector.get(term, 0.0) * value
            for term, value in document_vector.items()
        )
        document_norm = math.sqrt(
            sum(value * value for value in document_vector.values())
        )
        score = (
            dot_product / (query_norm * document_norm)
            if dot_product and query_norm and document_norm
            else 0.0
        )
        if score > 0:
            ranked.append((score, index))

    ranked.sort(key=lambda result: (-result[0], result[1]))
    return [
        {**articles[index], "score": score}
        for score, index in ranked[:limit]
    ]
