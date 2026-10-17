import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class RetrievedPassage:
    title: str
    content: str
    score: float


class BankingKnowledgeBase:
    def __init__(self, passages: list[dict[str, str]]) -> None:
        if not passages:
            raise ValueError("The banking guide must contain at least one passage.")
        for passage in passages:
            if not passage.get("title") or not passage.get("content"):
                raise ValueError("Every banking guide passage needs a title and content.")

        self._passages = passages
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(
            [passage["content"] for passage in passages]
        )

    @classmethod
    def from_json(cls, path: Path) -> "BankingKnowledgeBase":
        with path.open(encoding="utf-8") as guide_file:
            passages = json.load(guide_file)
        if not isinstance(passages, list):
            raise ValueError("The banking guide JSON must contain a list of passages.")
        return cls(passages)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedPassage(
                title=self._passages[index]["title"],
                content=self._passages[index]["content"],
                score=float(scores[index]),
            )
            for index in ranked_indices
            if scores[index] > 0
        ]
