from __future__ import annotations

from typing import List, Dict, Any, Tuple
import math
import re


class SimpleVectorizer:
    """Lightweight bag-of-words vectorizer with L2 normalization.

    This is an in-memory, dependency-free approximation of a vector store.
    It supports add/search with cosine similarity based on token counts.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]

    def vectorize(self, text: str) -> Dict[int, float]:
        tokens = self.tokenize(text)
        counts: Dict[int, int] = {}
        for t in tokens:
            idx = self.vocab.setdefault(t, len(self.vocab))
            counts[idx] = counts.get(idx, 0) + 1
        # L2 normalize
        norm = math.sqrt(sum(c * c for c in counts.values())) or 1.0
        return {i: c / norm for i, c in counts.items()}


def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    keys = set(a.keys()) & set(b.keys())
    return sum(a[k] * b[k] for k in keys)


class InMemoryVectorStore:
    def __init__(self):
        self.vec = SimpleVectorizer()
        self.items: List[Tuple[str, Dict[str, Any], Dict[int, float]]] = []

    def add(self, item_id: str, payload: Dict[str, Any], text: str) -> None:
        v = self.vec.vectorize(text)
        self.items.append((item_id, payload, v))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        q = self.vec.vectorize(query)
        scored = [(iid, payload, cosine(q, v)) for iid, payload, v in self.items]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        q_tokens = set(self.vec.tokenize(query))
        results = []
        for iid, payload, v in self.items:
            text = payload.get("text_index", "")
            tokens = set(self.vec.tokenize(text))
            overlap = len(q_tokens & tokens)
            if overlap:
                results.append((iid, payload, float(overlap)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
