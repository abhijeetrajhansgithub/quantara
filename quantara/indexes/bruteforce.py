from typing import Any

from quantara.indexes.base import BaseIndex
from quantara.utils.utils import cosine_similarity, dot_similarity, euclidean_distance


class BruteForceIndex(BaseIndex):

    def __init__(self) -> None:

        self._vectors: dict[str, list[float]] = {}

    # ==========================================================
    # Build
    # ==========================================================

    def build(self, vectors: dict[str, list[float]]) -> None:

        self._vectors = dict(vectors)

    # ==========================================================
    # CRUD
    # ==========================================================

    def add(self, record_id: str, vector: list[float]) -> None:

        self._vectors[record_id] = vector

    def remove(self, record_id: str) -> None:

        if record_id in self._vectors:

            del self._vectors[record_id]

    def update(self, record_id: str, vector: list[float]) -> None:

        self._vectors[record_id] = vector

    # ==========================================================
    # Search
    # ==========================================================

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        metric: str = "cosine",
        **kwargs: Any,
    ) -> list[tuple[str, float]]:

        scores: list[tuple[str, float]] = []

        for record_id, vector in self._vectors.items():

            if metric == "cosine":

                score = cosine_similarity(query_vector, vector)

            elif metric == "dot":

                score = dot_similarity(query_vector, vector)

            elif metric == "euclidean":

                score = euclidean_distance(query_vector, vector)

            else:

                raise ValueError(f"Unknown metric: {metric}")

            scores.append((record_id, score))

        if metric == "euclidean":

            scores.sort(key=lambda x: x[1])

        else:

            scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    # ==========================================================
    # Utility
    # ==========================================================

    def clear(self) -> None:

        self._vectors.clear()

    def size(self) -> int:

        return len(self._vectors)

    # ==========================================================
    # Persistence
    # ==========================================================

    def save(self, path: str) -> None:

        import pickle

        with open(path, "wb") as f:

            pickle.dump(self._vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:

        import pickle

        with open(path, "rb") as f:

            self._vectors = pickle.load(f)

    # ==========================================================
    # Magic Methods
    # ==========================================================

    def __len__(self) -> int:

        return len(self._vectors)

    def __contains__(self, record_id: str) -> bool:

        return record_id in self._vectors

    def __repr__(self) -> str:

        return f"BruteForceIndex(" f"vectors={len(self)}" f")"
