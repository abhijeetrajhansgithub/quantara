from abc import ABC, abstractmethod
from typing import Any


class BaseIndex(ABC):

    @abstractmethod
    def build(self, vectors: dict[str, list[float]]) -> None:
        """
        Build the index from a collection
        of vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, record_id: str, vector: list[float]) -> None:
        """
        Add a new vector to the index.
        """
        raise NotImplementedError

    @abstractmethod
    def remove(self, record_id: str) -> None:
        """
        Remove a vector from the index.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, record_id: str, vector: list[float]) -> None:
        """
        Update an existing vector.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int = 3, **kwargs: Any
    ) -> list[tuple[str, float]]:
        """
        Search for nearest neighbors.

        Returns:
            [
                (record_id, score),
                ...
            ]
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all vectors
        from the index.
        """
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        """
        Number of indexed vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist index to disk.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load index from disk.
        """
        raise NotImplementedError
