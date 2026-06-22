import os
import pickle

from typing import Any, Optional, Dict
from dataclasses import dataclass

from quantara.utils.utils import get_uuid, cosine_similarity

from quantara.errors.errors import (
    CollectionNotFoundError,
    IndexNotFoundError,
    EmbeddingDimensionError,
)


@dataclass(slots=True)
class Record:
    id: str
    name: str
    vector: list[float]
    metadata: dict[str, Any]


class Database:

    def __init__(
        self,
        db_name: str,
        dimensions: int | None = None,
        auto_dim: bool = True,
        auto_persist: bool = True,
    ):

        self.db_name = db_name
        self.dimensions = dimensions
        self.auto_dim = auto_dim
        self.auto_persist = auto_persist

        if not self.db_name.endswith(".db"):
            self.db_name += ".db"

        # FIX 3: Resolve path at instantiation time, not at module import time.
        # Using a module-level os.getcwd() means the path is frozen to wherever
        # Python was when the module was first imported, which breaks if the
        # process later calls os.chdir().
        self._path = os.path.join(os.getcwd(), self.db_name)

        self._records: dict[str, dict[str, Record]] = {"default": {}}

        self._load_doc()

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    def _validate_collection(self, collection: str) -> None:

        if collection not in self._records:
            raise CollectionNotFoundError(f"Collection '{collection}' not found.")

    def _validate_doc(self, collection: str, id: str) -> None:

        self._validate_collection(collection)

        if id not in self._records[collection]:
            raise IndexNotFoundError(
                f"Id '{id}' not found in collection '{collection}'."
            )

    # ==========================================================
    # Collections
    # ==========================================================

    def create_collection(self, collection: str) -> None:

        if not isinstance(collection, str):
            raise TypeError("Collection name must be a string.")

        if collection in self._records:
            return

        self._records[collection] = {}

        if self.auto_persist:
            self.persist_doc()

    def delete_collection(self, collection: str) -> None:

        if collection == "default":
            raise RuntimeError("Default collection cannot be deleted.")

        self._validate_collection(collection)

        del self._records[collection]

        if self.auto_persist:
            self.persist_doc()

    def list_collections(self) -> list[str]:

        return list(self._records.keys())

    # ==========================================================
    # CRUD
    # ==========================================================

    def insert_doc(
        self,
        name: Optional[str] = None,
        vector: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
        collection: str = "default",
        **kwargs,
    ) -> str:

        self._validate_collection(collection)

        _name = name if name is not None else kwargs.get("name")
        _vector = vector if vector is not None else kwargs.get("vector")
        _metadata = metadata if metadata is not None else kwargs.get("metadata")

        missing_params = []

        if _name is None:
            missing_params.append("name")

        if _vector is None:
            missing_params.append("vector")

        if missing_params:
            raise RuntimeError(f"Missing parameters: {', '.join(missing_params)}")

        if self.dimensions is None:
            if self.auto_dim:
                self.dimensions = len(_vector)
            else:
                raise EmbeddingDimensionError(
                    "Parameter `dimensions` cannot be None or undefined when "
                    "inserting into the vector database."
                )

        if len(_vector) != self.dimensions:
            raise EmbeddingDimensionError(
                f"Expected embedding dimension {self.dimensions}, "
                f"got {len(_vector)}."
            )

        record_id = get_uuid()

        self._records[collection][record_id] = Record(
            id=record_id, name=_name, vector=_vector, metadata=_metadata or {}
        )

        if self.auto_persist:
            self.persist_doc()

        return record_id

    def delete_doc(self, id: str, collection: str = "default") -> None:

        self._validate_doc(collection, id)

        del self._records[collection][id]

        if self.auto_persist:
            self.persist_doc()

    def update_doc(
        self,
        id: str,
        collection: str = "default",
        name: Optional[str] = None,
        vector: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:

        # FIX 1: Validate that the document exists BEFORE checking dimensions.
        # Previously, a dimension mismatch on a non-existent record would raise
        # EmbeddingDimensionError instead of the correct IndexNotFoundError,
        # hiding the real problem from the caller.
        self._validate_doc(collection, id)

        if vector is not None:
            if self.dimensions is not None:
                if len(vector) != self.dimensions:
                    raise EmbeddingDimensionError(
                        f"Expected embedding dimension {self.dimensions}, "
                        f"got {len(vector)}."
                    )
            else:
                # FIX 2: When auto_dim=True and self.dimensions is still None
                # (e.g. update_doc is called before any insert_doc), infer and
                # lock the dimension from this vector, consistent with insert_doc.
                if self.auto_dim:
                    self.dimensions = len(vector)

        record = self._records[collection][id]

        if name is not None:
            record.name = name

        if vector is not None:
            record.vector = vector

        if metadata is not None:
            record.metadata = metadata

        if self.auto_persist:
            self.persist_doc()

    def get_doc(self, id: str, collection: str = "default") -> Record:

        self._validate_doc(collection, id)

        return self._records[collection][id]

    # ==========================================================
    # Search
    # ==========================================================

    def search_doc(
        self,
        input_vector: list[float],
        filters: Dict[str, Any],
        top_k: int = 3,
        return_text_outputs: bool = False,
        collection: str = "default",
    ):

        if self.dimensions is not None:
            if len(input_vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Expected embedding dimension {self.dimensions}, "
                    f"got {len(input_vector)}."
                )

        self._validate_collection(collection)

        scores = []

        for record_id, record in self._records[collection].items():

            if filters is not None:
                valid = True

                for key, value in filters.items():
                    if record.metadata.get(key) != value:
                        valid = False

                        break

                if not valid:
                    continue

            score = cosine_similarity(input_vector, record.vector)

            if return_text_outputs:
                scores.append((record_id, score, record.name, record.metadata))
            else:
                scores.append((record_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    # ==========================================================
    # Utility
    # ==========================================================

    def list_docs(self, collection: str = "default") -> list[str]:

        self._validate_collection(collection)

        return list(self._records[collection].keys())

    def clear(self, collection: Optional[str] = None) -> None:

        if collection is None:
            self._records = {"default": {}}
        else:
            self._validate_collection(collection)
            self._records[collection].clear()

        if self.auto_persist:
            self.persist_doc()

    # ==========================================================
    # Persistence
    # ==========================================================

    def persist_doc(self) -> None:

        # FIX 4: Raise on failure instead of silently swallowing errors.
        # The old code printed a message and returned None, giving the caller
        # no way to detect that the write failed. Callers that depend on
        # durability (e.g. auto_persist=True) need to know when it breaks.
        with open(self._path, "wb") as f:
            pickle.dump(self._records, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_doc(self) -> None:

        if not os.path.exists(self._path):
            self._records = {"default": {}}
            return

        try:
            with open(self._path, "rb") as f:
                self._records = pickle.load(f)

        except EOFError:
            # Empty / truncated file — start fresh.
            self._records = {"default": {}}

        except Exception as e:
            # Corrupted file — start fresh and re-raise so the caller is aware.
            self._records = {"default": {}}
            raise RuntimeError(
                f"Failed to load database from '{self._path}': {e}"
            ) from e

    # ==========================================================
    # Statistics
    # ==========================================================

    def stats(self) -> dict[str, Any]:

        total_docs = 0
        dimensions = []

        for collection in self._records.values():
            for record in collection.values():
                total_docs += 1
                dimensions.append(len(record.vector))

        avg_dim = sum(dimensions) / len(dimensions) if dimensions else 0

        return {
            "collections": len(self._records),
            "documents": total_docs,
            "average_dimension": avg_dim,
            "database_path": self._path,
        }

    # ==========================================================
    # Magic Methods
    # ==========================================================

    def __len__(self) -> int:
        return sum(len(collection) for collection in self._records.values())

    def __contains__(self, id: str) -> bool:
        return any(id in collection for collection in self._records.values())

    def __repr__(self) -> str:
        return (
            f"Database("
            f"name='{self.db_name}', "
            f"collections={len(self._records)}, "
            f"documents={len(self)}"
            f")"
        )
