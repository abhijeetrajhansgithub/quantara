import os
import json
import pickle
import dataclasses

from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass

from quantara.utils.utils import (
    get_uuid,
    cosine_similarity,
    dot_similarity,
    euclidean_distance,
)

from quantara.errors.errors import (
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    IndexNotFoundError,
    EmbeddingDimensionError,
)


@dataclass(slots=True)
class Record:
    id: str
    name: str
    vector: list[float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class Config:
    dimensions: int | None
    auto_dim: bool
    auto_persist: bool


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
        self._path = os.path.join(os.getcwd(), self.db_name)

        self._records: dict[str, dict[str, Record | Config]] = {
            "_config": {
                "dimensions": self.dimensions,
                "auto_dim": self.auto_dim,
                "auto_persist": self.auto_persist,
            },
            "default": {},
        }

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
    # Config Manager
    # ==========================================================

    def _get_config(self) -> dict[str, Any]:
        return self._records["_config"]

    def _set_config(self, config: dict[str, Any]) -> None:
        self._records["_config"] = config

    def _update_config(self, key: str, value: Any) -> None:
        self._records["_config"][key] = value

    def _get_config_value(self, key: str) -> Any:
        return self._records["_config"][key]

    def _has_config_value(self, key: str) -> bool:
        return key in self._records["_config"]

    def _delete_config_value(self, key: str) -> None:
        del self._records["_config"][key]

    def _clear_config(self) -> None:
        self._records["_config"] = {}

    # ==========================================================
    # Public API for Config
    # ==========================================================

    def get_config(self) -> dict[str, Any]:
        return self._get_config()

    def set_config(self, config: dict[str, Any]) -> None:
        self._set_config(config)

    def update_config(self, key: str, value: Any) -> None:
        self._update_config(key, value)

    def get_config_value(self, key: str) -> Any:
        return self._get_config_value(key)

    def has_config_value(self, key: str) -> bool:
        return self._has_config_value(key)

    def delete_config_value(self, key: str) -> None:
        self._delete_config_value(key)

    def clear_config(self) -> None:
        self._clear_config()

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

        assert collection != "_config", "Cannot delete config collection"

        if collection == "default":
            raise RuntimeError("Default collection cannot be deleted.")

        self._validate_collection(collection)

        del self._records[collection]

        if self.auto_persist:
            self.persist_doc()

    def list_collections(self) -> list[str]:

        # FIX 6: dict_keys - list raises TypeError; use a generator instead.
        return [k for k in self._records if k != "_config"]

    def rename_collection(self, old_name: str, new_name: str) -> None:

        self._validate_collection(old_name)

        if new_name in self._records:
            raise CollectionAlreadyExistsError(  # FIX 7: now properly imported
                f"Collection '{new_name}' already exists."
            )

        self._records[new_name] = self._records[old_name]
        del self._records[old_name]

        if self.auto_persist:
            self.persist_doc()

    def clone_collection(self, source: str, target: str) -> None:

        self._validate_collection(source)

        if target in self._records:
            raise CollectionAlreadyExistsError(  # FIX 7: now properly imported
                f"Collection '{target}' already exists."
            )

        self._records[target] = self._records[source].copy()

        if self.auto_persist:
            self.persist_doc()

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

                self._records["_config"]["dimensions"] = self.dimensions
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
        self._validate_doc(collection, id)

        if vector is not None:
            if self.dimensions is not None:
                if len(vector) != self.dimensions:
                    raise EmbeddingDimensionError(
                        f"Expected embedding dimension {self.dimensions}, "
                        f"got {len(vector)}."
                    )
            else:
                # FIX 2: When auto_dim=True and self.dimensions is still None,
                # infer and lock the dimension from this vector.
                if self.auto_dim:
                    self.dimensions = len(vector)
                    self._records["_config"]["dimensions"] = self.dimensions

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
        filters: Dict[str, Any] = None,
        top_k: int = 3,
        return_text_outputs: bool = False,
        collection: str = "default",
        metric: str = "cosine",
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

            if metric == "cosine":
                score = cosine_similarity(input_vector, record.vector)
            elif metric == "dot":
                score = dot_similarity(input_vector, record.vector)
            elif metric == "euclidean":
                score = euclidean_distance(input_vector, record.vector)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if return_text_outputs:
                scores.append((record_id, score, record.name, record.metadata))
            else:
                scores.append((record_id, score))

        if metric == "euclidean":
            scores.sort(key=lambda x: x[1], reverse=False)
        else:
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
            # FIX 8: Preserve _config so subsequent operations don't crash.
            self._records = {
                "_config": {
                    "dimensions": self.dimensions,
                    "auto_dim": self.auto_dim,
                    "auto_persist": self.auto_persist,
                },
                "default": {},
            }
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
        with open(self._path, "wb") as f:
            pickle.dump(self._records, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_doc(self) -> None:

        if not os.path.exists(self._path):
            self._records = {
                "_config": {
                    "dimensions": self.dimensions,
                    "auto_dim": self.auto_dim,
                    "auto_persist": self.auto_persist,
                },
                "default": {},
            }
            return

        try:
            with open(self._path, "rb") as f:
                self._records = pickle.load(f)

        except EOFError:
            self._records = {
                "_config": {
                    "dimensions": self.dimensions,
                    "auto_dim": self.auto_dim,
                    "auto_persist": self.auto_persist,
                },
                "default": {},
            }

        except Exception as e:
            self._records = {
                "_config": {
                    "dimensions": self.dimensions,
                    "auto_dim": self.auto_dim,
                    "auto_persist": self.auto_persist,
                },
                "default": {},
            }

            raise RuntimeError(
                f"Failed to load database from '{self._path}': {e}"
            ) from e

    # ==========================================================
    # Statistics
    # ==========================================================

    def collection_stats(self, collection: str = "default") -> dict[str, Any]:
        self._validate_collection(collection)

        total_docs = 0
        dimensions = []

        for record in self._records[collection].values():
            total_docs += 1
            dimensions.append(len(record.vector))

        avg_dim = sum(dimensions) / len(dimensions) if dimensions else 0

        return {
            "collection": collection,
            "documents": total_docs,
            "average_dimension": avg_dim,
        }

    def stats(self) -> dict[str, Any]:

        total_docs = 0
        dimensions = []

        # FIX 9: Skip _config — it holds plain values, not Record objects.
        for collection_name, collection in self._records.items():
            if collection_name == "_config":
                continue

            for record in collection.values():
                total_docs += 1
                dimensions.append(len(record.vector))

        avg_dim = sum(dimensions) / len(dimensions) if dimensions else 0

        return {
            "collections": len(self._records) - 1,  # exclude _config
            "documents": total_docs,
            "average_dimension": avg_dim,
            "database_path": self._path,
        }

    # ==========================================================
    # Magic Methods
    # ==========================================================

    def __len__(self) -> int:
        # FIX 5: Exclude _config from the document count.
        return sum(
            len(collection)
            for name, collection in self._records.items()
            if name != "_config"
        )

    def __contains__(self, id: str) -> bool:
        return any(
            id in collection
            for name, collection in self._records.items()
            if name != "_config"
        )

    def __repr__(self) -> str:
        return (
            f"Database("
            f"name='{self.db_name}', "
            f"collections={len(self._records) - 1}, "
            f"documents={len(self)}"
            f")"
        )

    # ==========================================================
    # Import / Export
    # ==========================================================

    def _validate_consistent_format(self, data: dict) -> None:
        """
        Validate that all records in the given data dict have consistent format.
        """
        # FIX: Accept the data to validate as a parameter instead of
        # always reading self._records (which was the old state, not the
        # freshly-loaded JSON).
        for collection_name, collection in data.items():
            if collection_name == "_config":
                continue
            for record in collection.values():
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid record format: {record}")
                if "id" not in record:
                    raise ValueError(f"Missing 'id' in record: {record}")
                if "vector" not in record:
                    raise ValueError(f"Missing 'vector' in record: {record}")
                if "metadata" not in record:
                    raise ValueError(f"Missing 'metadata' in record: {record}")

    def import_from_json(self, json_path: str) -> None:
        """
        Import database from JSON file.

        Args:
            json_path: Path to JSON file to import from
        """

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # FIX 3a: json was never imported — added at the top of the file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # FIX: Validate the freshly-loaded data, not self._records.
        self._validate_consistent_format(data)

        # Update config from imported data.
        if "_config" in data:
            self.dimensions = data["_config"].get("dimensions", self.dimensions)
            self.auto_dim = data["_config"].get("auto_dim", self.auto_dim)
            self.auto_persist = data["_config"].get("auto_persist", self.auto_persist)
            self._records["_config"] = data["_config"]

        # FIX: Create missing collections before writing into them.
        for collection_name, collection in data.items():
            if collection_name == "_config":
                continue

            # Ensure the collection exists in self._records.
            self._records.setdefault(collection_name, {})

            for record_id, record_data in collection.items():
                self._records[collection_name][record_id] = Record(
                    id=record_data["id"],
                    name=record_data["name"],
                    vector=record_data["vector"],
                    metadata=record_data["metadata"],
                )

        if self.auto_persist:
            self.persist_doc()

    def export_to_json(self, json_path: str) -> None:
        """
        Export database to JSON file.

        Args:
            json_path: Path to JSON file to export to
        """

        if os.path.exists(json_path):
            raise FileExistsError(f"JSON file already exists: {json_path}")

        # FIX 3b: json was never imported — added at the top of the file.
        # FIX 4b: Record dataclasses are not JSON-serializable by default;
        # use dataclasses.asdict() via a custom default serializer.
        with open(json_path, "w") as f:
            json.dump(
                self._records,
                f,
                indent=4,
                default=lambda o: (
                    dataclasses.asdict(o) if dataclasses.is_dataclass(o) else o
                ),
            )

    # ==========================================================
    # Batch Management
    # ==========================================================

    def batch_insert_docs(
        self,
        objects: List[Tuple[str, list[float], dict[str, Any]]],
        collection: str = "default",
    ) -> List[str]:
        """
        Batch insert documents into the database.

        Args:
            objects: List of (name, vector, metadata) tuples
            collection: Collection name to insert into

        Returns:
            List of inserted record IDs in the same order as input
        """

        if not objects:
            raise ValueError("objects must not be empty.")

        self._validate_collection(collection)

        record_ids = []

        for name, vector, metadata in objects:
            record_id = get_uuid()

            # Dimension check / auto-dim (mirrors insert_doc logic)
            if self.dimensions is None:
                if self.auto_dim:
                    self.dimensions = len(vector)
                    self._records["_config"]["dimensions"] = self.dimensions
                else:
                    raise EmbeddingDimensionError(
                        "Parameter `dimensions` cannot be None or undefined when "
                        "inserting into the vector database."
                    )

            if len(vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Expected embedding dimension {self.dimensions}, "
                    f"got {len(vector)}."
                )

            self._records[collection][record_id] = Record(
                id=record_id, name=name, vector=vector, metadata=metadata or {}
            )

            record_ids.append(record_id)

        # Persist once after all inserts, not once per document
        if self.auto_persist:
            self.persist_doc()

        return record_ids
