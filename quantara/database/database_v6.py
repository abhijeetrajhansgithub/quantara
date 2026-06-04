import os
import json
import pickle
import dataclasses

from typing import (
    Any,
    Optional,
    Dict,
    List,
    Tuple,
    Literal
)
from dataclasses import dataclass

from quantara.utils.utils import (
    get_uuid,
    cosine_similarity,
    dot_similarity,
    euclidean_distance
)

from quantara.errors.errors import (
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    IndexNotFoundError,
    EmbeddingDimensionError
)

from quantara.indexes.bruteforce import BruteForceIndex
from quantara.indexes.registry import INDEX_REGISTRY

from quantara.database.models import Record, Config


class Database:

    INDEXES = INDEX_REGISTRY.get_indexes_dict()
    INDEXES["default"] = BruteForceIndex

    def __init__(
        self,
        db_name: str,
        dimensions: int | None = None,
        auto_dim: bool = True,
        auto_persist: bool = True,
        **kwargs
    ):
        self.db_name = db_name
        self.dimensions = dimensions
        self.auto_dim = auto_dim
        self.auto_persist = auto_persist

        self.provided_index_collection = kwargs.get("index_collection", "default")
        self.provided_index_path = kwargs.get("index_path", None)

        if not self.db_name.endswith(".db"):
            self.db_name += ".db"

        self._path = os.path.join(os.getcwd(), self.db_name)

        self._records: dict[str, dict[str, Record | Config]] = self._default_records()
        self._indexes: dict[str, Any] = {}

        self._load_doc()

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    def _validate_collection(self, collection: str) -> None:
        """Raise CollectionNotFoundError if the given collection does not exist in _records."""
        if collection not in self._records:
            raise CollectionNotFoundError(
                f"Collection '{collection}' not found."
            )

    def _validate_doc(self, collection: str, id: str) -> None:
        """Raise CollectionNotFoundError or IndexNotFoundError if the collection or record id is missing."""
        self._validate_collection(collection)

        if id not in self._records[collection]:
            raise IndexNotFoundError(
                f"Id '{id}' not found in collection '{collection}'."
            )

    def _validate_index_algo(self, index: str) -> None:
        """Raise ValueError if the given index type string is not present in the index registry."""
        if index not in self.INDEXES:
            raise ValueError(
                f"Index '{index}' not found in registry. "
                f"Available indexes: {list(self.INDEXES.keys())}"
            )

    # ==========================================================
    # Config Manager
    # ==========================================================

    def _get_config(self) -> dict[str, Any]:
        """Return the entire _config dict from _records."""
        return self._records["_config"]

    def _set_config(self, config: dict[str, Any]) -> None:
        """Replace the entire _config dict in _records with the given dict."""
        self._records["_config"] = config

    def _update_config(self, key: str, value: Any) -> None:
        """Set a single key-value pair inside _config."""
        self._records["_config"][key] = value

    def _get_config_value(self, key: str) -> Any:
        """Return the value for a given key from _config."""
        return self._records["_config"][key]

    def _has_config_value(self, key: str) -> bool:
        """Return True if the given key exists in _config."""
        return key in self._records["_config"]

    def _delete_config_value(self, key: str) -> None:
        """Delete a key from _config."""
        del self._records["_config"][key]

    def _clear_config(self) -> None:
        """Reset _config to an empty dict."""
        self._records["_config"] = {}

    # ==========================================================
    # Public API for Config
    # ==========================================================

    def get_config(self) -> dict[str, Any]:
        """Return the full database configuration dictionary."""
        return self._get_config()

    def set_config(self, config: dict[str, Any]) -> None:
        """Replace the full database configuration with the given dict."""
        self._set_config(config)

    def update_config(self, key: str, value: Any) -> None:
        """Update a single key inside the database configuration."""
        self._update_config(key, value)

    def get_config_value(self, key: str) -> Any:
        """Return the value of a single key from the database configuration."""
        return self._get_config_value(key)

    def has_config_value(self, key: str) -> bool:
        """Return True if the given key exists in the database configuration."""
        return self._has_config_value(key)

    def delete_config_value(self, key: str) -> None:
        """Remove a key from the database configuration."""
        self._delete_config_value(key)

    def clear_config(self) -> None:
        """Wipe the entire database configuration dictionary."""
        self._clear_config()

    # ==========================================================
    # Collections
    # ==========================================================

    def create_collection(
        self,
        collection: str,
        index: Literal["default", "bruteforce"] | str = "default"
    ) -> None:
        """
        Create a new named collection with the specified index type.
        Does nothing if the collection already exists.
        Persists the chosen index type in _config so it survives a reload.
        """
        self._validate_index_algo(index)

        if not isinstance(collection, str):
            raise TypeError("Collection name must be a string.")

        if collection in self._records:
            return

        self._records[collection] = {}
        self._indexes[collection] = self.INDEXES[index]()

        self._records["_config"].setdefault("_collection_indexes", {})[collection] = index

        if self.auto_persist:
            self.persist_doc()

    def delete_collection(self, collection: str) -> None:
        """
        Delete a named collection and its associated in-memory index.
        Raises ValueError if attempting to delete the reserved _config collection,
        and RuntimeError if attempting to delete the default collection.
        """
        if collection == "_config":
            raise ValueError("Cannot delete config collection.")

        if collection == "default":
            raise RuntimeError("Default collection cannot be deleted.")

        self._validate_collection(collection)

        del self._records[collection]
        del self._indexes[collection]

        self._records["_config"].get("_collection_indexes", {}).pop(collection, None)

        if self.auto_persist:
            self.persist_doc()

    def list_collections(self) -> list[str]:
        """Return a list of all user-facing collection names, excluding _config."""
        return [k for k in self._records if k != "_config"]

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """
        Rename an existing collection.
        Moves both the records and the in-memory index to the new key,
        and updates the stored index type in _config.
        """
        self._validate_collection(old_name)

        if new_name in self._records:
            raise CollectionAlreadyExistsError(
                f"Collection '{new_name}' already exists."
            )

        self._records[new_name] = self._records[old_name]
        del self._records[old_name]

        self._indexes[new_name] = self._indexes[old_name]
        del self._indexes[old_name]

        col_indexes = self._records["_config"].get("_collection_indexes", {})
        if old_name in col_indexes:
            col_indexes[new_name] = col_indexes.pop(old_name)

        if self.auto_persist:
            self.persist_doc()

    def clone_collection(self, source: str, target: str) -> None:
        """
        Shallow-copy a collection and its index into a new collection named target.
        Carries the source index type over to the clone in _config.
        """
        self._validate_collection(source)

        if target in self._records:
            raise CollectionAlreadyExistsError(
                f"Collection '{target}' already exists."
            )

        self._records[target] = self._records[source].copy()
        self._indexes[target] = self._indexes[source].copy()

        col_indexes = self._records["_config"].get("_collection_indexes", {})
        if source in col_indexes:
            col_indexes[target] = col_indexes[source]

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
        **kwargs
    ) -> str:
        """
        Insert a single record into the given collection.
        Auto-infers dimensions from the first vector if auto_dim is True.
        Returns the newly assigned record UUID.
        """
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
            raise RuntimeError(
                f"Missing parameters: {', '.join(missing_params)}"
            )

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
            id=record_id,
            name=_name,
            vector=_vector,
            metadata=_metadata or {}
        )

        self._indexes[collection].add(record_id, _vector)

        if self.auto_persist:
            self.persist_doc()

        return record_id

    def delete_doc(self, id: str, collection: str = "default") -> None:
        """
        Remove a record by id from the given collection and its index.
        Raises IndexNotFoundError if the id does not exist.
        """
        self._validate_doc(collection, id)

        del self._records[collection][id]
        self._indexes[collection].remove(id)

        if self.auto_persist:
            self.persist_doc()

    def update_doc(
        self,
        id: str,
        collection: str = "default",
        name: Optional[str] = None,
        vector: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Update one or more fields of an existing record.
        Only updates the in-memory index when a new vector is actually provided,
        to avoid corrupting the index with a None value.
        """
        self._validate_doc(collection, id)

        if vector is not None:
            if self.dimensions is not None:
                if len(vector) != self.dimensions:
                    raise EmbeddingDimensionError(
                        f"Expected embedding dimension {self.dimensions}, "
                        f"got {len(vector)}."
                    )
            else:
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

        if vector is not None:
            self._indexes[collection].update(id, vector)

        if self.auto_persist:
            self.persist_doc()

    def get_doc(self, id: str, collection: str = "default") -> Record:
        """Return the Record object for the given id from the specified collection."""
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
        basic: bool = True
    ):
        """
        Search for the top-k nearest records to input_vector in the given collection.

        When basic=True, performs a linear scan over all records, applying optional
        metadata filters and scoring via the chosen metric (cosine, dot, euclidean).

        When basic=False, delegates to the collection's index for approximate or
        exact nearest-neighbour search, then applies metadata filters post-hoc.

        Returns a list of (record_id, score) tuples, or
        (record_id, score, name, metadata) tuples when return_text_outputs=True.
        """
        if basic:
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
                    valid = all(
                        record.metadata.get(k) == v for k, v in filters.items()
                    )
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

            scores.sort(key=lambda x: x[1], reverse=(metric != "euclidean"))
            return scores[:top_k]

        else:
            results = self._indexes[collection].search(
                query_vector=input_vector,
                top_k=top_k,
                metric=metric
            )

            final_results = []

            for record_id, score in results:
                record = self._records[collection][record_id]

                if filters is not None:
                    valid = all(
                        record.metadata.get(k) == v for k, v in filters.items()
                    )
                    if not valid:
                        continue

                if return_text_outputs:
                    final_results.append(
                        (record_id, score, record.name, record.metadata)
                    )
                else:
                    final_results.append((record_id, score))

            return final_results

    # ==========================================================
    # Utility
    # ==========================================================

    def list_docs(self, collection: str = "default") -> list[str]:
        """Return a list of all record ids in the given collection."""
        self._validate_collection(collection)
        return list(self._records[collection].keys())

    def clear(self, collection: Optional[str] = None) -> None:
        """
        Clear all records and reset indexes.
        If collection is None, resets the entire database to its default state.
        If a collection name is given, clears only that collection's records and index.
        """
        if collection is None:
            self._records = self._default_records()
            self._indexes = {"default": self.INDEXES["default"]()}
        else:
            self._validate_collection(collection)
            self._records[collection].clear()
            self._indexes[collection].clear()

        if self.auto_persist:
            self.persist_doc()

    # ==========================================================
    # Persistence
    # ==========================================================

    def persist_doc(self) -> None:
        """Serialise the entire _records dict to disk using pickle."""
        with open(self._path, "wb") as f:
            pickle.dump(self._records, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _default_records(self) -> dict:
        """
        Return the baseline _records structure used when creating a new database
        or resetting an existing one. Contains a _config entry and an empty
        default collection.
        """
        return {
            "_config": {
                "dimensions": self.dimensions,
                "auto_dim": self.auto_dim,
                "auto_persist": self.auto_persist,
                "_collection_indexes": {
                    "default": "default"
                }
            },
            "default": {}
        }

    def _rebuild_indexes(self) -> None:
        """
        Reconstruct all in-memory indexes from the current _records state.

        Reads the index type per collection from _config['_collection_indexes'],
        falling back to "default" for any collection not listed or whose stored
        index type no longer exists in the registry. Wipes _indexes first to
        prevent vectors from being added twice.
        """
        self._indexes = {}

        col_indexes: dict[str, str] = (
            self._records.get("_config", {}).get("_collection_indexes", {})
        )

        for collection_name, collection in self._records.items():
            if collection_name == "_config":
                continue

            index_type = col_indexes.get(collection_name, "default")

            if index_type not in self.INDEXES:
                index_type = "default"

            self._indexes[collection_name] = self.INDEXES[index_type]()

            for record_id, record in collection.items():
                self._indexes[collection_name].add(record_id, record.vector)

    def _validate_index_schema(self, data: dict) -> None:
        """
        Validate that every record in the loaded data is a proper Record dataclass
        instance with the correct field types and consistent embedding dimensions.

        Checks performed per record:
          - Must be an instance of Record (not a raw dict from a corrupt pickle).
          - Must have non-empty string 'id' and 'name' fields.
          - Must have a non-empty list 'vector' of floats or ints.
          - Must have a dict 'metadata' field.
          - Vector dimension must match self.dimensions if already set, or must be
            consistent across all records in the same collection.

        Raises:
            TypeError:  if a record is not a Record instance.
            ValueError: if a required field is missing, has the wrong type,
                        or if vector dimensions are inconsistent.
        """
        config_dimensions: Optional[int] = (
            data.get("_config", {}).get("dimensions", None)
        )

        for collection_name, collection in data.items():
            if collection_name == "_config":
                continue

            if not isinstance(collection, dict):
                raise TypeError(
                    f"Collection '{collection_name}' must be a dict, "
                    f"got {type(collection).__name__}."
                )

            observed_dim: Optional[int] = None

            for record_id, record in collection.items():
                # Each stored value must be a Record dataclass, not a raw dict.
                if not isinstance(record, Record):
                    raise TypeError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"must be a Record instance, got {type(record).__name__}. "
                        f"The database file may be corrupt or was exported as JSON "
                        f"without being re-imported correctly."
                    )

                # id
                if not isinstance(record.id, str) or not record.id:
                    raise ValueError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"has an invalid 'id': {record.id!r}."
                    )

                # name
                if not isinstance(record.name, str) or not record.name:
                    raise ValueError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"has an invalid 'name': {record.name!r}."
                    )

                # vector
                if (
                    not isinstance(record.vector, (list, tuple))
                    or len(record.vector) == 0
                ):
                    raise ValueError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"has an invalid or empty 'vector'."
                    )

                if not all(isinstance(v, (int, float)) for v in record.vector):
                    raise ValueError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"contains non-numeric values in 'vector'."
                    )

                # metadata
                if not isinstance(record.metadata, dict):
                    raise ValueError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"has an invalid 'metadata': expected dict, "
                        f"got {type(record.metadata).__name__}."
                    )

                # Dimension consistency — first check against the config value,
                # then against the first vector seen in this collection.
                record_dim = len(record.vector)

                if config_dimensions is not None and record_dim != config_dimensions:
                    raise ValueError(
                        f"Record '{record_id}' in collection '{collection_name}' "
                        f"has vector dimension {record_dim}, but config declares "
                        f"dimensions={config_dimensions}."
                    )

                if observed_dim is None:
                    observed_dim = record_dim
                elif record_dim != observed_dim:
                    raise ValueError(
                        f"Inconsistent vector dimensions in collection "
                        f"'{collection_name}': expected {observed_dim}, "
                        f"got {record_dim} for record '{record_id}'."
                    )

    def _load_doc(self) -> None:
        """
        Load the database from disk.

        Behaviour:
          - If no file exists at _path, initialises a fresh default database
            and returns.
          - Otherwise deserialises _records from the pickle file and validates
            the schema via _validate_index_schema.
          - If a provided_index_path ending in '.index' is given, loads that
            external index file for the provided_index_collection instead of
            rebuilding it from scratch.
          - If no external index path is given, calls _rebuild_indexes to
            reconstruct all in-memory indexes from _records.
          - On EOFError (empty/truncated file), silently resets to defaults.
          - On any other exception, resets to defaults and re-raises as
            RuntimeError to surface the underlying cause to the caller.
        """
        if not os.path.exists(self._path):
            self._records = self._default_records()
            self._rebuild_indexes()
            return

        try:
            with open(self._path, "rb") as f:
                loaded: dict[str, dict[str, Record]] = pickle.load(f)

            # Validate the deserialized data before accepting it.
            self._validate_index_schema(loaded)

            # Accept the validated data.
            self._records = loaded

            # Sync instance-level dimension from config if not already set.
            stored_dim = self._records.get("_config", {}).get("dimensions", None)
            if self.dimensions is None and stored_dim is not None:
                self.dimensions = stored_dim

            if (
                self.provided_index_path
                and self.provided_index_collection
                and self.provided_index_path.endswith(".index")
            ):
                # Rebuild all indexes first so every collection has an index
                # object, then overwrite just the requested collection's index
                # with the pre-saved file.
                self._rebuild_indexes()
                self._load_index(
                    collection=self.provided_index_collection,
                    path=self.provided_index_path
                )
            else:
                self._rebuild_indexes()

        except EOFError:
            self._records = self._default_records()
            self._rebuild_indexes()

        except (TypeError, ValueError) as e:
            # Schema validation failure — reset to defaults and surface the error.
            self._records = self._default_records()
            self._rebuild_indexes()
            raise RuntimeError(
                f"Database schema validation failed for '{self._path}': {e}"
            ) from e

        except Exception as e:
            self._records = self._default_records()
            self._rebuild_indexes()
            raise RuntimeError(
                f"Failed to load database from '{self._path}': {e}"
            ) from e

    # ==========================================================
    # Statistics
    # ==========================================================

    def collection_stats(self, collection: str = "default") -> dict[str, Any]:
        """
        Return statistics for a single collection: document count,
        average vector dimension, and the index type in use.
        """
        self._validate_collection(collection)

        dimensions = [
            len(record.vector)
            for record in self._records[collection].values()
        ]

        avg_dim = sum(dimensions) / len(dimensions) if dimensions else 0

        col_indexes = self._records["_config"].get("_collection_indexes", {})

        return {
            "collection": collection,
            "documents": len(dimensions),
            "average_dimension": avg_dim,
            "index_type": col_indexes.get(collection, "default")
        }

    def stats(self) -> dict[str, Any]:
        """
        Return aggregate statistics for the entire database: total collection
        count, total document count, average vector dimension across all
        collections, and the file path of the database on disk.
        """
        all_dims = []

        for collection_name, collection in self._records.items():
            if collection_name == "_config":
                continue
            for record in collection.values():
                all_dims.append(len(record.vector))

        avg_dim = sum(all_dims) / len(all_dims) if all_dims else 0

        return {
            "collections": len(self._records) - 1,
            "documents": len(all_dims),
            "average_dimension": avg_dim,
            "database_path": self._path
        }

    # ==========================================================
    # Magic Methods
    # ==========================================================

    def __len__(self) -> int:
        """Return the total number of records across all collections."""
        return sum(
            len(collection)
            for name, collection in self._records.items()
            if name != "_config"
        )

    def __contains__(self, id: str) -> bool:
        """Return True if the given record id exists in any collection."""
        return any(
            id in collection
            for name, collection in self._records.items()
            if name != "_config"
        )

    def __repr__(self) -> str:
        """Return a concise string representation of the Database instance."""
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
        Validate that all records in the given raw dict (as loaded from JSON)
        are plain dicts with the required keys: id, vector, and metadata.
        Used before importing from a JSON export.
        """
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
        Import records from a JSON file produced by export_to_json.
        Merges collections from the file into the current database,
        restoring the correct index type per collection from _config.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self._validate_consistent_format(data)

        if "_config" in data:
            self.dimensions = data["_config"].get("dimensions", self.dimensions)
            self.auto_dim = data["_config"].get("auto_dim", self.auto_dim)
            self.auto_persist = data["_config"].get("auto_persist", self.auto_persist)
            self._records["_config"] = data["_config"]
            self._records["_config"].setdefault(
                "_collection_indexes", {"default": "default"}
            )

        col_indexes = self._records["_config"].get("_collection_indexes", {})

        for collection_name, collection in data.items():
            if collection_name == "_config":
                continue

            self._records.setdefault(collection_name, {})

            index_type = col_indexes.get(collection_name, "default")
            if index_type not in self.INDEXES:
                index_type = "default"

            self._indexes.setdefault(
                collection_name, self.INDEXES[index_type]()
            )

            for record_id, record_data in collection.items():
                self._records[collection_name][record_id] = Record(
                    id=record_data["id"],
                    name=record_data["name"],
                    vector=record_data["vector"],
                    metadata=record_data["metadata"]
                )
                self._indexes[collection_name].add(record_id, record_data["vector"])

        if self.auto_persist:
            self.persist_doc()

    def export_to_json(self, json_path: str) -> None:
        """
        Serialise the entire database to a human-readable JSON file.
        Raises FileExistsError if the target path already exists.
        Record dataclasses are converted to dicts via dataclasses.asdict.
        """
        if os.path.exists(json_path):
            raise FileExistsError(f"JSON file already exists: {json_path}")

        with open(json_path, 'w') as f:
            json.dump(
                self._records,
                f,
                indent=4,
                default=lambda o: (
                    dataclasses.asdict(o) if dataclasses.is_dataclass(o) else o
                )
            )

    # ==========================================================
    # Batch Management
    # ==========================================================

    def batch_insert_docs(
        self,
        objects: List[Tuple[str, list[float], dict[str, Any]]],
        collection: str = "default"
    ) -> List[str]:
        """
        Insert multiple records into a collection in one call.
        Accepts a list of (name, vector, metadata) tuples.
        Validates dimensions for each record and persists only once at the end,
        rather than once per record, to avoid repeated disk writes.
        Returns a list of UUIDs in the same order as the input list.
        """
        if not objects:
            raise ValueError("objects must not be empty.")

        self._validate_collection(collection)

        record_ids = []

        for name, vector, metadata in objects:
            record_id = get_uuid()

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
                id=record_id,
                name=name,
                vector=vector,
                metadata=metadata or {}
            )

            self._indexes[collection].add(record_id, vector)
            record_ids.append(record_id)

        if self.auto_persist:
            self.persist_doc()

        return record_ids

    # ==========================================================
    # Index Import / Export
    # ==========================================================

    def save_index(self, collection: str = "default", path: str = None) -> str:
        """
        Persist a collection's in-memory index to a binary .index file.
        Defaults to <db_name>_<collection>.index alongside the database file.
        Returns the path the index was saved to.
        """
        self._validate_collection(collection)

        if path is None:
            base = self._path.replace(".db", "")
            path = f"{base}_{collection}.index"

        self._indexes[collection].save(path)
        return path

    def _load_index(self, collection: str = "default", path: str = None) -> None:
        """
        Replace a collection's current in-memory index with one loaded from disk.
        Defaults to <db_name>_<collection>.index alongside the database file.
        Raises FileNotFoundError if the index file does not exist.
        """
        self._validate_collection(collection)

        if path is None:
            base = self._path.replace(".db", "")
            path = f"{base}_{collection}.index"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")

        self._indexes[collection].load(path)