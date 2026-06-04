import os
import pickle

from typing import Any, Optional
from dataclasses import dataclass

from quantara.utils.utils import (
    get_uuid,
    cosine_similarity
)


_PARENT_DIR_ = os.getcwd()


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
        auto_persist: bool = True
    ):

        self.db_name = db_name
        self.auto_persist = auto_persist

        if not self.db_name.endswith(".db"):
            self.db_name += ".db"

        self._path = os.path.join(
            _PARENT_DIR_,
            self.db_name
        )

        self._records: dict[str, Record] = {}

        self.load_doc()

    def insert_doc(
        self,
        name: Optional[str] = None,
        vector: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> str:

        _name = (
            name
            if name is not None
            else kwargs.get("name")
        )

        _vector = (
            vector
            if vector is not None
            else kwargs.get("vector")
        )

        _metadata = (
            metadata
            if metadata is not None
            else kwargs.get("metadata")
        )

        missing_params = []

        if _name is None:
            missing_params.append("name")

        if _vector is None:
            missing_params.append("vector")

        if missing_params:
            raise RuntimeError(
                f"Missing parameters: {', '.join(missing_params)}"
            )

        record_id = get_uuid()

        self._records[record_id] = Record(
            id=record_id,
            name=_name,
            vector=_vector,
            metadata=_metadata or {}
        )

        if self.auto_persist:
            self.persist_doc()

        return record_id

    def delete_doc(
        self,
        id: str
    ) -> None:

        if id not in self._records:
            raise KeyError(
                f"Key '{id}' not found."
            )

        del self._records[id]

        if self.auto_persist:
            self.persist_doc()

    def update_doc(
        self,
        id: str,
        name: Optional[str] = None,
        vector: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:

        if id not in self._records:
            raise KeyError(
                f"Key '{id}' not found."
            )

        record = self._records[id]

        if name is not None:
            record.name = name

        if vector is not None:
            record.vector = vector

        if metadata is not None:
            record.metadata = metadata

        if self.auto_persist:
            self.persist_doc()

    def get_doc(
        self,
        id: str
    ) -> Record:

        if id not in self._records:
            raise KeyError(
                f"Key '{id}' not found."
            )

        return self._records[id]

    def search_doc(
        self,
        input_vector: list[float],
        top_k: int = 3,
        return_text_outputs: bool = False
    ):

        scores = []

        for record_id, record in self._records.items():

            score = cosine_similarity(
                input_vector,
                record.vector
            )

            if return_text_outputs:

                scores.append(
                    (
                        record_id,
                        score,
                        record.name,
                        record.metadata
                    )
                )

            else:

                scores.append(
                    (
                        record_id,
                        score
                    )
                )

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scores[:top_k]

    def list_docs(
        self
    ) -> list[str]:

        return list(
            self._records.keys()
        )

    def persist_doc(
        self
    ) -> None:

        try:

            with open(
                self._path,
                "wb"
            ) as f:

                pickle.dump(
                    self._records,
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL
                )

        except Exception as e:

            print(
                f"[ERROR] Persist failed: {str(e)}"
            )

    def load_doc(
        self
    ) -> None:

        try:

            if not os.path.exists(
                self._path
            ):
                self._records = {}
                return

            with open(
                self._path,
                "rb"
            ) as f:

                self._records = pickle.load(
                    f
                )

        except EOFError:

            self._records = {}

        except Exception as e:

            print(
                f"[ERROR] Load failed: {str(e)}"
            )

            self._records = {}

    def clear(
        self
    ) -> None:

        self._records.clear()

        if self.auto_persist:
            self.persist_doc()

    def stats(
        self
    ) -> dict[str, Any]:

        dimensions = []

        for record in self._records.values():

            dimensions.append(
                len(record.vector)
            )

        avg_dim = (
            sum(dimensions)
            / len(dimensions)
            if dimensions
            else 0
        )

        return {
            "documents": len(
                self._records
            ),
            "average_dimension": avg_dim,
            "database_path": self._path
        }

    def __len__(
        self
    ) -> int:

        return len(
            self._records
        )

    def __contains__(
        self,
        id: str
    ) -> bool:

        return id in self._records

    def __repr__(
        self
    ) -> str:

        return (
            f"Database("
            f"name='{self.db_name}', "
            f"documents={len(self)}"
            f")"
        )