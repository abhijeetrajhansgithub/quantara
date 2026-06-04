from typing import Any, Dict, Optional
from dataclasses import dataclass
import pickle

from quantara.utils.utils import (
    get_uuid,
    cosine_similarity
)


@dataclass(slots=True)
class Record:
    id: str
    name: str
    vector: list[float]
    metadata: dict[str, Any]


class Database:

    def __init__(self, db_name: str):

        self.db_name = db_name

        if not self.db_name.endswith(".db"):
            self.db_name += ".db"

        self._records: dict[str, Record] = {}

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
                self.db_name,
                "wb"
            ) as f:

                pickle.dump(
                    self._records,
                    f
                )

            print(
                "Database persisted successfully."
            )

        except Exception as e:

            print(
                f"[ERROR] {str(e)}"
            )

    def load_doc(
        self
    ) -> None:

        try:

            with open(
                self.db_name,
                "rb"
            ) as f:

                self._records = pickle.load(
                    f
                )

            print(
                "Database loaded successfully."
            )

        except FileNotFoundError:

            print(
                f"[INFO] Database file '{self.db_name}' does not exist."
            )

        except EOFError:

            self._records = {}

            print(
                "[INFO] Database file is empty."
            )

        except Exception as e:

            print(
                f"[ERROR] {str(e)}"
            )

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