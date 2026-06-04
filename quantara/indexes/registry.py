from typing import Type
from inspect import isclass

from quantara.indexes.base import BaseIndex
from quantara.indexes.bruteforce import BruteForceIndex


class IndexRegistry:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(
                cls
            )

            cls._instance._registry = {}

        return cls._instance

    def register(
        self,
        name: str,
        index_cls: Type[BaseIndex]
    ) -> None:

        if not isclass(index_cls):

            raise TypeError(
                "index_cls must be a class."
            )

        if not issubclass(
            index_cls,
            BaseIndex
        ):

            raise TypeError(
                "Index must inherit from BaseIndex."
            )

        self._registry[
            name.lower()
        ] = index_cls

    def unregister(
        self,
        name: str
    ) -> None:

        self._registry.pop(
            name.lower(),
            None
        )

    def get(
        self,
        name: str
    ) -> Type[BaseIndex]:

        name = name.lower()

        if name not in self._registry:

            raise KeyError(
                f"Index '{name}' is not registered."
            )

        return self._registry[
            name
        ]

    def exists(
        self,
        name: str
    ) -> bool:

        return (
            name.lower()
            in self._registry
        )

    def list_indexes(
        self
    ) -> list[str]:

        return sorted(
            self._registry.keys()
        )
    
    def get_indexes_dict(self) -> dict[str, Type[BaseIndex]]:
        return self._registry.copy()

    def clear(
        self
    ) -> None:

        self._registry.clear()

    def __len__(
        self
    ) -> int:

        return len(
            self._registry
        )

    def __contains__(
        self,
        name: str
    ) -> bool:

        return (
            name.lower()
            in self._registry
        )

    def __repr__(
        self
    ) -> str:

        return (
            f"IndexRegistry("
            f"indexes={len(self)}"
            f")"
        )


# ==========================================================
# Interfaces
# ==========================================================

INDEX_REGISTRY = IndexRegistry()


# ==========================================================
# Public API
# ==========================================================

def register_index(
    name: str,
    index_cls: Type[BaseIndex]
) -> None:

    INDEX_REGISTRY.register(
        name,
        index_cls
    )


def unregister_index(
    name: str
) -> None:

    INDEX_REGISTRY.unregister(
        name
    )


def get_index(
    name: str
) -> Type[BaseIndex]:

    return INDEX_REGISTRY.get(
        name
    )


def list_indexes(
) -> list[str]:

    return INDEX_REGISTRY.list_indexes()

