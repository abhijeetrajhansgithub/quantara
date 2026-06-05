# quantara/__init__.py

from quantara.database.database import Database

from quantara.indexes.base import BaseIndex

from quantara.embed.ollama_embed import OllamaEmbedding

from quantara.indexes.registry import (
    register_index,
    unregister_index,
    get_index,
    list_indexes
)

from quantara.indexes.builtin import (
    load_builtin_indexes
)

load_builtin_indexes()

__all__ = [
    "Database",
    "BaseIndex",
    "OllamaEmbedding",
    "register_index",
    "unregister_index",
    "get_index",
    "list_indexes"
]