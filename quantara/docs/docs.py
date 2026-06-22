STRING = r"""
# Quantara

A lightweight, local-first vector database built in pure Python with support for collections, metadata filtering, persistence, pluggable indexes, and custom index registration.

Quantara is designed for developers, researchers, and students who want a simple yet extensible vector database that runs entirely on their local machine without requiring external services.

---

## Features

### Core Database

* Collection-based organization
* CRUD operations
* Batch document insertion
* Metadata support
* Metadata filtering during search
* Collection cloning
* Collection renaming
* Collection deletion
* Collection statistics

### Vector Search

* Cosine Similarity
* Dot Product Similarity
* Euclidean Distance
* Top-K nearest neighbor search
* Configurable search metrics

### Persistence

* Local `.db` storage
* Automatic persistence
* JSON import/export
* Database configuration persistence
* Index persistence and restoration

### Indexing

* Pluggable indexing architecture
* Built-in Brute Force index
* Custom index registration API
* Collection-specific indexes
* Index save/load support

### Developer Experience

* Type hints throughout the codebase
* Dataclass-based records
* Extensible architecture
* Local-first design
* No external database dependencies

---

## Main API

### Database Management

- `Database(...)` - Create or load a database.
- `save()` - Persist the database to disk.
- `close()` - Save and close the database.
- `clear()` - Clear all records or a specific collection.

### Collections

- `create_collection()` - Create a new collection.
- `delete_collection()` - Delete a collection.
- `rename_collection()` - Rename a collection.
- `clone_collection()` - Clone a collection.
- `list_collections()` - List all collections.

### Documents

- `insert_doc()` - Insert a document.
- `batch_insert_docs()` - Insert multiple documents.
- `get_doc()` - Retrieve a document.
- `update_doc()` - Update a document.
- `delete_doc()` - Delete a document.
- `list_docs()` - List document IDs.

### Search

- `search_doc()` - Perform vector similarity search.

### Configuration

- `get_config()`
- `set_config()`
- `update_config()`
- `get_config_value()`
- `has_config_value()`
- `delete_config_value()`
- `clear_config()`

### Statistics

- `stats()` - Database statistics.
- `collection_stats()` - Collection statistics.

### Import / Export

- `export_to_json()` - Export database to JSON.
- `import_from_json()` - Import database from JSON.

### Index Management

- `save_index()` - Save a collection index.
- `register_index()` - Register a custom index.
- `unregister_index()` - Unregister a custom index.
- `get_index()` - Retrieve a registered index.
- `list_indexes()` - List available indexes.
"""

def get_readme() -> str:
    return STRING