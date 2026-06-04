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

## Installation

```bash
pip install quantara
```

Or install from source:

```bash
git clone https://github.com/<your-username>/quantara.git

cd quantara

pip install -e .
```

---

## Quick Start

```python
import ollama

from quantara import Database

EMBEDDING_MODEL = "snowflake-arctic-embed:335m"

db = Database(
    "my_database",
    auto_persist=True
)

text = "I want to build AI agents."

embedding = ollama.embeddings(
    model=EMBEDDING_MODEL,
    prompt=text
)["embedding"]

db.insert_doc(
    name=text,
    vector=embedding,
    metadata={
        "category": "ai"
    }
)

query = "How can I build autonomous AI systems?"

query_embedding = ollama.embeddings(
    model=EMBEDDING_MODEL,
    prompt=query
)["embedding"]

results = db.search_doc(
    query_embedding,
    top_k=3
)

print(results)
```

---

## Collections

Create a collection:

```python
db.create_collection(
    "research"
)
```

Insert into a collection:

```python
db.insert_doc(
    collection="research",
    name="Transformers Paper",
    vector=embedding,
    metadata={
        "author": "Vaswani"
    }
)
```

List collections:

```python
db.list_collections()
```

Rename a collection:

```python
db.rename_collection(
    "research",
    "papers"
)
```

Clone a collection:

```python
db.clone_collection(
    "papers",
    "papers_backup"
)
```

Delete a collection:

```python
db.delete_collection(
    "papers_backup"
)
```

---

## Metadata Filtering

Search only documents matching specific metadata:

```python
results = db.search_doc(
    query_embedding,
    collection="research",
    filters={
        "author": "Vaswani"
    }
)
```

---

## Batch Insertion

```python
db.batch_insert_docs(
    objects=[
        (
            "Document 1",
            embedding_1,
            {
                "type": "paper"
            }
        ),
        (
            "Document 2",
            embedding_2,
            {
                "type": "article"
            }
        )
    ],
    collection="research"
)
```

---

## Similarity Metrics

Quantara supports multiple similarity metrics:

```python
db.search_doc(
    query_embedding,
    metric="cosine"
)
```

```python
db.search_doc(
    query_embedding,
    metric="dot"
)
```

```python
db.search_doc(
    query_embedding,
    metric="euclidean"
)
```

---

## Persistence

Persist database state:

```python
db.persist_doc()
```

Export database:

```python
db.export_to_json(
    "backup.json"
)
```

Import database:

```python
db.import_from_json(
    "backup.json"
)
```

---

## Indexes

### Saving an Index

```python
db.save_index(
    collection="research",
    path="research.index"
)
```

### Loading an Index

```python
db = Database(
    "my_database",
    index_paths={
        "research": "research.index"
    }
)
```

---

## Creating Custom Indexes

Quantara provides a pluggable indexing architecture.

Create a custom index:

```python
from quantara import BaseIndex


class MyIndex(BaseIndex):

    def build(self, vectors):
        ...

    def add(self, record_id, vector):
        ...

    def remove(self, record_id):
        ...

    def update(self, record_id, vector):
        ...

    def search(
        self,
        query_vector,
        top_k=3,
        **kwargs
    ):
        ...

    def clear(self):
        ...

    def size(self):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
```

Register the index:

```python
from quantara import register_index

register_index(
    "myindex",
    MyIndex
)
```

Use it in a collection:

```python
db.create_collection(
    "research",
    index_type="myindex"
)
```

---

## Statistics

Database statistics:

```python
db.stats()
```

Collection statistics:

```python
db.collection_stats(
    "research"
)
```

---

## Architecture

```text
Database
│
├── Collections
│   ├── Records
│   └── Indexes
│
├── Persistence Layer
│
├── Metadata Layer
│
└── Search Layer
```

---

## Project Goals

Quantara aims to provide:

* A lightweight local vector database
* A simple developer experience
* Extensible indexing support
* Fast experimentation for AI and RAG applications
* An educational implementation of vector database concepts

---

## Roadmap

### Version 1.x

* Brute Force Index
* Metadata Filtering
* Collection Management
* Persistent Storage
* Custom Index Registration

### Version 2.x

* HNSW Index
* Approximate Nearest Neighbor Search
* Additional Search Optimizations

### Version 3.x

* Vector Quantization
* Compressed Vector Storage
* Advanced Indexing Strategies

---

## License

MIT License

---

## Author

Abhijeet Rajhans

Built as a learning-focused vector database project exploring vector search, indexing systems, persistence, and database architecture in Python.
