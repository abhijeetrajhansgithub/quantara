import os
from pprint import pprint

import ollama

from quantara import Database

EMBEDDING_MODEL = "snowflake-arctic-embed:335m"

DB_NAME = "quantara_v6_test"

COLLECTIONS = [
    "research",
    "history",
    "science",
    "technology",
    "books",
    "finance",
    "health",
]

DOCUMENTS = {
    "research": [
        "Artificial intelligence is transforming the future of scientific research.",
        "Large language models have demonstrated impressive reasoning abilities.",
        "Machine learning enables computers to learn patterns from data.",
    ],
    "history": [
        "Ancient civilizations developed near major river systems.",
        "The Roman Empire influenced law, culture, and governance.",
        "Archaeological discoveries help us understand human history.",
    ],
    "science": [
        "Physics seeks to understand the fundamental laws of nature.",
        "Biology studies living organisms and their interactions.",
        "Chemistry explores the composition and properties of matter.",
    ],
    "technology": [
        "Cloud computing enables scalable software deployment.",
        "Cybersecurity protects systems against malicious attacks.",
        "Modern databases can handle billions of records efficiently.",
    ],
    "books": [
        "Reading books develops critical thinking skills.",
        "Literature reflects culture and society.",
        "Libraries preserve knowledge across generations.",
    ],
    "finance": [
        "Stock markets facilitate capital allocation.",
        "Risk management is essential in financial planning.",
        "Interest rates influence economic activity.",
    ],
    "health": [
        "Exercise contributes to physical well-being.",
        "Nutrition plays a key role in human health.",
        "Preventive healthcare reduces disease burden.",
    ],
}

INDEX_PATHS = {
    "default": f"{DB_NAME}_default.index",
    "research": f"{DB_NAME}_research.index",
    "history": f"{DB_NAME}_history.index",
    "science": f"{DB_NAME}_science.index",
    "technology": f"{DB_NAME}_technology.index",
    "books": f"{DB_NAME}_books.index",
    "finance": f"{DB_NAME}_finance.index",
    "health": f"{DB_NAME}_health.index",
}


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cleanup_artifacts() -> None:
    files = [f"{DB_NAME}.db", *INDEX_PATHS.values()]
    for path in files:
        if os.path.exists(path):
            os.remove(path)


def embed(text: str) -> list[float]:
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text,
    )
    return response["embedding"]


def build_db(load_indexes: bool = False) -> Database:
    if load_indexes:
        return Database(
            db_name=DB_NAME,
            dimensions=None,
            auto_dim=True,
            auto_persist=True,
            index_paths=INDEX_PATHS,
        )

    return Database(
        db_name=DB_NAME,
        dimensions=None,
        auto_dim=True,
        auto_persist=True,
    )


def create_collections(db: Database) -> None:
    for collection in COLLECTIONS:
        db.create_collection(collection)

    ensure(
        set(db.list_collections()) >= set(COLLECTIONS),
        "Collections were not created correctly.",
    )


def insert_documents(db: Database) -> None:
    for collection in COLLECTIONS:
        docs = DOCUMENTS[collection]

        if collection in {"research", "history"}:
            payload = []
            for idx, sentence in enumerate(docs):
                payload.append(
                    (
                        sentence,
                        embed(sentence),
                        {
                            "collection": collection,
                            "doc_index": idx,
                            "word_count": len(sentence.split()),
                        },
                    )
                )

            db.batch_insert_docs(
                objects=payload,
                collection=collection,
            )
        else:
            for idx, sentence in enumerate(docs):
                db.insert_doc(
                    collection=collection,
                    name=sentence,
                    vector=embed(sentence),
                    metadata={
                        "collection": collection,
                        "doc_index": idx,
                        "word_count": len(sentence.split()),
                    },
                )

    for collection in COLLECTIONS:
        ensure(
            len(db.list_docs(collection)) == len(DOCUMENTS[collection]),
            f"Unexpected doc count in collection '{collection}' after insert.",
        )


def perform_operations(db: Database) -> None:
    print("\n=== Initial Stats ===")
    pprint(db.stats())

    print("\n=== Collection Stats ===")
    for collection in COLLECTIONS:
        pprint(db.collection_stats(collection))

    queries = [
        ("How do neural networks learn?", "research"),
        ("Tell me about ancient empires", "history"),
        ("How does cloud computing work?", "technology"),
        ("What affects interest rates?", "finance"),
    ]

    print("\n=== Search Results (basic=False, index-backed) ===")
    for query, collection in queries:
        results = db.search_doc(
            input_vector=embed(query),
            collection=collection,
            top_k=3,
            metric="cosine",
            return_text_outputs=True,
            basic=False,
        )
        print(f"\nCollection: {collection}")
        pprint(results)
        ensure(results, f"No search results returned for collection '{collection}'.")

    print("\n=== Metadata Filter Test ===")
    filtered = db.search_doc(
        input_vector=embed("computer learning systems"),
        collection="research",
        filters={"doc_index": 1},
        top_k=5,
        metric="cosine",
        return_text_outputs=True,
        basic=False,
    )
    pprint(filtered)
    ensure(filtered, "Metadata filter returned no results.")
    ensure(
        all(item[3].get("doc_index") == 1 for item in filtered),
        "Metadata filtering did not restrict results correctly.",
    )

    print("\n=== Update Test ===")
    research_docs = db.list_docs(collection="research")
    target_id = research_docs[0]

    db.update_doc(
        id=target_id,
        collection="research",
        name="Updated research note",
        vector=embed("Adaptive models improve scientific workflows."),
        metadata={
            "updated": True,
            "source": "integration_test",
        },
    )

    updated_doc = db.get_doc(
        target_id,
        collection="research",
    )
    pprint(updated_doc)
    ensure(
        updated_doc.metadata.get("updated") is True,
        "Update did not persist metadata correctly.",
    )

    print("\n=== Delete Test ===")
    finance_docs = db.list_docs(collection="finance")
    delete_id = finance_docs[0]

    db.delete_doc(
        delete_id,
        collection="finance",
    )

    remaining_finance = db.list_docs(collection="finance")
    pprint(remaining_finance)
    ensure(
        len(remaining_finance) == 2,
        "Delete did not reduce the finance collection size.",
    )

    print("\n=== Metric Tests ===")
    metric_query = embed("machine learning")

    for metric in ["cosine", "dot", "euclidean"]:
        results = db.search_doc(
            metric_query,
            collection="science",
            metric=metric,
            top_k=3,
            return_text_outputs=True,
            basic=False,
        )
        print(f"\nMetric: {metric}")
        pprint(results)
        ensure(results, f"Metric '{metric}' returned no results.")

    print("\n=== Post-Operation Stats ===")
    pprint(db.stats())


def save_all_indexes(db: Database) -> None:
    print("\n=== Saving Indexes ===")
    for collection in db.list_collections():
        if collection == "_config":
            continue

        path = INDEX_PATHS.get(collection)
        if path is None:
            continue

        saved_path = db.save_index(
            collection=collection,
            path=path,
        )
        print(f"Saved index for '{collection}' -> {saved_path}")


def delete_collections_without_persist(db: Database) -> None:
    print("\n=== Deleting Collections (scratch copy only) ===")
    for collection in list(db.list_collections()):
        if collection == "default":
            continue
        db.delete_collection(collection)

    print("Remaining collections after delete:", db.list_collections())
    ensure(
        db.list_collections() == ["default"],
        "Scratch DB should only retain the default collection after deletion.",
    )


def verify_rebuilt_db(db: Database) -> None:
    print("\n=== Rebuilt DB Stats ===")
    pprint(db.stats())

    for collection in COLLECTIONS:
        ensure(
            collection in db.list_collections(),
            f"Collection '{collection}' missing after rebuild.",
        )

    ensure(
        len(db.list_docs("research")) == 3,
        "Research collection did not reload correctly.",
    )
    ensure(
        len(db.list_docs("finance")) == 2,
        "Finance collection did not reload correctly after delete.",
    )

    rebuilt_results = db.search_doc(
        input_vector=embed("artificial intelligence research"),
        collection="research",
        top_k=3,
        metric="cosine",
        return_text_outputs=True,
        basic=False,
    )

    print("\n=== Rebuilt Search Results ===")
    pprint(rebuilt_results)
    ensure(
        rebuilt_results,
        "Rebuilt database search returned no results.",
    )

    print("\n=== Rebuilt Collection Stats ===")
    pprint(db.collection_stats("research"))


def main() -> None:
    cleanup_artifacts()

    # Phase 1: create collections, then insert docs, then perform operations.
    db = build_db(load_indexes=False)
    create_collections(db)
    insert_documents(db)
    perform_operations(db)

    # Phase 2: save collections to indexes.
    save_all_indexes(db)
    db.persist_doc()

    print("[1] All collections list: ", db.list_collections())

    # Phase 3: delete collections in a scratch copy only.
    scratch = build_db(load_indexes=True)
    # delete_collections_without_persist(scratch)

    print("[2] All collections list: ", scratch.list_collections())

    # Phase 4: build again from the persisted db + index files.
    rebuilt = build_db(load_indexes=True)
    verify_rebuilt_db(rebuilt)

    print("\nDone.")


if __name__ == "__main__":
    main()