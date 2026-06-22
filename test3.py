from quantara import Database
from quantara.errors.errors import EmbeddingDimensionError


def header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def expect_dimension_error(test_name: str, func) -> None:   # type: ignore

    try:
        func()

        print(f"[FAIL] {test_name}")

    except EmbeddingDimensionError:

        print(f"[PASS] {test_name}")

    except Exception as e:

        print(f"[FAIL] {test_name}")

        print(f"Expected EmbeddingDimensionError " f"but got {type(e).__name__}: {e}")


def main() -> None:

    header("CREATE DATABASE")

    db = Database(db_name="dimension_test")

    # =====================================================
    # TEST 1
    # Auto dimension inference
    # =====================================================

    header("TEST 1 - AUTO DIMENSION INFERENCE")

    doc_id = db.insert_doc(name="doc1", vector=[1.0, 2.0, 3.0], metadata={})

    print("Detected dimensions:", db.dimensions)

    assert db.dimensions == 3

    print("[PASS]")

    # =====================================================
    # TEST 2
    # Insert wrong dimension
    # =====================================================

    header("TEST 2 - INSERT WRONG DIMENSION")

    expect_dimension_error(
        "insert dimension mismatch",
        lambda: db.insert_doc(name="bad_doc", vector=[1.0, 2.0], metadata={}),
    )

    # =====================================================
    # TEST 3
    # Batch insert valid dimensions
    # =====================================================

    header("TEST 3 - BATCH INSERT VALID")

    ids = db.batch_insert_docs(
        [("batch1", [1.0, 2.0, 3.0], {}), ("batch2", [4.0, 5.0, 6.0], {})]
    )

    print("[PASS]")

    print("Inserted:", len(ids))

    # =====================================================
    # TEST 4
    # Batch insert invalid dimension
    # =====================================================

    header("TEST 4 - BATCH INSERT INVALID")

    expect_dimension_error(
        "batch insert mismatch",
        lambda: db.batch_insert_docs([("bad_batch", [1.0, 2.0], {})]),
    )

    # =====================================================
    # TEST 5
    # Update valid dimension
    # =====================================================

    header("TEST 5 - UPDATE VALID")

    db.update_doc(id=doc_id, vector=[7.0, 8.0, 9.0])

    print("[PASS]")

    # =====================================================
    # TEST 6
    # Update invalid dimension
    # =====================================================

    header("TEST 6 - UPDATE INVALID")

    expect_dimension_error(
        "update mismatch", lambda: db.update_doc(id=doc_id, vector=[1.0, 2.0])
    )

    # =====================================================
    # TEST 7
    # Persist + Reload
    # =====================================================

    header("TEST 7 - PERSIST + RELOAD")

    db.save()

    db2 = Database(db_name="dimension_test")

    print("Reloaded dimensions:", db2.dimensions)

    assert db2.dimensions == 3

    print("[PASS]")

    # =====================================================
    # TEST 8
    # Insert invalid dimension after reload
    # =====================================================

    header("TEST 8 - RELOAD DIMENSION CHECK")

    expect_dimension_error(
        "reload mismatch",
        lambda: db2.insert_doc(name="bad_after_reload", vector=[1.0, 2.0], metadata={}),
    )

    # =====================================================
    # TEST 9
    # Search dimension mismatch
    # =====================================================

    header("TEST 9 - SEARCH DIMENSION CHECK")

    expect_dimension_error(
        "search mismatch", lambda: db2.search_doc(input_vector=[1.0, 2.0])
    )

    # =====================================================
    # TEST 10
    # Collection dimension consistency
    # =====================================================

    header("TEST 10 - CROSS COLLECTION CONSISTENCY")

    db2.create_collection("test_collection")

    db2.insert_doc(
        collection="test_collection", name="good", vector=[1.0, 2.0, 3.0], metadata={}
    )

    expect_dimension_error(
        "cross collection mismatch",
        lambda: db2.insert_doc(
            collection="test_collection", name="bad", vector=[1.0, 2.0], metadata={}
        ),
    )

    header("FINAL STATS")

    print(db2.stats())

    print("\nALL TESTS COMPLETED")


if __name__ == "__main__":
    main()
