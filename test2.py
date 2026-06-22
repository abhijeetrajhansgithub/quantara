from quantara import Database


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:

    print_header("CREATE EMPTY DATABASE")

    db = Database(db_name="empty_test")

    print("Collections:")
    print(db.list_collections())

    print("\nStats:")
    print(db.stats())

    print("\nCollection Stats:")
    print(db.collection_stats())

    print("\nDocs:")
    print(db.list_docs())

    # --------------------------------------------------
    # Search on empty collection
    # --------------------------------------------------

    print_header("SEARCH EMPTY DATABASE")

    try:
        results = db.search_doc(input_vector=[0.0], collection="default")

        print("Search Results:")
        print(results)

    except Exception as e:
        print("Search Error:")
        print(type(e).__name__, e)

    # --------------------------------------------------
    # Persistence
    # --------------------------------------------------

    print_header("PERSIST EMPTY DATABASE")

    db.save()

    print("Persist successful.")

    # --------------------------------------------------
    # Reload
    # --------------------------------------------------

    print_header("RELOAD EMPTY DATABASE")

    db2 = Database(db_name="empty_test")

    print("Collections:")
    print(db2.list_collections())

    print("\nStats:")
    print(db2.stats())

    print("\nCollection Stats:")
    print(db2.collection_stats())

    print("\nDocs:")
    print(db2.list_docs())

    # --------------------------------------------------
    # Clear
    # --------------------------------------------------

    print_header("CLEAR EMPTY DATABASE")

    db2.clear()

    print("Collections:")
    print(db2.list_collections())

    print("\nStats:")
    print(db2.stats())

    print("\nCollection Stats:")
    print(db2.collection_stats())

    print("\nDocs:")
    print(db2.list_docs())

    print_header("EMPTY DATABASE TEST COMPLETE")


if __name__ == "__main__":
    main()
