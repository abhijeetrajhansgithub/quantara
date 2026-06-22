from quantara.indexes.registry import INDEX_REGISTRY

from quantara.indexes.bruteforce import BruteForceIndex


def load_builtin_indexes() -> None:

    builtins = {"bruteforce": BruteForceIndex}

    for name, index_cls in builtins.items():

        if not INDEX_REGISTRY.exists(name):

            INDEX_REGISTRY.register(name, index_cls)
