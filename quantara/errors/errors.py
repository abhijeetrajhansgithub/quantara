class CollectionNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


class IndexNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


class EmbeddingDimensionError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


class CollectionAlreadyExistsError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
