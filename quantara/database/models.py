from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Record:
    id: str
    name: str
    vector: list[float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class Config:
    dimensions: int | None
    auto_dim: bool
    auto_persist: bool
