import uuid
from typing import Sequence
import numpy as np
from numpy.typing import NDArray


def get_uuid() -> str:
    return str(uuid.uuid4())
    


def cosine_similarity(
    a: Sequence[float] | NDArray[np.float32],
    b: Sequence[float] | NDArray[np.float32],
) -> float:

    a_arr: NDArray[np.float32] = np.asarray(
        a,
        dtype=np.float32,
    )

    b_arr: NDArray[np.float32] = np.asarray(
        b,
        dtype=np.float32,
    )

    similarity: float = float(
        np.dot(a_arr, b_arr)
        / (
            np.linalg.norm(a_arr)
            * np.linalg.norm(b_arr)
        )
    )

    return similarity


def dot_similarity(
    a: Sequence[float] | NDArray[np.float32],
    b: Sequence[float] | NDArray[np.float32],
) -> float:

    a_arr = np.asarray(
        a,
        dtype=np.float32
    )

    b_arr = np.asarray(
        b,
        dtype=np.float32
    )

    return float(
        np.dot(
            a_arr,
            b_arr
        )
    )



def euclidean_distance(
    a: Sequence[float] | NDArray[np.float32],
    b: Sequence[float] | NDArray[np.float32],
) -> float:

    a_arr = np.asarray(
        a,
        dtype=np.float32
    )

    b_arr = np.asarray(
        b,
        dtype=np.float32
    )

    return float(
        np.linalg.norm(
            a_arr - b_arr
        )
    )