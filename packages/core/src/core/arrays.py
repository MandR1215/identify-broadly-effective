from typing import Optional
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

def as_float_array(x: np.ndarray) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim < 2:
        raise ValueError("input must have at least one-dimensional.")

    return arr


def as_mask(x: np.ndarray, shape: Optional[tuple[int, ...]]=None) -> BoolArray:
    mask = np.asarray(x, dtype=bool)

    if mask.ndim == 1:
        mask = mask[None, :]

    if shape is not None:
        if mask.shape[-1] != shape[-1]:
            raise ValueError(
                f"The last dimension must match the given shape."
                f"Got x.shape={mask.shape}, shape={shape}."
            )
        try:
            mask = np.broadcast_to(mask, shape)
        except ValueError as e:
            raise ValueError(
                f"x must be broadcastable to the given shape."
                f"Got x.shape={mask.shape}, shape={shape}."
            ) from e

    return mask