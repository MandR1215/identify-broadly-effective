from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .arrays import FloatArray


class Allocator(Protocol):
    def allocate(self, size: tuple[int, ...]) -> FloatArray: ...


@dataclass
class AllAssignment:
    def allocate(self, size: tuple[int, ...]) -> FloatArray:
        return np.ones(shape=size, dtype=np.float64)


@dataclass
class Balanced:
    def allocate(self, size: tuple[int, ...]) -> FloatArray:
        if len(size) < 2:
            raise ValueError("size must include participant and candidate axes.")

        n, m = size[-2:]

        if n <= 0 or m <= 0:
            raise ValueError("participant and candidate axes must be positive.")

        if m > n:
            raise ValueError("number of candidates must be at most number of participants.")

        base_size = n // m
        remainder = n % m

        block_sizes = np.full(m, base_size, dtype=np.int64)
        block_sizes[:remainder] += 1

        # Candidate ids for the fixed ordered balanced partition I_1, ..., I_m.
        # shape: (n,)
        candidate_ids_1d = np.repeat(np.arange(m), block_sizes)

        # Reuse the same deterministic partition over any batch axes.
        # shape: (*batch, n)
        candidate_ids = np.broadcast_to(candidate_ids_1d, size[:-1])
        
        # Convert candidate ids to one-hot candidate masks.
        # shape: (*batch, n, m)
        return np.eye(m, dtype=np.float64)[candidate_ids]
        