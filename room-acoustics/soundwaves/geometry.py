"""Room geometry and reflection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class ImageSource:
    position: np.ndarray
    reflection_count: int
    indices: Tuple[int, int, int]


def _image_coordinates(source: float, length: float, k: np.ndarray) -> np.ndarray:
    """1D images of `source` between walls at 0 and `length` after |k| reflections.

    k = 0 is the source itself; k = 1 / k = -1 are the first reflections off the
    wall at `length` / at 0, alternating between the two walls for larger |k|.
    """
    odd = k % 2 == 1
    return np.where(odd, -source + (k + 1) * length, source + k * length)


@dataclass(frozen=True)
class RoomBox:
    size: Tuple[float, float, float]

    def image_arrays(
        self, source_position: np.ndarray, order: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized image sources with at most `order` total wall reflections.

        Returns (positions (N, 3), reflection_counts (N,), indices (N, 3)), where
        indices[:, axis] = k is the signed per-axis reflection index.
        """
        order = max(int(order), 0)
        k = np.arange(-order, order + 1)
        kx, ky = (a.ravel() for a in np.meshgrid(k, k, indexing="ij"))
        rem = order - np.abs(kx) - np.abs(ky)
        kx, ky, rem = kx[rem >= 0], ky[rem >= 0], rem[rem >= 0]
        # Enumerate kz in [-rem, rem] per (kx, ky) without building the full cube.
        n_z = 2 * rem + 1
        starts = np.repeat(np.cumsum(n_z) - n_z, n_z)
        kz = np.arange(n_z.sum()) - starts - np.repeat(rem, n_z)
        kx, ky = np.repeat(kx, n_z), np.repeat(ky, n_z)
        counts = np.abs(kx) + np.abs(ky) + np.abs(kz)
        positions = np.column_stack(
            [
                _image_coordinates(source_position[axis], self.size[axis], ki)
                for axis, ki in enumerate((kx, ky, kz))
            ]
        )
        return positions, counts, np.column_stack([kx, ky, kz])

    def image_sources(self, source_position: np.ndarray, order: int) -> List[ImageSource]:
        positions, counts, indices = self.image_arrays(source_position, order)
        return [
            ImageSource(position=p, reflection_count=int(n), indices=tuple(int(v) for v in idx))
            for p, n, idx in zip(positions, counts, indices)
        ]
