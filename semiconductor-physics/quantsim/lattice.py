from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BasisAtom:
    element: str
    position: np.ndarray  # fractional coordinates within the unit cell

    @staticmethod
    def from_fractional(element: str, frac: Iterable[float]) -> "BasisAtom":
        return BasisAtom(element=element, position=np.asarray(frac, dtype=float))


@dataclass(frozen=True)
class Lattice:
    vectors: np.ndarray  # shape (3, 3), rows are lattice vectors in Cartesian space

    @staticmethod
    def from_vectors(vectors: Sequence[Sequence[float]]) -> "Lattice":
        vec = np.asarray(vectors, dtype=float)
        if vec.shape != (3, 3):
            raise ValueError("Lattice vectors must be a 3x3 matrix")
        return Lattice(vectors=vec)

    @staticmethod
    def cubic(a: float) -> "Lattice":
        return Lattice.from_vectors(
            [
                [a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, 0.0, a],
            ]
        )

    @staticmethod
    def fcc(a: float) -> "Lattice":
        return Lattice.from_vectors(
            [
                [0.0, a / 2.0, a / 2.0],
                [a / 2.0, 0.0, a / 2.0],
                [a / 2.0, a / 2.0, 0.0],
            ]
        )

    @staticmethod
    def bcc(a: float) -> "Lattice":
        return Lattice.from_vectors(
            [
                [-a / 2.0, a / 2.0, a / 2.0],
                [a / 2.0, -a / 2.0, a / 2.0],
                [a / 2.0, a / 2.0, -a / 2.0],
            ]
        )

    @staticmethod
    def hexagonal(a: float, c: float) -> "Lattice":
        return Lattice.from_vectors(
            [
                [a, 0.0, 0.0],
                [a / 2.0, np.sqrt(3.0) * a / 2.0, 0.0],
                [0.0, 0.0, c],
            ]
        )

    def reciprocal(self) -> "Lattice":
        a1, a2, a3 = self.vectors
        volume = np.dot(a1, np.cross(a2, a3))
        scale = np.linalg.norm(a1) * np.linalg.norm(a2) * np.linalg.norm(a3)
        if scale == 0.0 or abs(volume) < 1.0e-12 * scale:
            raise ValueError("Lattice vectors are degenerate (zero volume)")
        b1 = 2.0 * np.pi * np.cross(a2, a3) / volume
        b2 = 2.0 * np.pi * np.cross(a3, a1) / volume
        b3 = 2.0 * np.pi * np.cross(a1, a2) / volume
        return Lattice.from_vectors([b1, b2, b3])

    def cartesian_from_fractional(self, frac: np.ndarray) -> np.ndarray:
        return frac @ self.vectors


@dataclass(frozen=True)
class Crystal:
    lattice: Lattice
    basis: tuple[BasisAtom, ...]

    @staticmethod
    def from_lattice_and_basis(
        lattice: Lattice, basis: Iterable[BasisAtom]
    ) -> "Crystal":
        return Crystal(lattice=lattice, basis=tuple(basis))

    def expanded_positions(self, repeats: Sequence[int]) -> np.ndarray:
        repeats = np.asarray(repeats, dtype=int)
        if repeats.shape != (3,):
            raise ValueError("repeats must be a length-3 sequence")
        grid = np.mgrid[0 : repeats[0], 0 : repeats[1], 0 : repeats[2]]
        grid = grid.reshape(3, -1).T
        positions = []
        for cell in grid:
            for atom in self.basis:
                frac = (cell + atom.position) / repeats
                positions.append(self.lattice.cartesian_from_fractional(frac))
        return np.asarray(positions)

    def expanded_positions_with_elements(
        self, repeats: Sequence[int]
    ) -> tuple[np.ndarray, list[str]]:
        repeats = np.asarray(repeats, dtype=int)
        if repeats.shape != (3,):
            raise ValueError("repeats must be a length-3 sequence")
        grid = np.mgrid[0 : repeats[0], 0 : repeats[1], 0 : repeats[2]]
        grid = grid.reshape(3, -1).T
        positions = []
        elements: list[str] = []
        for cell in grid:
            for atom in self.basis:
                frac = (cell + atom.position) / repeats
                positions.append(self.lattice.cartesian_from_fractional(frac))
                elements.append(atom.element)
        return np.asarray(positions), elements
