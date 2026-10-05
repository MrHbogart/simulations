from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _scale_points(points: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    return {label: scale * vec for label, vec in points.items()}


def cubic_kpoints(scale: float) -> dict[str, np.ndarray]:
    points = {
        "G": np.array([0.0, 0.0, 0.0]),
        "X": np.array([0.0, 0.5, 0.0]),
        "M": np.array([0.5, 0.5, 0.0]),
        "R": np.array([0.5, 0.5, 0.5]),
    }
    return _scale_points(points, scale)


def fcc_kpoints(scale: float) -> dict[str, np.ndarray]:
    # Fractional coordinates of the FCC primitive reciprocal vectors (Setyawan-Curtarolo).
    points = {
        "G": np.array([0.0, 0.0, 0.0]),
        "X": np.array([0.5, 0.0, 0.5]),
        "W": np.array([0.5, 0.25, 0.75]),
        "K": np.array([0.375, 0.375, 0.75]),
        "L": np.array([0.5, 0.5, 0.5]),
        "U": np.array([0.625, 0.25, 0.625]),
    }
    return _scale_points(points, scale)


def bcc_kpoints(scale: float) -> dict[str, np.ndarray]:
    # Fractional coordinates of the BCC primitive reciprocal vectors (Setyawan-Curtarolo).
    points = {
        "G": np.array([0.0, 0.0, 0.0]),
        "H": np.array([0.5, -0.5, 0.5]),
        "N": np.array([0.0, 0.0, 0.5]),
        "P": np.array([0.25, 0.25, 0.25]),
    }
    return _scale_points(points, scale)


@dataclass(frozen=True)
class BrillouinZonePath:
    labels: tuple[str, ...]
    points: dict[str, np.ndarray]

    @staticmethod
    def for_lattice(kind: str, scale: float = 1.0) -> "BrillouinZonePath":
        kind = kind.lower()
        if kind == "cubic":
            points = cubic_kpoints(scale)
            labels = ("G", "X", "M", "G", "R", "X")
        elif kind == "fcc":
            points = fcc_kpoints(scale)
            labels = ("G", "X", "W", "K", "G", "L", "U", "W", "L", "K")
        elif kind == "bcc":
            points = bcc_kpoints(scale)
            labels = ("G", "H", "N", "G", "P", "H")
        else:
            raise ValueError(f"Unknown lattice kind: {kind}")
        return BrillouinZonePath(labels=labels, points=points)

    def sample(self, points_per_segment: int = 50) -> np.ndarray:
        if points_per_segment < 2:
            raise ValueError("points_per_segment must be >= 2")
        coords = []
        for start_label, end_label in zip(self.labels[:-1], self.labels[1:]):
            start = self.points[start_label]
            end = self.points[end_label]
            for t in np.linspace(0.0, 1.0, points_per_segment, endpoint=False):
                coords.append(start + t * (end - start))
        coords.append(self.points[self.labels[-1]])
        return np.asarray(coords)

    def segments(self) -> Iterable[tuple[str, str]]:
        return zip(self.labels[:-1], self.labels[1:])
