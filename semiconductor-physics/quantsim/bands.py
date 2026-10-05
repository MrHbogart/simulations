from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bz import BrillouinZonePath
from .lattice import Lattice

_HBAR = 1.054_571_817e-34  # J*s
_ELECTRON_MASS = 9.109_383_7015e-31  # kg
_EV_J = 1.602_176_634e-19  # J per eV
_HC_EV_NM = 1_239.841_984  # eV*nm


@dataclass(frozen=True)
class Dopant:
    element: str
    concentration_cm3: float
    kind: str  # "n" or "p"


@dataclass(frozen=True)
class SemiconductorParams:
    name: str
    lattice_constant_m: float
    band_gap_ev: float
    electron_effective_mass: float
    hole_effective_mass: float
    temperature_k: float = 300.0


_DOPANT_GAP_COEFFS_EV_PER_1E19 = {
    "P": -0.03,
    "B": -0.025,
}


class SemiconductorModel:
    def __init__(self, params: SemiconductorParams, dopants: list[Dopant] | None = None):
        self.params = params
        self.dopants = dopants or []

    def band_gap_ev(self) -> float:
        base_gap = self.params.band_gap_ev
        delta = 0.0
        for dopant in self.dopants:
            coeff = _DOPANT_GAP_COEFFS_EV_PER_1E19.get(dopant.element, 0.0)
            delta += coeff * (dopant.concentration_cm3 / 1.0e19)
        return max(base_gap + delta, 0.0)

    def band_edges(self, k_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k_norm = np.linalg.norm(k_cart, axis=1)
        m_e = self.params.electron_effective_mass * _ELECTRON_MASS
        m_h = self.params.hole_effective_mass * _ELECTRON_MASS
        kinetic_e = (_HBAR**2 * k_norm**2) / (2.0 * m_e) / _EV_J
        kinetic_h = (_HBAR**2 * k_norm**2) / (2.0 * m_h) / _EV_J
        gap = self.band_gap_ev()
        valence = -kinetic_h
        conduction = gap + kinetic_e
        return valence, conduction

    def absorption_edge_wavelength_nm(self) -> float:
        gap = self.band_gap_ev()
        if gap <= 0.0:
            return float("inf")
        return _HC_EV_NM / gap


def sample_band_structure(
    lattice: Lattice,
    path: BrillouinZonePath,
    model: SemiconductorModel,
    points_per_segment: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[str]]:
    if points_per_segment < 2:
        raise ValueError("points_per_segment must be >= 2")
    k_frac = path.sample(points_per_segment)
    k_cart = k_frac @ lattice.reciprocal().vectors
    distances = _cumulative_distances(k_cart)

    ticks = [0.0]
    labels = [path.labels[0]]
    index = 0
    for _ in path.segments():
        index += points_per_segment
        ticks.append(distances[index])
        labels.append(path.labels[len(labels)])

    valence, conduction = model.band_edges(k_cart)
    return distances, valence, conduction, ticks, labels


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])
