"""Visualization helpers for EM fields."""

from __future__ import annotations

import numpy as np


def field_magnitude(Ex, Ey, Ez):
    return np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)


def poynting_vector(Ex, Ey, Ez, Hx, Hy, Hz):
    """Time-averaged Poynting vector (phasor fields)."""
    Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
    Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
    return Sx, Sy, Sz


def take_slice(volume, axis: str, index: int):
    if axis == "x":
        return volume[index, :, :]
    if axis == "y":
        return volume[:, index, :]
    if axis == "z":
        return volume[:, :, index]
    raise ValueError("axis must be 'x', 'y', or 'z'")
