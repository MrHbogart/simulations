"""Grid utilities for EM simulations."""

from __future__ import annotations

import numpy as np


def create_grid(nx: int, ny: int, nz: int, extent: float):
    """Create a cubic grid centered at the origin.

    Args:
        nx, ny, nz: Number of points along each axis.
        extent: Half-length of the box in meters (box spans [-extent, extent]).

    Returns:
        X, Y, Z meshgrid arrays and spacing dx, dy, dz.
    """
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    z = np.linspace(-extent, extent, nz)
    dx = x[1] - x[0] if nx > 1 else 0.0
    dy = y[1] - y[0] if ny > 1 else 0.0
    dz = z[1] - z[0] if nz > 1 else 0.0
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z, dx, dy, dz


def cart_to_spherical(X, Y, Z, eps: float = 1e-12):
    """Convert Cartesian grid to spherical coordinates.

    Returns r, theta, phi where theta is polar angle from +z.
    """
    r = np.sqrt(X**2 + Y**2 + Z**2) + eps
    theta = np.arccos(np.clip(Z / r, -1.0, 1.0))
    phi = np.arctan2(Y, X)
    return r, theta, phi
