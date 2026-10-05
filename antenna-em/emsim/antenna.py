"""Antenna models."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def c0():
    return 299_792_458.0


def mu0():
    return 4.0e-7 * np.pi


def eps0():
    return 1.0 / (mu0() * c0() ** 2)


def eta0():
    return np.sqrt(mu0() / eps0())


@dataclass
class DipoleAntenna:
    """Simple z-directed dipole antenna model."""

    length: float
    radius: float
    current_amp: float
    frequency: float

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency

    @property
    def k(self) -> float:
        return self.omega / c0()

    @property
    def dipole_moment(self) -> complex:
        # p = I0 * l / (i * omega)
        return self.current_amp * self.length / (1j * self.omega)

    def current_distribution(self, z: np.ndarray) -> np.ndarray:
        """Sinusoidal current distribution for a thin, center-fed dipole."""
        L = self.length
        k = self.k
        z_abs = np.abs(z)
        # Zero current at the ends, peak at the feed (z=0).
        return self.current_amp * np.sin(k * (L / 2.0 - z_abs))


@dataclass
class WireAntenna:
    """Thin-wire antenna modeled as a polyline path."""

    path: np.ndarray
    radius: float
    current_amp: float
    frequency: float
    current_profile: str = "uniform"
    phase: float = 0.0

    def __post_init__(self) -> None:
        path = np.asarray(self.path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError("path must be an (N, 3) array of points")
        self.path = path

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.frequency

    @property
    def k(self) -> float:
        return self.omega / c0()

    def is_closed(self) -> bool:
        return np.allclose(self.path[0], self.path[-1], atol=1e-9)

    def segment_data(self):
        segments = self.path[1:] - self.path[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths == 0.0):
            raise ValueError("path contains repeated points (zero-length segments)")
        directions = segments / lengths[:, None]
        centers = 0.5 * (self.path[1:] + self.path[:-1])
        s_edges = np.concatenate(([0.0], np.cumsum(lengths)))
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        return centers, directions, lengths, s_centers, s_edges[-1]

    def current_distribution(self, s_centers: np.ndarray, total_length: float) -> np.ndarray:
        if self.current_profile not in {"uniform", "sinusoidal"}:
            raise ValueError("current_profile must be 'uniform' or 'sinusoidal'")
        if self.current_profile == "uniform" or self.is_closed():
            current = np.full_like(s_centers, self.current_amp, dtype=float)
        else:
            s_rel = s_centers - 0.5 * total_length
            current = self.current_amp * np.sin(self.k * (total_length / 2.0 - np.abs(s_rel)))
        return current * np.exp(1j * self.phase)


def make_straight_wire(length: float, n_points: int = 101, axis: str = "z") -> np.ndarray:
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    coords = np.linspace(-length / 2.0, length / 2.0, n_points)
    path = np.zeros((n_points, 3))
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    path[:, idx] = coords
    return path


def make_circular_loop(radius: float, n_points: int = 181, plane: str = "xy") -> np.ndarray:
    plane = plane.lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)
    if plane == "xz":
        return np.column_stack([x, z, y])
    if plane == "yz":
        return np.column_stack([z, x, y])
    return np.column_stack([x, y, z])


def make_rectangular_loop(
    width: float, height: float, n_points: int = 201, plane: str = "xy"
) -> np.ndarray:
    plane = plane.lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    w = width / 2.0
    h = height / 2.0
    corners = np.array([
        [-w, -h, 0.0],
        [w, -h, 0.0],
        [w, h, 0.0],
        [-w, h, 0.0],
        [-w, -h, 0.0],
    ])
    t = np.linspace(0.0, 1.0, n_points)
    seg1 = corners[0] + (corners[1] - corners[0]) * t[:, None]
    seg2 = corners[1] + (corners[2] - corners[1]) * t[:, None]
    seg3 = corners[2] + (corners[3] - corners[2]) * t[:, None]
    seg4 = corners[3] + (corners[4] - corners[3]) * t[:, None]
    path = np.vstack([seg1, seg2[1:], seg3[1:], seg4[1:]])
    if plane == "xz":
        return path[:, [0, 2, 1]]
    if plane == "yz":
        return path[:, [2, 0, 1]]
    return path


def make_v_dipole(length: float, angle_deg: float, n_points: int = 121, plane: str = "xz") -> np.ndarray:
    plane = plane.lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    half = length / 2.0
    angle = np.deg2rad(angle_deg / 2.0)
    arm_len = half / np.cos(angle)
    t = np.linspace(0.0, 1.0, n_points)
    arm1 = np.column_stack([
        -arm_len * t * np.sin(angle),
        np.zeros_like(t),
        arm_len * t * np.cos(angle),
    ])
    arm2 = np.column_stack([
        arm_len * t * np.sin(angle),
        np.zeros_like(t),
        arm_len * t * np.cos(angle),
    ])
    path = np.vstack([arm1[::-1], arm2[1:]])
    if plane == "xy":
        return path[:, [0, 2, 1]]
    if plane == "yz":
        return path[:, [2, 0, 1]]
    return path


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to `n_points` evenly spaced along its arc length.

    Each segment is treated as a Hertzian dipole, so segments must stay short
    compared to the wavelength; a 4-corner custom path is far too coarse as is.
    """
    points = np.asarray(points, dtype=float)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    s_new = np.linspace(0.0, s[-1], n_points)
    return np.column_stack([np.interp(s_new, s, points[:, i]) for i in range(3)])
