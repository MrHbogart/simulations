from __future__ import annotations

from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from .bz import BrillouinZonePath
from .lattice import Crystal, Lattice


def plot_crystal(
    crystal: Crystal,
    repeats: Iterable[int] = (2, 2, 2),
    ax: plt.Axes | None = None,
):
    positions, elements = crystal.expanded_positions_with_elements(repeats)
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    unique_elements = list(dict.fromkeys(elements))
    cmap = plt.get_cmap("tab10")
    color_map = {
        element: cmap(index % cmap.N) for index, element in enumerate(unique_elements)
    }
    colors = [color_map[element] for element in elements]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=40, c=colors)

    origin = np.zeros(3)
    for vec in crystal.lattice.vectors:
        ax.plot(
            [origin[0], vec[0]],
            [origin[1], vec[1]],
            [origin[2], vec[2]],
            color="black",
            linewidth=2.0,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Crystal structure")
    return ax


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def plot_kpath(
    lattice: Lattice,
    path: BrillouinZonePath,
    points_per_segment: int = 50,
    ax: plt.Axes | None = None,
):
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(distances, np.zeros_like(distances), color="black")
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Γ" if label == "G" else label for label in labels])
    ax.set_title("High-symmetry k-path")
    ax.set_xlabel("k-path distance")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
    return ax


def plot_reciprocal_lattice(
    lattice: Lattice,
    repeats: int = 1,
    ax: plt.Axes | None = None,
):
    reciprocal = lattice.reciprocal()
    b1, b2, b3 = reciprocal.vectors
    grid = np.mgrid[-repeats : repeats + 1, -repeats : repeats + 1, -repeats : repeats + 1]
    points = grid.reshape(3, -1).T
    k_points = points @ reciprocal.vectors
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(k_points[:, 0], k_points[:, 1], k_points[:, 2], s=30)
    origin = np.zeros(3)
    for vec in (b1, b2, b3):
        ax.plot(
            [origin[0], vec[0]],
            [origin[1], vec[1]],
            [origin[2], vec[2]],
            color="black",
            linewidth=2.0,
        )
    ax.set_xlabel("k_x")
    ax.set_ylabel("k_y")
    ax.set_zlabel("k_z")
    ax.set_title("Reciprocal lattice")
    return ax


def plot_band_structure(
    distances: np.ndarray,
    valence: np.ndarray,
    conduction: np.ndarray,
    ticks: list[float],
    labels: list[str],
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distances, valence, color="#1f77b4", label="Valence band")
    ax.plot(distances, conduction, color="#d62728", label="Conduction band")
    ax.fill_between(distances, valence, conduction, color="gray", alpha=0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Γ" if label == "G" else label for label in labels])
    ax.set_xlabel("k-path distance")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Band structure along k-path")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False)
    return ax


def plot_bands(
    distances: np.ndarray,
    energies: np.ndarray,
    ticks: list[float],
    labels: list[str],
    vbm_ev: float | None = None,
    cbm_ev: float | None = None,
    vbm_k_index: int | None = None,
    cbm_k_index: int | None = None,
    ax: plt.Axes | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    for band in energies.T:
        ax.plot(distances, band, color="#1f77b4", linewidth=1.0, alpha=0.9)
    if vbm_k_index is not None and vbm_ev is not None:
        ax.scatter(
            distances[vbm_k_index],
            vbm_ev,
            color="#2ca02c",
            s=40,
            label="VBM",
            zorder=5,
        )
    if cbm_k_index is not None and cbm_ev is not None:
        ax.scatter(
            distances[cbm_k_index],
            cbm_ev,
            color="#d62728",
            s=40,
            label="CBM",
            zorder=5,
        )
    if vbm_ev is not None and cbm_ev is not None and np.isfinite(vbm_ev) and np.isfinite(cbm_ev):
        ax.axhline(vbm_ev, color="#2ca02c", linestyle="--", linewidth=1.0)
        ax.axhline(cbm_ev, color="#d62728", linestyle="--", linewidth=1.0)
        ax.fill_between(distances, vbm_ev, cbm_ev, color="gray", alpha=0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["Γ" if label == "G" else label for label in labels])
    ax.set_xlabel("k-path distance")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Plane-wave band structure")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", frameon=False)
    return ax
