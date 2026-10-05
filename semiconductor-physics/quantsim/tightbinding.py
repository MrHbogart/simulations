from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TightBindingModel:
    lattice_constant_m: float
    hopping_ev: float
    sublattice_offset_ev: float


def bipartite_supercell(
    nx: int, ny: int, nz: int, lattice_constant_m: float, sublattice_offset_ev: float
) -> tuple[np.ndarray, np.ndarray]:
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    positions = np.zeros((nx * ny * nz, 3), dtype=float)
    onsite = np.zeros(nx * ny * nz, dtype=float)
    idx = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                positions[idx] = np.array([i, j, k], dtype=float) * lattice_constant_m
                parity = (i + j + k) % 2
                onsite[idx] = sublattice_offset_ev if parity == 0 else -sublattice_offset_ev
                idx += 1
    return positions, onsite


def build_supercell(
    lattice_vectors: np.ndarray,
    motif_frac: np.ndarray,
    motif_species: list[str],
    repeats: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must be 3x3")
    motif_frac = np.asarray(motif_frac, dtype=float)
    if motif_frac.ndim != 2 or motif_frac.shape[1] != 3:
        raise ValueError("motif_frac must be (n, 3)")
    if len(motif_species) != motif_frac.shape[0]:
        raise ValueError("motif_species length must match motif_frac")
    nx, ny, nz = repeats
    if min(nx, ny, nz) < 1:
        raise ValueError("repeats must be >= 1")

    frac_positions = []
    species: list[str] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell = np.array([i, j, k], dtype=float)
                for frac, sp in zip(motif_frac, motif_species):
                    frac_positions.append((cell + frac) / np.array([nx, ny, nz]))
                    species.append(sp)
    frac_positions = np.asarray(frac_positions, dtype=float)
    supercell_vectors = np.array(
        [
            lattice_vectors[0] * nx,
            lattice_vectors[1] * ny,
            lattice_vectors[2] * nz,
        ],
        dtype=float,
    )
    cart_positions = frac_positions @ supercell_vectors
    return frac_positions, cart_positions, species


def add_sites(
    frac_positions: np.ndarray,
    species: list[str],
    new_frac: np.ndarray,
    new_species: list[str],
) -> tuple[np.ndarray, list[str]]:
    new_frac = np.asarray(new_frac, dtype=float)
    if new_frac.ndim != 2 or new_frac.shape[1] != 3:
        raise ValueError("new_frac must be (n, 3)")
    if len(new_species) != new_frac.shape[0]:
        raise ValueError("new_species length must match new_frac")
    combined = np.vstack([frac_positions, new_frac])
    species_out = list(species) + list(new_species)
    return combined, species_out


def neighbor_pairs(
    frac_positions: np.ndarray,
    supercell_vectors: np.ndarray,
    cutoff_m: float,
) -> list[tuple[int, int, float]]:
    i, j = np.triu_indices(frac_positions.shape[0], k=1)
    delta = frac_positions[i] - frac_positions[j]
    delta -= np.round(delta)  # minimum image under periodic boundaries
    dist = np.linalg.norm(delta @ supercell_vectors, axis=1)
    keep = dist <= cutoff_m
    return list(zip(i[keep].tolist(), j[keep].tolist(), dist[keep].tolist()))


def build_hamiltonian_from_pairs(
    onsite_ev: np.ndarray,
    pairs: list[tuple[int, int, float]],
    hopping_fn,
) -> np.ndarray:
    n = onsite_ev.size
    h = np.zeros((n, n), dtype=float)
    np.fill_diagonal(h, onsite_ev)
    for i, j, dist in pairs:
        hop = float(hopping_fn(dist))
        h[i, j] = hop
        h[j, i] = hop
    return h


def supercell_neighbors(nx: int, ny: int, nz: int, periodic: bool = True) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = i * ny * nz + j * nz + k
                for di, dj, dk in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ):
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    if periodic:
                        ni %= nx
                        nj %= ny
                        nk %= nz
                    else:
                        if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                            continue
                    n_idx = ni * ny * nz + nj * nz + nk
                    if idx < n_idx:
                        neighbors.append((idx, n_idx))
    return neighbors


def apply_dopants(
    onsite_ev: np.ndarray,
    dopant_indices: np.ndarray,
    delta_ev: float,
) -> np.ndarray:
    updated = onsite_ev.copy()
    updated[dopant_indices] += delta_ev
    return updated


def build_hamiltonian(
    onsite_ev: np.ndarray,
    neighbors: list[tuple[int, int]],
    hopping_ev: float,
) -> np.ndarray:
    n = onsite_ev.size
    h = np.zeros((n, n), dtype=float)
    np.fill_diagonal(h, onsite_ev)
    for i, j in neighbors:
        h[i, j] = hopping_ev
        h[j, i] = hopping_ev
    return h


def solve_schrodinger(hamiltonian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    energies, states = np.linalg.eigh(hamiltonian)
    return energies, states


def half_filling_fermi_level(energies: np.ndarray) -> float:
    n = energies.size
    if n % 2 == 0:
        return 0.5 * (energies[n // 2 - 1] + energies[n // 2])
    return energies[n // 2]


def band_gap_from_half_filling(energies: np.ndarray) -> tuple[float, float, float]:
    n = energies.size
    n_occ = n // 2
    v_max = energies[n_occ - 1]
    c_min = energies[n_occ]
    return float(c_min - v_max), float(v_max), float(c_min)


def wavefunction_density(
    state: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    return np.abs(state.reshape(nx, ny, nz)) ** 2


def wavefunction_density_grid(
    state: np.ndarray,
    frac_positions: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    if frac_positions.shape[0] != state.size:
        raise ValueError("frac_positions and state size must match")
    nx, ny, nz = grid_shape
    grid = np.zeros((nx, ny, nz), dtype=float)
    frac = np.mod(frac_positions, 1.0)
    ix = np.clip((frac[:, 0] * nx).astype(int), 0, nx - 1)
    iy = np.clip((frac[:, 1] * ny).astype(int), 0, ny - 1)
    iz = np.clip((frac[:, 2] * nz).astype(int), 0, nz - 1)
    weights = np.abs(state) ** 2
    for i, j, k, w in zip(ix, iy, iz, weights):
        grid[i, j, k] += w
    return grid


def density_of_states(
    energies: np.ndarray, grid: np.ndarray, sigma_ev: float
) -> np.ndarray:
    diff = grid[:, None] - energies[None, :]
    weights = np.exp(-0.5 * (diff / sigma_ev) ** 2)
    norm = sigma_ev * np.sqrt(2.0 * np.pi)
    return np.sum(weights / norm, axis=1)
