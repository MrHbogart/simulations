from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bz import BrillouinZonePath
from .lattice import Lattice

_HBAR = 1.054_571_817e-34  # J*s
_ELECTRON_MASS = 9.109_383_7015e-31  # kg
_EV_J = 1.602_176_634e-19  # J per eV
_RY_EV = 13.605_693_122_994  # eV per Rydberg

__all__ = [
    "EmpiricalPseudopotential",
    "DopantPerturbation",
    "EpmResult",
    "solve_epm_band_structure",
    "calibrate_form_factors_to_gap",
    "apply_scissor_correction",
]


@dataclass(frozen=True)
class EmpiricalPseudopotential:
    name: str
    lattice_constant_m: float
    vs_form_factors_ev: dict[int, float]
    va_form_factors_ev: dict[int, float]
    structure_basis: np.ndarray

    @staticmethod
    def silicon() -> "EmpiricalPseudopotential":
        # Cohen-Bergstresser (1966) form factors for Si, published in Rydberg.
        # Vs, Va indexed by |G|^2 = 3, 8, 11 in units (2*pi/a)^2.
        return EmpiricalPseudopotential(
            name="Si",
            lattice_constant_m=5.431e-10,
            vs_form_factors_ev={3: -0.21 * _RY_EV, 8: 0.04 * _RY_EV, 11: 0.08 * _RY_EV},
            va_form_factors_ev={3: 0.0, 8: 0.0, 11: 0.0},
            structure_basis=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
        )


@dataclass(frozen=True)
class DopantPerturbation:
    element: str
    concentration_cm3: float
    delta_vs_form_factors_ev: dict[int, float]
    delta_va_form_factors_ev: dict[int, float] | None = None


@dataclass(frozen=True)
class EpmResult:
    distances: np.ndarray
    energies_ev: np.ndarray
    ticks: list[float]
    labels: list[str]
    gap_ev: float
    vbm_ev: float
    cbm_ev: float
    vbm_k_index: int
    cbm_k_index: int
    direct_gap_ev: float
    is_direct_gap: bool


def _form_factor_from_g2(g2: float, form_factors_ev: dict[int, float]) -> float:
    key = int(round(g2))
    return form_factors_ev.get(key, 0.0)


def _generate_g_vectors(lattice: Lattice, g_cut: float) -> np.ndarray:
    bvecs = lattice.reciprocal().vectors
    b_norms = np.linalg.norm(bvecs, axis=1)
    max_index = int(np.ceil(g_cut / min(b_norms))) + 1
    indices = range(-max_index, max_index + 1)
    g_list = []
    for h in indices:
        for k in indices:
            for l in indices:
                g = h * bvecs[0] + k * bvecs[1] + l * bvecs[2]
                if np.linalg.norm(g) <= g_cut * (1.0 + 1.0e-12):
                    g_list.append(g)
    return np.asarray(g_list)


def _effective_form_factors(
    base: EmpiricalPseudopotential, dopants: list[DopantPerturbation]
) -> tuple[dict[int, float], dict[int, float]]:
    vs_form_factors = dict(base.vs_form_factors_ev)
    va_form_factors = dict(base.va_form_factors_ev)
    if not dopants:
        return vs_form_factors, va_form_factors
    # Atomic density for Si (approx) to convert cm^-3 -> fraction.
    atomic_density_cm3 = 5.0e22
    for dopant in dopants:
        fraction = dopant.concentration_cm3 / atomic_density_cm3
        for g2, delta in dopant.delta_vs_form_factors_ev.items():
            vs_form_factors[g2] = vs_form_factors.get(g2, 0.0) + fraction * delta
        if dopant.delta_va_form_factors_ev:
            for g2, delta in dopant.delta_va_form_factors_ev.items():
                va_form_factors[g2] = va_form_factors.get(g2, 0.0) + fraction * delta
    return vs_form_factors, va_form_factors


def solve_epm_band_structure(
    lattice: Lattice,
    model: EmpiricalPseudopotential,
    path: BrillouinZonePath,
    g_cut: float,
    n_bands: int = 8,
    points_per_segment: int = 40,
    dopants: list[DopantPerturbation] | None = None,
    scale_factor: float = 1.0,
    valence_band_index: int = 4,
    target_gap_ev: float | None = None,
    dopant_gap_shift_ev: float = 0.0,
) -> EpmResult:
    dopants = dopants or []
    vs_form_factors, va_form_factors = _effective_form_factors(model, dopants)
    if scale_factor != 1.0:
        vs_form_factors = {k: v * scale_factor for k, v in vs_form_factors.items()}
        va_form_factors = {k: v * scale_factor for k, v in va_form_factors.items()}
    k_frac = path.sample(points_per_segment)
    k_cart = k_frac @ lattice.reciprocal().vectors
    # Basis is |k + G| <= g_cut at each k, so the set respects the symmetry of k.
    k_max = float(np.max(np.linalg.norm(k_cart, axis=1)))
    g_pool = _generate_g_vectors(lattice, g_cut + k_max)
    distances = _cumulative_distances(k_cart)

    ticks = [0.0]
    labels = [path.labels[0]]
    index = 0
    for _ in path.segments():
        index += points_per_segment
        ticks.append(distances[index])
        labels.append(path.labels[len(labels)])

    basis_frac = model.structure_basis
    energies = []
    for k_vec in k_cart:
        g_vectors = g_pool[np.linalg.norm(k_vec + g_pool, axis=1) <= g_cut + 1.0e-12 * g_cut]
        if g_vectors.shape[0] < n_bands:
            raise ValueError("Too few plane waves for n_bands; increase g_cut")
        h_mat = _build_hamiltonian(
            k_vec,
            g_vectors,
            lattice,
            vs_form_factors,
            va_form_factors,
            model.lattice_constant_m,
            model.structure_basis,
        )
        vals = np.linalg.eigvalsh(h_mat)
        energies.append(vals[:n_bands])
    energies_ev = np.asarray(energies)

    vbm_ev, cbm_ev, vbm_k, cbm_k = _band_edges_from_bands(
        energies_ev, valence_band_index
    )
    gap_ev = cbm_ev - vbm_ev
    if target_gap_ev is not None and np.isfinite(gap_ev):
        energies_ev, vbm_ev, cbm_ev, gap_ev = apply_scissor_correction(
            energies_ev, valence_band_index, target_gap_ev
        )
        vbm_ev, cbm_ev, vbm_k, cbm_k = _band_edges_from_bands(
            energies_ev, valence_band_index
        )
        gap_ev = cbm_ev - vbm_ev
    if dopant_gap_shift_ev != 0.0 and np.isfinite(gap_ev):
        energies_ev, vbm_ev, cbm_ev, gap_ev = apply_scissor_correction(
            energies_ev, valence_band_index, gap_ev + dopant_gap_shift_ev
        )
        vbm_ev, cbm_ev, vbm_k, cbm_k = _band_edges_from_bands(
            energies_ev, valence_band_index
        )
        gap_ev = cbm_ev - vbm_ev

    direct_gap_ev = _direct_gap_from_bands(energies_ev, valence_band_index)
    is_direct = vbm_k == cbm_k
    return EpmResult(
        distances=distances,
        energies_ev=energies_ev,
        ticks=ticks,
        labels=labels,
        gap_ev=gap_ev,
        vbm_ev=vbm_ev,
        cbm_ev=cbm_ev,
        vbm_k_index=vbm_k,
        cbm_k_index=cbm_k,
        direct_gap_ev=direct_gap_ev,
        is_direct_gap=is_direct,
    )


def _build_hamiltonian(
    k_vec: np.ndarray,
    g_vectors: np.ndarray,
    lattice: Lattice,
    vs_form_factors: dict[int, float],
    va_form_factors: dict[int, float],
    lattice_constant_m: float,
    structure_basis: np.ndarray,
) -> np.ndarray:
    k_plus_g = k_vec + g_vectors
    kinetic = _HBAR**2 * np.sum(k_plus_g**2, axis=1) / (2.0 * _ELECTRON_MASS) / _EV_J

    g_diff = g_vectors[:, None, :] - g_vectors[None, :, :]
    g2 = np.rint(_g2_norm(g_diff, lattice_constant_m)).astype(int)
    vs = np.vectorize(lambda n: vs_form_factors.get(n, 0.0), otypes=[float])(g2)
    va = np.vectorize(lambda n: va_form_factors.get(n, 0.0), otypes=[float])(g2)

    # Origin at the bond centre: atoms sit at +/- tau, tau = half the basis separation
    # (a/8 (1,1,1) for diamond), giving S_s = cos(G.tau) and S_a = sin(G.tau).
    basis_cart = lattice.cartesian_from_fractional(np.asarray(structure_basis, dtype=float))
    tau = 0.5 * (basis_cart[1] - basis_cart[0])
    phase = g_diff @ tau
    h_mat = vs * np.cos(phase) + 1j * va * np.sin(phase)
    np.fill_diagonal(h_mat, kinetic)
    return h_mat


def _g2_norm(g_vec: np.ndarray, lattice_constant_m: float) -> float:
    a = lattice_constant_m
    scale = (a / (2.0 * np.pi)) ** 2
    return scale * np.sum(g_vec * g_vec, axis=-1)


def _band_edges_from_bands(
    energies_ev: np.ndarray, valence_band_index: int
) -> tuple[float, float, int, int]:
    if energies_ev.ndim != 2:
        return float("nan"), float("nan"), -1, -1
    if valence_band_index < 1 or valence_band_index >= energies_ev.shape[1]:
        return float("nan"), float("nan"), -1, -1
    vbm_band = energies_ev[:, valence_band_index - 1]
    cbm_band = energies_ev[:, valence_band_index]
    vbm_k = int(np.argmax(vbm_band))
    cbm_k = int(np.argmin(cbm_band))
    vbm = vbm_band[vbm_k]
    cbm = cbm_band[cbm_k]
    return float(vbm), float(cbm), vbm_k, cbm_k


def _direct_gap_from_bands(energies_ev: np.ndarray, valence_band_index: int) -> float:
    if energies_ev.ndim != 2:
        return float("nan")
    if valence_band_index < 1 or valence_band_index >= energies_ev.shape[1]:
        return float("nan")
    v_band = energies_ev[:, valence_band_index - 1]
    c_band = energies_ev[:, valence_band_index]
    return float(np.min(c_band - v_band))


def apply_scissor_correction(
    energies_ev: np.ndarray, valence_band_index: int, target_gap_ev: float
) -> tuple[np.ndarray, float, float, float]:
    if target_gap_ev <= 0.0:
        raise ValueError("target_gap_ev must be positive")
    vbm, cbm, _, _ = _band_edges_from_bands(energies_ev, valence_band_index)
    gap = cbm - vbm
    if not np.isfinite(gap):
        return energies_ev, vbm, cbm, gap
    shift = target_gap_ev - gap
    corrected = energies_ev.copy()
    corrected[:, valence_band_index:] += shift
    vbm_c, cbm_c, _, _ = _band_edges_from_bands(corrected, valence_band_index)
    return corrected, vbm_c, cbm_c, cbm_c - vbm_c


def calibrate_form_factors_to_gap(
    lattice: Lattice,
    model: EmpiricalPseudopotential,
    path: BrillouinZonePath,
    g_cut: float,
    target_gap_ev: float,
    n_bands: int = 8,
    points_per_segment: int = 24,
    dopants: list[DopantPerturbation] | None = None,
    max_iter: int = 20,
    valence_band_index: int = 4,
) -> tuple[float, EpmResult]:
    dopants = dopants or []
    low, high = 0.2, 3.0
    best_scale = 1.0
    best_gap = None
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        result = solve_epm_band_structure(
            lattice=lattice,
            model=model,
            path=path,
            g_cut=g_cut,
            n_bands=n_bands,
            points_per_segment=points_per_segment,
            dopants=dopants,
            scale_factor=mid,
            valence_band_index=valence_band_index,
        )
        gap = result.gap_ev
        if not np.isfinite(gap):
            low = mid
            continue
        best_scale = mid
        best_gap = gap
        if gap < target_gap_ev:
            low = mid
        else:
            high = mid
    if best_gap is None:
        raise ValueError("Calibration failed to find a finite gap")
    final = solve_epm_band_structure(
        lattice=lattice,
        model=model,
        path=path,
        g_cut=g_cut,
        n_bands=n_bands,
        points_per_segment=points_per_segment,
        dopants=dopants,
        scale_factor=best_scale,
        valence_band_index=valence_band_index,
    )
    return best_scale, final


def _cumulative_distances(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])
