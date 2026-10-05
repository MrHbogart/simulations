from __future__ import annotations

import numpy as np

_Q = 1.602_176_634e-19  # C
_KB = 1.380_649e-23  # J/K


def fermi_dirac(energies_ev: np.ndarray, fermi_ev: float, temperature_k: float) -> np.ndarray:
    kt_ev = (_KB * temperature_k) / _Q
    return 1.0 / (1.0 + np.exp((energies_ev - fermi_ev) / kt_ev))


def intrinsic_fermi_level_ev(band_gap_ev: float) -> float:
    return band_gap_ev / 2.0


def fermi_level_si(
    n_cm3: float,
    p_cm3: float,
    band_gap_ev: float,
    temperature_k: float = 300.0,
    nc300_cm3: float = 2.8e19,
    nv300_cm3: float = 1.04e19,
) -> float:
    """Estimate Fermi level (eV) relative to the valence band edge for Si.

    Boltzmann statistics referenced to the band edges, E_F = E_c - kT ln(N_c / n)
    (or E_v + kT ln(N_v / p)). Unlike E_i + kT ln(n / n_i) with E_i = E_g / 2, this
    stays consistent with the band gap passed in and only reaches E_c when n -> N_c.
    """
    kt_ev = (_KB * temperature_k) / _Q
    nc = nc300_cm3 * (temperature_k / 300.0) ** 1.5
    nv = nv300_cm3 * (temperature_k / 300.0) ** 1.5

    n_eff = n_cm3 - p_cm3
    if n_eff > 0.0:
        return band_gap_ev - kt_ev * np.log(nc / n_eff)
    if n_eff < 0.0:
        return kt_ev * np.log(nv / -n_eff)
    return intrinsic_fermi_level_ev(band_gap_ev) + 0.5 * kt_ev * np.log(nv / nc)
