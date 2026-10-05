from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_Q = 1.602_176_634e-19  # C
_KB = 1.380_649e-23  # J/K
_H = 6.626_070_15e-34  # J*s
_C = 299_792_458.0  # m/s

_R_SUN = 6.9634e8  # m
_D_SUN = 1.495_978_707e11  # m


@dataclass(frozen=True)
class ShockleyQueisserResult:
    voltages: np.ndarray
    currents: np.ndarray
    power: np.ndarray
    efficiency: float
    v_mpp: float
    j_mpp: float
    p_in: float
    j_sc: float
    v_oc: float


def _photon_flux_spectral(E_ev: np.ndarray, T: float, solid_angle: float) -> np.ndarray:
    E_j = E_ev * _Q
    exponent = E_j / (_KB * T)
    exponent = np.clip(exponent, 1.0e-9, 1.0e4)
    prefactor = 2.0 * E_j**2 / (_H**3 * _C**2)
    return solid_angle * prefactor / (np.exp(exponent) - 1.0)


def _solar_solid_angle() -> float:
    return np.pi * (_R_SUN / _D_SUN) ** 2


def shockley_queisser_limit(
    band_gap_ev: float,
    cell_temperature_k: float = 300.0,
    sun_temperature_k: float = 5778.0,
    n_points: int = 4000,
) -> ShockleyQueisserResult:
    if band_gap_ev <= 0.0:
        raise ValueError("band_gap_ev must be positive")

    e_max = 4.0
    energies = np.linspace(1.0e-4, e_max, n_points)

    sun_solid_angle = _solar_solid_angle()
    sun_flux = _photon_flux_spectral(energies, sun_temperature_k, sun_solid_angle)
    cell_flux = _photon_flux_spectral(energies, cell_temperature_k, np.pi)

    mask = energies >= band_gap_ev
    if not np.any(mask):
        raise ValueError("band_gap_ev exceeds integration range")

    # Fluxes are per joule; energies are in eV, so dE_J = _Q * dE_eV.
    j_sc = _Q * _Q * np.trapezoid(sun_flux[mask], energies[mask])
    j_0 = _Q * _Q * np.trapezoid(cell_flux[mask], energies[mask])

    p_in = _Q * np.trapezoid(sun_flux * energies * _Q, energies)

    voltages = np.linspace(0.0, band_gap_ev, 300)
    currents = j_sc - j_0 * (np.exp(_Q * voltages / (_KB * cell_temperature_k)) - 1.0)
    power = voltages * currents

    idx = int(np.argmax(power))
    v_mpp = float(voltages[idx])
    j_mpp = float(currents[idx])
    efficiency = float(power[idx] / p_in)

    v_oc = float(( _KB * cell_temperature_k / _Q) * np.log(1.0 + j_sc / j_0))

    return ShockleyQueisserResult(
        voltages=voltages,
        currents=currents,
        power=power,
        efficiency=efficiency,
        v_mpp=v_mpp,
        j_mpp=j_mpp,
        p_in=float(p_in),
        j_sc=float(j_sc),
        v_oc=v_oc,
    )
