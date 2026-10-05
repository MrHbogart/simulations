from __future__ import annotations

import numpy as np

Q = 1.602_176_634e-19  # C
KB = 1.380_649e-23  # J/K


def thermal_voltage_v(temperature_k: float) -> float:
    return (KB * temperature_k) / Q


def varshni_gap_ev(
    temperature_k: np.ndarray | float,
    eg0_ev: float,
    alpha_ev_per_k: float,
    beta_k: float,
) -> np.ndarray:
    t_k = np.asarray(temperature_k, dtype=float)
    return eg0_ev - (alpha_ev_per_k * t_k**2) / (t_k + beta_k)


def effective_density_of_states_cm3(n300_cm3: float, temperature_k: np.ndarray | float) -> np.ndarray:
    t_k = np.asarray(temperature_k, dtype=float)
    return n300_cm3 * (t_k / 300.0) ** 1.5


def intrinsic_concentration_cm3(
    band_gap_ev: np.ndarray | float,
    nc_cm3: np.ndarray | float,
    nv_cm3: np.ndarray | float,
    temperature_k: np.ndarray | float,
) -> np.ndarray:
    t_k = np.asarray(temperature_k, dtype=float)
    kt_ev = (KB * t_k) / Q
    return np.sqrt(np.asarray(nc_cm3) * np.asarray(nv_cm3)) * np.exp(
        -np.asarray(band_gap_ev) / (2.0 * kt_ev)
    )


def fermi_level_nondegenerate_ev(
    band_gap_ev: float,
    temperature_k: float,
    n_cm3: float,
    p_cm3: float,
    ni_cm3: float,
) -> float:
    kt_ev = (KB * temperature_k) / Q
    e_i = 0.5 * band_gap_ev
    n_eff = max(n_cm3 - p_cm3, 0.0)
    p_eff = max(p_cm3 - n_cm3, 0.0)
    if n_eff > 0.0:
        return e_i + kt_ev * np.log(n_eff / ni_cm3)
    if p_eff > 0.0:
        return e_i - kt_ev * np.log(p_eff / ni_cm3)
    return e_i


def carrier_concentrations_from_doping(
    n_d_cm3: np.ndarray | float,
    p_a_cm3: np.ndarray | float,
    ni_cm3: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    n_d = np.asarray(n_d_cm3, dtype=float)
    p_a = np.asarray(p_a_cm3, dtype=float)
    ni = np.asarray(ni_cm3, dtype=float)
    net = n_d - p_a
    n_cm3 = np.where(net >= 0.0, np.maximum(n_d, ni), ni**2 / np.maximum(p_a, 1.0))
    p_cm3 = (ni**2) / np.maximum(n_cm3, 1.0)
    return n_cm3, p_cm3


def mobility_power_law(mu_300_cm2_v_s: float, temperature_k: np.ndarray | float, exponent: float = -1.5) -> np.ndarray:
    t_k = np.asarray(temperature_k, dtype=float)
    return mu_300_cm2_v_s * (t_k / 300.0) ** exponent


def conductivity_s_per_cm(
    n_cm3: np.ndarray | float,
    p_cm3: np.ndarray | float,
    mu_n_cm2_v_s: np.ndarray | float,
    mu_p_cm2_v_s: np.ndarray | float,
) -> np.ndarray:
    return Q * (np.asarray(n_cm3) * np.asarray(mu_n_cm2_v_s) + np.asarray(p_cm3) * np.asarray(mu_p_cm2_v_s))

