from __future__ import annotations

import numpy as np

from .stats import KB, Q, thermal_voltage_v

_EPS0_F_PER_CM = 8.854_187_8128e-14


def built_in_potential_v(
    temperature_k: float,
    n_cm3: float,
    p_cm3: float,
    ni_cm3: float,
) -> float:
    return thermal_voltage_v(temperature_k) * np.log((n_cm3 * p_cm3) / (ni_cm3**2))


def depletion_width_cm(
    temperature_k: float,
    n_cm3: float,
    p_cm3: float,
    ni_cm3: float,
    epsilon_rel: float = 11.7,
) -> tuple[float, float, float]:
    v_bi = built_in_potential_v(temperature_k, n_cm3, p_cm3, ni_cm3)
    eps = _EPS0_F_PER_CM * epsilon_rel
    width = np.sqrt(2.0 * eps * v_bi / Q * (1.0 / n_cm3 + 1.0 / p_cm3))
    x_n = width * p_cm3 / (n_cm3 + p_cm3)
    x_p = width * n_cm3 / (n_cm3 + p_cm3)
    return float(width), float(x_n), float(x_p)


def diode_saturation_current_density_a_per_cm2(
    ni_cm3: float,
    dn_cm2_s: float,
    dp_cm2_s: float,
    l_n_cm: float,
    l_p_cm: float,
    n_cm3: float,
    p_cm3: float,
) -> float:
    term_n = dp_cm2_s / (l_p_cm * n_cm3)
    term_p = dn_cm2_s / (l_n_cm * p_cm3)
    return Q * ni_cm3**2 * (term_n + term_p)


def shockley_diode_current_density_a_per_cm2(
    voltages_v: np.ndarray,
    temperature_k: float,
    j0_a_per_cm2: float,
    ideality: float = 1.0,
) -> np.ndarray:
    v_t = thermal_voltage_v(temperature_k) * ideality
    exponent = np.clip(voltages_v / v_t, -100.0, 100.0)
    return j0_a_per_cm2 * (np.exp(exponent) - 1.0)

