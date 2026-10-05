from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    lattice_constant_m: float
    eg0_ev: float
    varshni_alpha_ev_per_k: float
    varshni_beta_k: float
    nc300_cm3: float
    nv300_cm3: float
    mu_n_300_cm2_v_s: float
    mu_p_300_cm2_v_s: float


def silicon() -> Material:
    return Material(
        name="Si",
        lattice_constant_m=5.431e-10,
        eg0_ev=1.17,
        varshni_alpha_ev_per_k=4.73e-4,
        varshni_beta_k=636.0,
        nc300_cm3=2.8e19,
        nv300_cm3=1.04e19,
        mu_n_300_cm2_v_s=1350.0,
        mu_p_300_cm2_v_s=480.0,
    )


def gaas() -> Material:
    return Material(
        name="GaAs",
        lattice_constant_m=5.653e-10,
        eg0_ev=1.52,
        varshni_alpha_ev_per_k=5.41e-4,
        varshni_beta_k=204.0,
        nc300_cm3=4.7e17,
        nv300_cm3=7.0e18,
        mu_n_300_cm2_v_s=8500.0,
        mu_p_300_cm2_v_s=400.0,
    )


def germanium() -> Material:
    return Material(
        name="Ge",
        lattice_constant_m=5.658e-10,
        eg0_ev=0.743,
        varshni_alpha_ev_per_k=4.77e-4,
        varshni_beta_k=235.0,
        nc300_cm3=1.04e19,
        nv300_cm3=6.0e18,
        mu_n_300_cm2_v_s=3900.0,
        mu_p_300_cm2_v_s=1900.0,
    )

