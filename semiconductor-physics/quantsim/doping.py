from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DopantCase:
    label: str
    n_cm3: float
    p_cm3: float


def band_gap_narrowing_si(n_cm3: float, p_cm3: float, temperature_k: float = 300.0) -> float:
    """Return an empirical band-gap narrowing (eV) for doped Si.

    This is a compact, literature-inspired model that yields ~0.05-0.12 eV
    narrowing for 1e19-1e20 cm^-3 doping, which is typical for Si emitters.
    """
    if n_cm3 <= 0.0 and p_cm3 <= 0.0:
        return 0.0

    def _narrowing_term(conc: float) -> float:
        if conc <= 1.0e16:
            return 0.0
        x = np.log10(conc / 1.0e17)
        return 0.018 * x + 0.005 * x * x

    delta = _narrowing_term(n_cm3) + _narrowing_term(p_cm3)

    # Mild temperature dependence (smaller narrowing at higher T).
    delta *= (300.0 / temperature_k) ** 0.2
    return float(max(delta, 0.0))
