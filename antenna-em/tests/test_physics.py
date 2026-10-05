"""Physics sanity checks for the antenna field solvers."""

import numpy as np

from emsim import DipoleAntenna, WireAntenna, dipole_fields, make_straight_wire, resample_polyline, wire_antenna_fields
from emsim.antenna import mu0
from emsim.viz import poynting_vector

ANTENNA = DipoleAntenna(length=0.01, radius=1e-4, current_amp=1.0, frequency=300e6)


def _curl_e_over_minus_j_omega_mu(point, h=1e-4):
    """Faraday's law, H = curl(E) / (-j omega mu0), via central differences."""
    def e(dx=0.0, dy=0.0, dz=0.0):
        x, y, z = (np.array([c]) for c in point + np.array([dx, dy, dz]))
        return np.array([f[0] for f in dipole_fields(ANTENNA, x, y, z)[:3]])

    d = [(e(**{k: h}) - e(**{k: -h})) / (2 * h) for k in ("dx", "dy", "dz")]  # d[j][i] = dE_i/dx_j
    curl = np.array([d[1][2] - d[2][1], d[2][0] - d[0][2], d[0][1] - d[1][0]])
    return curl / (-1j * ANTENNA.omega * mu0())


def test_h_field_satisfies_faraday_in_near_and_far_zone():
    for r in (0.05, 0.3, 5.0):  # kr = 0.3, 1.9, 31
        point = r * np.array([0.6, 0.0, 0.8])
        x, y, z = (np.array([c]) for c in point)
        h = np.array([f[0] for f in dipole_fields(ANTENNA, x, y, z)[3:]])
        np.testing.assert_allclose(h, _curl_e_over_minus_j_omega_mu(point), rtol=1e-4)


def test_radiated_power_matches_hertzian_formula():
    # P = eta0 (k l)^2 |I|^2 / (12 pi) for a short uniform-current dipole.
    r, theta = 50.0, np.linspace(1e-3, np.pi - 1e-3, 2001)
    x, y, z = r * np.sin(theta), np.zeros_like(theta), r * np.cos(theta)
    s = poynting_vector(*dipole_fields(ANTENNA, x, y, z))
    s_r = (s[0] * x + s[2] * z) / r
    power = np.trapezoid(s_r * 2 * np.pi * r**2 * np.sin(theta), theta)
    eta0 = 376.730313
    expected = eta0 * (ANTENNA.k * ANTENNA.length) ** 2 / (12 * np.pi)
    assert abs(power - expected) / expected < 1e-3


def test_resampled_path_gives_same_far_field():
    straight = make_straight_wire(1.0, n_points=201)
    coarse = WireAntenna(path=straight[[0, 100, 200]], radius=1e-3, current_amp=1.0, frequency=300e6)
    fine = WireAntenna(path=resample_polyline(straight[[0, 100, 200]], 201), radius=1e-3, current_amp=1.0, frequency=300e6)
    reference = WireAntenna(path=straight, radius=1e-3, current_amp=1.0, frequency=300e6)
    pt = (np.array([20.0]), np.array([0.0]), np.array([0.0]))
    ez = lambda ant: wire_antenna_fields(ant, *pt)[2][0]
    assert abs(ez(fine) - ez(reference)) < 1e-9 * abs(ez(reference))
    assert len(fine.path) == 201 and len(coarse.path) == 3
