"""Physics sanity checks for the semiconductor models."""

import numpy as np

from quantsim import (
    BrillouinZonePath,
    EmpiricalPseudopotential,
    Lattice,
    build_supercell,
    neighbor_pairs,
    shockley_queisser_limit,
    solve_epm_band_structure,
)


def test_fcc_kpath_lands_on_cartesian_high_symmetry_points():
    a = 5.431e-10
    lattice = Lattice.fcc(a)
    points = BrillouinZonePath.for_lattice("fcc").points
    cart = {k: v @ lattice.reciprocal().vectors / (2 * np.pi / a) for k, v in points.items()}
    np.testing.assert_allclose(cart["X"], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(cart["L"], [0.5, 0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(cart["W"], [0.5, 1, 0], atol=1e-12)
    np.testing.assert_allclose(cart["K"], [0.75, 0.75, 0], atol=1e-12)


def test_silicon_epm_reproduces_cohen_bergstresser():
    model = EmpiricalPseudopotential.silicon()
    lattice = Lattice.fcc(model.lattice_constant_m)
    g_cut = 2 * np.pi / model.lattice_constant_m * np.sqrt(11.5)
    r = solve_epm_band_structure(lattice, model, BrillouinZonePath.for_lattice("fcc"), g_cut, n_bands=8, points_per_segment=20)
    gamma = r.energies_ev[0] - r.vbm_ev
    assert abs(r.vbm_ev - r.energies_ev[0, 3]) < 5e-3  # VBM at Gamma (meV basis-size jitter)
    assert abs(r.distances[r.cbm_k_index] / r.ticks[1] - 0.85) < 0.06  # CBM ~85% along Gamma-X
    assert 0.6 < r.gap_ev < 1.2 and not r.is_direct_gap
    assert abs(gamma[4] - 3.4) < 0.3  # Gamma25' -> Gamma15
    assert abs(gamma[0] + 12.5) < 0.5  # valence band width
    x = r.energies_ev[20]
    assert abs(x[4] - x[5]) < 1e-6  # X1 is doubly degenerate (non-symmorphic diamond)


def test_shockley_queisser_limit_for_silicon():
    sq = shockley_queisser_limit(1.12)
    assert abs(sq.p_in - 1317) < 15  # 5778 K blackbody at 1 AU ~ solar constant
    assert 0.29 < sq.efficiency < 0.32
    assert 400 < sq.j_sc < 600  # A/m^2


def test_diamond_supercell_has_four_nearest_neighbours():
    a = 5.431e-10
    vecs = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    frac, _, _ = build_supercell(vecs, np.array([[0, 0, 0], [0.25, 0.25, 0.25]]), ["A", "B"], (3, 3, 3))
    pairs = neighbor_pairs(frac, vecs * 3, 1.1 * a * np.sqrt(3) / 4)
    assert len(pairs) == 2 * frac.shape[0]  # 4 bonds per atom, each pair counted once


def test_fermi_level_stays_inside_gap_for_heavy_doping():
    from quantsim import fermi_level_si

    gap = 1.12
    assert 0.0 < fermi_level_si(1e19, 0.0, gap) < gap
    assert abs(fermi_level_si(1e16, 0.0, gap) - (gap - 0.0259 * np.log(2.8e19 / 1e16))) < 2e-3
    assert 0.0 < fermi_level_si(0.0, 1e18, gap) < 0.2
