"""Core package for crystal and band-structure scaffolding."""

from .lattice import BasisAtom, Crystal, Lattice
from .bz import BrillouinZonePath
from .plotting import (
    plot_band_structure,
    plot_bands,
    plot_crystal,
    plot_kpath,
    plot_reciprocal_lattice,
)
from .bands import Dopant, SemiconductorModel, SemiconductorParams, sample_band_structure
from .epm import (
    DopantPerturbation,
    EmpiricalPseudopotential,
    apply_scissor_correction,
    calibrate_form_factors_to_gap,
    solve_epm_band_structure,
)
from .doping import DopantCase, band_gap_narrowing_si
from .diode import (
    built_in_potential_v,
    depletion_width_cm,
    diode_saturation_current_density_a_per_cm2,
    shockley_diode_current_density_a_per_cm2,
)
from .fermi import fermi_dirac, fermi_level_si, intrinsic_fermi_level_ev
from .materials import Material, gaas, germanium, silicon
from .stats import (
    conductivity_s_per_cm,
    effective_density_of_states_cm3,
    fermi_level_nondegenerate_ev,
    intrinsic_concentration_cm3,
    mobility_power_law,
    thermal_voltage_v,
    varshni_gap_ev,
)
from .solar import ShockleyQueisserResult, shockley_queisser_limit
from .tightbinding import (
    TightBindingModel,
    add_sites,
    apply_dopants,
    band_gap_from_half_filling,
    bipartite_supercell,
    build_hamiltonian_from_pairs,
    build_supercell,
    build_hamiltonian,
    density_of_states,
    half_filling_fermi_level,
    solve_schrodinger,
    neighbor_pairs,
    supercell_neighbors,
    wavefunction_density,
    wavefunction_density_grid,
)

__all__ = [
    "BasisAtom",
    "Crystal",
    "Lattice",
    "BrillouinZonePath",
    "plot_crystal",
    "plot_kpath",
    "plot_reciprocal_lattice",
    "plot_band_structure",
    "plot_bands",
    "Dopant",
    "SemiconductorParams",
    "SemiconductorModel",
    "sample_band_structure",
    "EmpiricalPseudopotential",
    "DopantPerturbation",
    "apply_scissor_correction",
    "calibrate_form_factors_to_gap",
    "solve_epm_band_structure",
    "DopantCase",
    "band_gap_narrowing_si",
    "fermi_dirac",
    "fermi_level_si",
    "intrinsic_fermi_level_ev",
    "Material",
    "silicon",
    "gaas",
    "germanium",
    "varshni_gap_ev",
    "thermal_voltage_v",
    "effective_density_of_states_cm3",
    "intrinsic_concentration_cm3",
    "fermi_level_nondegenerate_ev",
    "mobility_power_law",
    "conductivity_s_per_cm",
    "built_in_potential_v",
    "depletion_width_cm",
    "diode_saturation_current_density_a_per_cm2",
    "shockley_diode_current_density_a_per_cm2",
    "ShockleyQueisserResult",
    "shockley_queisser_limit",
    "TightBindingModel",
    "bipartite_supercell",
    "build_supercell",
    "add_sites",
    "neighbor_pairs",
    "build_hamiltonian_from_pairs",
    "supercell_neighbors",
    "apply_dopants",
    "build_hamiltonian",
    "solve_schrodinger",
    "half_filling_fermi_level",
    "band_gap_from_half_filling",
    "wavefunction_density",
    "wavefunction_density_grid",
    "density_of_states",
]
