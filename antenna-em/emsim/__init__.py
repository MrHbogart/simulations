"""Lightweight electromagnetic simulation helpers."""

from .grid import create_grid, cart_to_spherical
from .antenna import (
    DipoleAntenna,
    WireAntenna,
    make_straight_wire,
    make_circular_loop,
    make_rectangular_loop,
    make_v_dipole,
    resample_polyline,
)
from .fields import (
    dipole_fields,
    line_dipole_fields,
    wire_antenna_fields,
    internal_antenna_fields,
    empty_internal_fields,
    combine_fields,
)
from .viz import field_magnitude, poynting_vector, take_slice

__all__ = [
    "create_grid",
    "cart_to_spherical",
    "DipoleAntenna",
    "WireAntenna",
    "make_straight_wire",
    "make_circular_loop",
    "make_rectangular_loop",
    "make_v_dipole",
    "resample_polyline",
    "dipole_fields",
    "line_dipole_fields",
    "wire_antenna_fields",
    "internal_antenna_fields",
    "empty_internal_fields",
    "combine_fields",
    "field_magnitude",
    "poynting_vector",
    "take_slice",
]
