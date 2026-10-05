"""Field evaluation helpers."""

from __future__ import annotations

import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def _tqdm(iterable, **kwargs):
        return iterable

from .antenna import DipoleAntenna, WireAntenna, c0, eps0, eta0


def hertzian_dipole_fields_vec(p_vec, k: float, X, Y, Z, eps: float = 1e-9):
    """Phasor fields from a Hertzian dipole with vector moment at the origin."""
    r = np.sqrt(X**2 + Y**2 + Z**2) + eps
    nx = X / r
    ny = Y / r
    nz = Z / r

    px, py, pz = p_vec
    n_dot_p = nx * px + ny * py + nz * pz
    exp_term = np.exp(-1j * k * r)

    tx = 3.0 * nx * n_dot_p - px
    ty = 3.0 * ny * n_dot_p - py
    tz = 3.0 * nz * n_dot_p - pz

    n_cross_p_x = ny * pz - nz * py
    n_cross_p_y = nz * px - nx * pz
    n_cross_p_z = nx * py - ny * px
    cx = n_cross_p_y * nz - n_cross_p_z * ny
    cy = n_cross_p_z * nx - n_cross_p_x * nz
    cz = n_cross_p_x * ny - n_cross_p_y * nx

    coeff_near = (1.0 / r**3 + 1j * k / r**2)
    coeff_rad = (k**2 / r)

    prefactor = 1.0 / (4.0 * np.pi * eps0())

    Ex = prefactor * (coeff_rad * cx + coeff_near * tx) * exp_term
    Ey = prefactor * (coeff_rad * cy + coeff_near * ty) * exp_term
    Ez = prefactor * (coeff_rad * cz + coeff_near * tz) * exp_term

    # Exact dipole H (no 1/r^3 term): H = c/(4 pi) (n x p) (k^2/r - j k/r^2) e^{-jkr}.
    # The far-field shortcut H = n x E / eta0 adds a spurious near-field 1/r^3 term.
    h_coeff = (c0() / (4.0 * np.pi)) * (k**2 / r - 1j * k / r**2) * exp_term
    Hx = h_coeff * n_cross_p_x
    Hy = h_coeff * n_cross_p_y
    Hz = h_coeff * n_cross_p_z

    return Ex, Ey, Ez, Hx, Hy, Hz


def hertzian_dipole_fields(p: complex, k: float, X, Y, Z, eps: float = 1e-9):
    """Phasor fields from a z-directed Hertzian dipole at the origin."""
    return hertzian_dipole_fields_vec(np.array([0.0, 0.0, p], dtype=complex), k, X, Y, Z, eps=eps)


def dipole_fields(antenna: DipoleAntenna, X, Y, Z, eps: float = 1e-9):
    """Compute phasor E and H fields from a z-directed oscillating dipole.

    Uses the exact (near + intermediate + far) Hertzian dipole fields with
    exp(-j k r) spatial phase, i.e. e^{j omega t} time dependence.
    """
    p = antenna.dipole_moment
    k = antenna.k
    return hertzian_dipole_fields(p, k, X, Y, Z, eps=eps)


def line_dipole_fields(
    antenna: DipoleAntenna,
    X,
    Y,
    Z,
    n_segments: int = 121,
    eps: float = 1e-9,
):
    """Approximate a finite-length dipole by summing Hertzian segments."""
    L = antenna.length
    k = antenna.k
    omega = antenna.omega

    z = np.linspace(-L / 2.0, L / 2.0, n_segments)
    dz = z[1] - z[0] if n_segments > 1 else L
    I = antenna.current_distribution(z)

    Ex = np.zeros_like(X, dtype=complex)
    Ey = np.zeros_like(X, dtype=complex)
    Ez = np.zeros_like(X, dtype=complex)
    Hx = np.zeros_like(X, dtype=complex)
    Hy = np.zeros_like(X, dtype=complex)
    Hz = np.zeros_like(X, dtype=complex)

    for zi, Ii in _tqdm(
        zip(z, I),
        total=len(z),
        desc="Line dipole segments",
        leave=False,
    ):
        p_seg = (Ii * dz) / (1j * omega)
        Exi, Eyi, Ezi, Hxi, Hyi, Hzi = hertzian_dipole_fields(
            p_seg, k, X, Y, Z - zi, eps=eps
        )
        Ex += Exi
        Ey += Eyi
        Ez += Ezi
        Hx += Hxi
        Hy += Hyi
        Hz += Hzi

    return Ex, Ey, Ez, Hx, Hy, Hz


def wire_antenna_fields(antenna: WireAntenna, X, Y, Z, eps: float = 1e-9):
    """Compute phasor fields for a thin-wire antenna described by a polyline."""
    centers, directions, lengths, s_centers, total_length = antenna.segment_data()
    I = antenna.current_distribution(s_centers, total_length)
    k = antenna.k
    omega = antenna.omega

    Ex = np.zeros_like(X, dtype=complex)
    Ey = np.zeros_like(X, dtype=complex)
    Ez = np.zeros_like(X, dtype=complex)
    Hx = np.zeros_like(X, dtype=complex)
    Hy = np.zeros_like(X, dtype=complex)
    Hz = np.zeros_like(X, dtype=complex)

    for center, direction, length, Ii in _tqdm(
        zip(centers, directions, lengths, I),
        total=len(centers),
        desc="Wire segments",
        leave=False,
    ):
        p_vec = (Ii * length) / (1j * omega) * direction
        Exi, Eyi, Ezi, Hxi, Hyi, Hzi = hertzian_dipole_fields_vec(
            p_vec, k, X - center[0], Y - center[1], Z - center[2], eps=eps
        )
        Ex += Exi
        Ey += Eyi
        Ez += Ezi
        Hx += Hxi
        Hy += Hyi
        Hz += Hzi

    return Ex, Ey, Ez, Hx, Hy, Hz


def internal_antenna_fields(antenna: DipoleAntenna, X, Y, Z, eps: float = 1e-9):
    """Simple internal fields for a cylindrical dipole antenna.

    This is a toy model: standing-wave axial E field and azimuthal H field.
    """
    a = antenna.radius
    L = antenna.length
    I0 = antenna.current_amp
    k = antenna.k

    r_perp = np.sqrt(X**2 + Y**2) + eps
    inside = (r_perp <= a) & (np.abs(Z) <= L / 2.0)

    # Standing-wave axial E field
    Ez = np.zeros_like(X, dtype=complex)
    z_norm = np.clip(Z / (L / 2.0), -1.0, 1.0)
    Ez_inside = (I0 * eta0() / (2.0 * np.pi * a)) * np.cos(0.5 * np.pi * z_norm)
    Ez = np.where(inside, Ez_inside, Ez)

    # Azimuthal H from a uniform current distribution
    Hphi = np.zeros_like(X, dtype=complex)
    Hphi_inside = (I0 * r_perp) / (2.0 * np.pi * a**2)
    Hphi = np.where(inside, Hphi_inside, Hphi)

    # Convert Hphi to Cartesian components
    cos_phi = X / r_perp
    sin_phi = Y / r_perp
    Hx = -sin_phi * Hphi
    Hy = cos_phi * Hphi
    Hz = np.zeros_like(Hx)

    Ex = np.zeros_like(Ez)
    Ey = np.zeros_like(Ez)

    # Apply a propagation phase along z for a phasor feel
    phase = np.exp(-1j * k * Z)
    Ez = Ez * phase
    Hx = Hx * phase
    Hy = Hy * phase

    return Ex, Ey, Ez, Hx, Hy, Hz, inside


def combine_fields(external, internal, inside_mask):
    """Combine external and internal fields with a hard mask."""
    Ex, Ey, Ez, Hx, Hy, Hz = external
    iEx, iEy, iEz, iHx, iHy, iHz = internal

    Ex = np.where(inside_mask, iEx, Ex)
    Ey = np.where(inside_mask, iEy, Ey)
    Ez = np.where(inside_mask, iEz, Ez)
    Hx = np.where(inside_mask, iHx, Hx)
    Hy = np.where(inside_mask, iHy, Hy)
    Hz = np.where(inside_mask, iHz, Hz)

    return Ex, Ey, Ez, Hx, Hy, Hz


def empty_internal_fields(X, Y, Z):
    """Return zero internal fields and an empty mask."""
    zeros = np.zeros_like(X, dtype=complex)
    mask = np.zeros_like(X, dtype=bool)
    return zeros, zeros, zeros, zeros, zeros, zeros, mask
