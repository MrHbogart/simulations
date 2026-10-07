"""3D renders of the silicon EPM band structure.

Writes into the latest results_silicon_epm_diode/<run>/ folder:
  si_band_surfaces_3d.png      valence and conduction bands over the kz = 0 plane
  si_conduction_valleys_3d.gif the six Delta conduction valleys inside the first Brillouin zone

Run from this folder: python render_3d.py  (~1 min)
"""

from itertools import permutations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from quantsim import EmpiricalPseudopotential, Lattice, epm_energies

plt.style.use("dark_background")

model = EmpiricalPseudopotential.silicon()
lattice = Lattice.fcc(model.lattice_constant_m)
unit = 2 * np.pi / model.lattice_constant_m  # k in units of 2*pi/a
g_cut = unit * np.sqrt(11.5)
out = sorted(Path("results_silicon_epm_diode").glob("*/"))[-1]


def in_bz(k):
    """First Brillouin zone of fcc (truncated octahedron), k in units of 2*pi/a."""
    a = np.abs(k)
    return (a.max(axis=-1) <= 1.0 + 1e-9) & (a.sum(axis=-1) <= 1.5 + 1e-9)


def energies(k):
    return epm_energies(lattice, model, k * unit, g_cut, n_bands=5)


# Band edges along Gamma-X for the energy reference.
line = np.zeros((101, 3))
line[:, 0] = np.linspace(0, 1, 101)
e_line = energies(line)
vbm, cbm = e_line[:, 3].max(), e_line[:, 4].min()

# --- Band surfaces over kz = 0 -------------------------------------------------
n = 81
kx, ky = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
plane = np.column_stack([kx.ravel(), ky.ravel(), np.zeros(n * n)])
mask = in_bz(plane)
e_plane = np.full((n * n, 5), np.nan)
e_plane[mask] = energies(plane[mask])
e_plane = (e_plane - vbm).reshape(n, n, 5)

fig = plt.figure(figsize=(9, 7.5))
ax = fig.add_subplot(projection="3d", computed_zorder=False)
for band, cmap in ((2, "Blues_r"), (3, "Blues_r"), (4, "inferno")):
    ax.plot_surface(kx, ky, e_plane[..., band], cmap=cmap, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.95)
ax.set_xlabel(r"$k_x$ ($2\pi/a$)")
ax.set_ylabel(r"$k_y$ ($2\pi/a$)")
ax.set_zlabel(r"$E - E_\mathrm{VBM}$ (eV)")
ax.set_title("Si (EPM): top valence bands and lowest conduction band, $k_z = 0$\n"
             f"conduction minima at ±0.85 X, indirect gap {cbm - vbm:.2f} eV (bare Cohen–Bergstresser)", pad=4)
ax.view_init(elev=24, azim=-58)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0, 0, 0, 0))
fig.tight_layout()
fig.savefig(out / "si_band_surfaces_3d.png", dpi=130)
plt.close(fig)

# --- Conduction valleys inside the Brillouin zone ------------------------------
# Solve one valley (+X) on a fine box, then copy it to the other five by cubic symmetry.
box = np.stack(np.meshgrid(np.linspace(0.4, 1.0, 31), *[np.linspace(-0.3, 0.3, 31)] * 2,
                           indexing="ij"), -1).reshape(-1, 3)
e_c = energies(box)[:, 4] - cbm
window = 0.15  # eV above the CBM
valley, e_valley = box[e_c < window], e_c[e_c < window]
pts = np.concatenate([sign * np.roll(valley, axis, axis=1) for axis in range(3) for sign in (1, -1)])
e_pts = np.tile(e_valley, 6)

# BZ edges: the 24 vertices are permutations of (0, +-1/2, +-1); edges have length sqrt(2)/2.
verts = np.unique([p for s in product((1, -1), repeat=2)
                   for p in permutations((0, 0.5 * s[0], 1.0 * s[1]))], axis=0)
dist = np.linalg.norm(verts[:, None] - verts[None], axis=-1)
edges = [(i, j) for i, j in zip(*np.nonzero(np.isclose(dist, np.sqrt(0.5)))) if i < j]

fig = plt.figure(figsize=(6.4, 6.4))
ax = fig.add_subplot(projection="3d")
for i, j in edges:
    ax.plot(*verts[[i, j]].T, color="#cde", lw=1.0, alpha=0.8)
order = np.argsort(-e_pts)  # brightest (lowest energy) cores drawn last
ax.scatter(*pts[order].T, c=e_pts[order], cmap="plasma", vmin=0, vmax=window,
           s=4, alpha=0.25, linewidths=0, depthshade=False)
labels = {r"$\Gamma$": (0, 0, 0), "X": (1, 0, 0), "L": (0.5, 0.5, 0.5), "W": (1, 0.5, 0), "K": (0.75, 0.75, 0)}
for name, p in labels.items():
    ax.scatter(*p, color="white", s=12)
    ax.text(*(np.array(p) * 1.08), name, color="white", fontsize=11)
ax.set_box_aspect((1, 1, 1), zoom=1.25)
ax.set_axis_off()
ax.set_title(f"Si conduction valleys: $E_c < E_\\mathrm{{CBM}} + {window * 1000:.0f}$ meV\n"
             "six Δ ellipsoids along ⟨100⟩ inside the first Brillouin zone", fontsize=11)
fig.tight_layout()

anim = FuncAnimation(fig, lambda f: ax.view_init(elev=22, azim=30 + 5 * f), frames=72)
anim.save(out / "si_conduction_valleys_3d.gif", writer="pillow", fps=15, dpi=80)
plt.close(fig)

# Valleys sit on the six +-k_i axes, ~85% of the way to X (same check as tests/test_physics.py).
core = pts[e_pts < 0.02]
assert np.all(np.sort(np.abs(core), axis=1)[:, -1] > 0.7), "valley cores should lie along <100>"
print(f"wrote 3D renders to {out}; gap {cbm - vbm:.2f} eV, {len(pts)} valley points")
