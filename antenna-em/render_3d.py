"""3D far-field radiation pattern of the single_antenna.ipynb V-dipole.

Writes results/wire_v_dipole_f300MHz_n181_planexz_angle50deg_L1m/radiation_pattern_3d.gif.
Run from this folder: python render_3d.py  (~1 min)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from emsim import WireAntenna, field_magnitude, make_v_dipole, wire_antenna_fields

plt.style.use("dark_background")

FREQUENCY = 300e6
antenna = WireAntenna(path=make_v_dipole(1.0, 50.0, n_points=181, plane="xz"), radius=0.01,
                      current_amp=1.0, frequency=FREQUENCY, current_profile="sinusoidal")
out = Path("results/wire_v_dipole_f300MHz_n181_planexz_angle50deg_L1m")

# |E|^2 on a sphere 100 wavelengths out (far zone for a 1 m antenna at 1 m wavelength).
theta, phi = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121), indexing="ij")
r_far = 100 * 299_792_458.0 / FREQUENCY
n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
power = field_magnitude(*wire_antenna_fields(antenna, *(r_far * n))[:3]) ** 2
power /= power.max()

# The V lies in the x-z plane, symmetric under x -> -x, i.e. phi -> pi - phi (index 60 - j).
j = np.arange(120)
assert np.allclose(power[:, j], power[:, (60 - j) % 120], atol=1e-6)

fig = plt.figure(figsize=(6.4, 6.4))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(*(power * n), facecolors=cm.inferno(0.15 + 0.85 * power), rstride=1, cstride=1,
                linewidth=0, antialiased=False, alpha=0.8, shade=False)
wire = antenna.path / np.abs(antenna.path).max() * 0.9  # antenna, scaled into the plot
ax.plot(*wire.T, color="#4fd1ff", lw=3)
ax.scatter(0, 0, 0, color="white", s=18)  # feed point
lim = 1.0
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
ax.set_box_aspect((1, 1, 1), zoom=1.2)
ax.set_axis_off()
ax.set_title("V-dipole (1 m = 1 λ, 50°, 300 MHz): 3D radiation pattern\n"
             "surface radius and colour = normalised power $|E|^2$", fontsize=11)
fig.tight_layout()

anim = FuncAnimation(fig, lambda f: ax.view_init(elev=18, azim=5 * f), frames=72)
anim.save(out / "radiation_pattern_3d.gif", writer="pillow", fps=15, dpi=80)
print(f"wrote {out / 'radiation_pattern_3d.gif'}")
