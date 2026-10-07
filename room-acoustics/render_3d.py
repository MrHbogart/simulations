"""3D render of the image-source model behind room_simulation.ipynb.

Mirrored copies of the room (one per image source, up to 2 reflections) with each
image source coloured by reflection order and sized by its amplitude at the
listener, plus the six first-order reflection paths inside the real room.

Writes result/<run>/image_sources_3d.gif. Run from this folder: python render_3d.py  (~1 min)
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from soundwaves import RoomBox, SimulationConfig, SoundSource, WaveSimulation
from soundwaves.analysis import reflection_point, wall_from_indices

plt.style.use("dark_background")

config = SimulationConfig(room_size=(10.0, 5.0, 2.7), attenuation=0.01, wall_absorption=0.2, reflection_order=2)
room = RoomBox(size=config.room_size)
source = np.array([4.0, 2.0, 0.5])
receiver = np.array([6.0, 2.0, 1.8])
sim = WaveSimulation(room=room, source=SoundSource.guitar_amp_spectrum(position=source), config=config)
out = Path("result/room10.0x5.0x2.7_nx100_ny50_nz27_abs0p20_ref2")

positions, counts, indices = room.image_arrays(source, config.reflection_order)
_, _, amplitude, _, _ = sim.image_gains(receiver)
size = np.array(config.room_size)
assert np.allclose(positions[counts == 0][0], source)
assert (counts == 1).sum() == 6  # one mirror per wall

corners = np.array(list(product((0, 1), repeat=3)), dtype=float)
box_edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.abs(corners[i] - corners[j]).sum() == 1]

fig = plt.figure(figsize=(7.2, 6.4))
ax = fig.add_subplot(projection="3d", computed_zorder=False)
# Image k (per axis) lies in the room copy translated by k * size; mirroring doesn't change a box.
for k, n in zip(indices, counts):
    c = (k + corners) * size
    is_real = n == 0
    for i, j in box_edges:
        ax.plot(*c[[i, j]].T, color="white" if is_real else "#567", lw=1.6 if is_real else 0.4,
                alpha=1.0 if is_real else 0.35)

colors = ["#ffffff", "#ff5d8f", "#ffb347"]  # source, 1st, 2nd order
for order in range(config.reflection_order + 1):
    sel = counts == order
    ax.scatter(*positions[sel].T, s=260 * np.sqrt(amplitude[sel] / amplitude.max()) + 8, color=colors[order],
               edgecolors="none", alpha=0.9, depthshade=False,
               label="source" if order == 0 else f"{order} reflection{'s' * (order > 1)}")
ax.scatter(*receiver, marker="*", s=180, color="#4fd1ff", label="listener", depthshade=False)

# First-order paths: source -> wall reflection point -> listener (straight line from the image).
for p, k in zip(positions[counts == 1], indices[counts == 1]):
    hit = reflection_point(receiver, p, wall_from_indices(tuple(k)), config.room_size)
    ax.plot(*np.array([source, hit, receiver]).T, color="#7ee8a5", lw=1.4, alpha=0.95)
    ax.plot(*np.array([hit, p]).T, color="#7ee8a5", lw=0.6, ls=":", alpha=0.6)

span = np.abs(positions - size / 2).max(axis=0)
ax.set(xlim=size[0] / 2 + span[0] * np.array([-1, 1]), ylim=size[1] / 2 + span[1] * np.array([-1, 1]),
       zlim=size[2] / 2 + span[2] * np.array([-1, 1]))
ax.set_box_aspect(2 * span, zoom=1.35)
ax.set_axis_off()
ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=0.5)
ax.set_title("Image-source model: 10 × 5 × 2.7 m room, α = 0.2, up to 2 reflections\n"
             "marker size = amplitude at the listener; dotted = first-order image paths", fontsize=10)
fig.tight_layout()

anim = FuncAnimation(fig, lambda f: ax.view_init(elev=24, azim=-60 + 5 * f), frames=72)
anim.save(out / "image_sources_3d.gif", writer="pillow", fps=15, dpi=80)
print(f"wrote {out / 'image_sources_3d.gif'}: {len(positions)} image sources")
