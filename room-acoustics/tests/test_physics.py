"""Physics sanity checks for the image-source room model."""

import numpy as np

from soundwaves import RoomBox, SimulationConfig, SoundSource, WaveSimulation, compute_band_metrics
from soundwaves.analysis import _estimate_decay_slope

ROOM = (10.0, 5.0, 2.7)


def test_first_order_images_mirror_all_six_walls():
    src = np.array([4.0, 2.0, 0.5])
    images = RoomBox(ROOM).image_sources(src, order=1)
    first = {img.indices: img.position for img in images if img.reflection_count == 1}
    assert len(images) == 7 and len(first) == 6
    np.testing.assert_allclose(first[(-1, 0, 0)], [-4.0, 2.0, 0.5])  # wall x = 0
    np.testing.assert_allclose(first[(1, 0, 0)], [16.0, 2.0, 0.5])  # wall x = Lx
    np.testing.assert_allclose(first[(0, 0, -1)], [4.0, 2.0, -0.5])  # floor


def test_image_count_matches_octahedral_number():
    # Number of (kx, ky, kz) with |kx|+|ky|+|kz| <= N is (2N+1)(2N^2+2N+3)/3.
    for n in (0, 1, 2, 5):
        positions, _, _ = RoomBox(ROOM).image_arrays(np.array([1.0, 1.0, 1.0]), n)
        assert len(positions) == (2 * n + 1) * (2 * n * n + 2 * n + 3) // 3


def test_high_frequency_decay_matches_specular_theory():
    """At high frequency the image sum is incoherent, so the energy arriving at time t
    is the direction average of (1 - alpha)^(c t sum_i |u_i| / L_i). A specular
    shoebox is not diffuse, so this (not Eyring) is the right reference."""
    alpha, c, fs, duration = 0.2, 343.0, 16000, 1.0
    dt = 1.0 / fs
    t = np.arange(int(duration * fs)) * dt
    i = np.arange(4000) + 0.5
    polar, azimuth = np.arccos(1 - 2 * i / i.size), np.pi * (1 + 5**0.5) * i
    u = np.column_stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)])
    rate = np.mean((1 - alpha) ** (c * t[:, None] * (np.abs(u) @ (1 / np.array(ROOM)))[None, :]), axis=1)
    edc = 10 * np.log10(np.cumsum(rate[::-1])[::-1] / rate.sum())
    expected_t30 = -60.0 / _estimate_decay_slope(edc, dt, (-5.0, -35.0))

    config = SimulationConfig(room_size=ROOM, dt=dt, duration=duration, reflection_order=150, wall_absorption=alpha)
    source = SoundSource.guitar_amp_spectrum(position=np.array([4.0, 2.0, 0.5]))
    rir = WaveSimulation(RoomBox(ROOM), source, config).impulse_response(np.array([6.0, 2.0, 1.8]))
    t30 = compute_band_metrics(rir.response, dt, centers_hz=[4000.0])[4000.0].t30
    assert abs(t30 - expected_t30) / expected_t30 < 0.1, (t30, expected_t30)
