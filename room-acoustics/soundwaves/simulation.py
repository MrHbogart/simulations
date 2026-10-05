"""Simulation core for room sound propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SimulationConfig
from .geometry import RoomBox
from .source import SoundSource
from .analysis import ImpulseResponseResult, ReflectionArrival


@dataclass
class WaveSimulation:
    room: RoomBox
    source: SoundSource
    config: SimulationConfig

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lx, ly, lz = self.config.room_size
        x = np.linspace(0.0, lx, self.config.nx)
        y = np.linspace(0.0, ly, self.config.ny)
        z = np.linspace(0.0, lz, self.config.nz)
        return np.meshgrid(x, y, z, indexing="ij")

    def field_at(self, t: float) -> np.ndarray:
        """Return pressure field at time t on the configured grid."""
        xx, yy, zz = self.grid()
        field = np.zeros_like(xx)
        positions, counts, _ = self.room.image_arrays(
            self.source.position, self.config.reflection_order
        )
        gains = _pressure_reflection(self.config.wall_absorption) ** counts
        for position, gain in zip(positions, gains):
            r = np.sqrt((xx - position[0]) ** 2 + (yy - position[1]) ** 2 + (zz - position[2]) ** 2)
            field += gain * self.source.pressure(r, t, self.config.c, self.config.attenuation)

        return field

    def image_gains(self, receiver_position: np.ndarray, order: int | None = None):
        """Distances, delays and amplitudes of every image source at a receiver."""
        order = self.config.reflection_order if order is None else order
        positions, counts, indices = self.room.image_arrays(self.source.position, order)
        r = np.linalg.norm(positions - np.asarray(receiver_position, dtype=float), axis=1)
        valid = r > 0.0
        r, counts, indices = r[valid], counts[valid], indices[valid]
        amplitude = _pressure_reflection(self.config.wall_absorption) ** counts * self.source.amplitude
        amplitude = amplitude * np.exp(-self.config.attenuation * r) / np.maximum(r, 1e-3)
        return r, r / self.config.c, amplitude, counts, indices

    def impulse_response(
        self,
        receiver_position: np.ndarray,
        normalize: bool = True,
        progress: bool = False,  # kept for API compatibility; vectorized now
    ) -> ImpulseResponseResult:
        """Compute a broadband impulse response at a receiver location."""
        steps = self.config.steps
        times = np.arange(steps) * self.config.dt
        _, delay, amplitude, _, _ = self.image_gains(receiver_position)
        response = _bin_arrivals(delay, amplitude, self.config.dt, steps)
        if normalize and np.max(np.abs(response)) > 0:
            response = response / np.max(np.abs(response))
        return ImpulseResponseResult(time=times, response=response)

    def reflection_arrivals(self, receiver_position: np.ndarray) -> list[ReflectionArrival]:
        """Return arrival times and amplitudes for each image source."""
        _, delay, amplitude, counts, _ = self.image_gains(receiver_position)
        order = np.argsort(delay)
        return [
            ReflectionArrival(time=float(delay[i]), amplitude=float(amplitude[i]), reflection_count=int(counts[i]))
            for i in order
        ]


def _bin_arrivals(delay: np.ndarray, amplitude: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """Accumulate image arrivals into a sampled impulse response."""
    idx = np.rint(delay / dt).astype(int)
    keep = idx < steps
    response = np.zeros(steps, dtype=float)
    np.add.at(response, idx[keep], amplitude[keep])
    return response


def _pressure_reflection(absorption: float) -> float:
    """Pressure reflection factor for an energy absorption coefficient alpha."""
    return float(np.sqrt(1.0 - absorption))
