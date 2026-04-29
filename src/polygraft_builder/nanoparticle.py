"""Spherical coarse-grained nanoparticle builder."""

from __future__ import annotations

import numpy as np

from .topology import System


def build_nanoparticle(
    np_radius: float,
    np_bead_spacing: float = 1.0,
    surface_only: bool = False,
    surface_tolerance: float | None = None,
) -> System:
    """Build a spherical nanoparticle on a cubic bead lattice."""

    if np_radius <= 0.0:
        raise ValueError("np_radius must be positive")
    if np_bead_spacing <= 0.0:
        raise ValueError("np_bead_spacing must be positive")
    if surface_tolerance is None:
        surface_tolerance = 0.75 * np_bead_spacing

    coords = np.arange(-np_radius, np_radius + 0.5 * np_bead_spacing, np_bead_spacing)
    grid = np.array(np.meshgrid(coords, coords, coords, indexing="ij")).reshape(3, -1).T
    radii = np.linalg.norm(grid, axis=1)
    inside = radii <= np_radius + 1.0e-9
    if surface_only:
        mask = inside & (radii >= np_radius - surface_tolerance)
    else:
        mask = inside
    positions = grid[mask]
    if len(positions) == 0:
        raise ValueError("nanoparticle contains no beads; adjust radius or spacing")
    surface_sites = detect_surface_sites(positions, np_radius, surface_tolerance)
    metadata = {
        "builder": "nanoparticle",
        "np_radius": np_radius,
        "np_bead_spacing": np_bead_spacing,
        "surface_only": surface_only,
        "surface_tolerance": surface_tolerance,
        "surface_site_indices": surface_sites.tolist(),
        "num_surface_sites": int(len(surface_sites)),
    }
    return System(
        positions=positions,
        atom_types=np.full(len(positions), 3, dtype=int),
        labels=["NP"] * len(positions),
        metadata=metadata,
    )


def detect_surface_sites(
    positions: np.ndarray,
    np_radius: float | None = None,
    surface_tolerance: float = 1.0,
) -> np.ndarray:
    """Detect bead indices close to the outer spherical surface."""

    positions = np.asarray(positions, dtype=float)
    radii = np.linalg.norm(positions, axis=1)
    radius = float(np.max(radii) if np_radius is None else np_radius)
    return np.flatnonzero(radii >= radius - surface_tolerance)


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """Return approximately uniform unit vectors on a sphere."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(n_points)
    z = 1.0 - (2.0 * indices + 1.0) / n_points
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden_angle * indices
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
