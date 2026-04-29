"""Polymer-grafted nanoparticle builder."""

from __future__ import annotations

import numpy as np

from .nanoparticle import build_nanoparticle, fibonacci_sphere
from .sequence import generate_random_sequence, sequence_to_labels
from .topology import System
from .utils import pairwise_min_distance, rng_from_seed, validate_chain_length
from .walks import build_chain, random_unit_vector


def select_graft_sites(
    surface_positions: np.ndarray,
    num_grafts: int,
    mode: str = "random",
    min_site_distance: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Select unique grafting-site indices from surface positions."""

    if num_grafts <= 0:
        raise ValueError("num_grafts must be positive")
    if num_grafts > len(surface_positions):
        raise ValueError("num_grafts cannot exceed available graft sites")
    rng = rng_from_seed(seed)
    positions = np.asarray(surface_positions, dtype=float)

    if mode == "random":
        order = rng.permutation(len(positions))
    elif mode == "uniform":
        targets = fibonacci_sphere(num_grafts)
        directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        chosen: list[int] = []
        for target in targets:
            scores = directions @ target
            for index in np.argsort(scores)[::-1]:
                if index not in chosen:
                    chosen.append(int(index))
                    break
        order = np.asarray(chosen + [i for i in range(len(positions)) if i not in chosen], dtype=int)
    else:
        raise ValueError("graft-mode must be 'random' or 'uniform'")

    selected: list[int] = []
    for index in order:
        candidate = positions[index]
        if min_site_distance > 0.0 and selected:
            if pairwise_min_distance(positions[selected], candidate) < min_site_distance:
                continue
        selected.append(int(index))
        if len(selected) == num_grafts:
            return np.asarray(selected, dtype=int)
    raise ValueError("could not select enough graft sites with the requested min_site_distance")


def build_grafted_nanoparticle(
    np_radius: float,
    num_grafts: int,
    chain_length: int,
    np_bead_spacing: float = 1.0,
    bond_length: float = 1.0,
    walk: str = "random",
    graft_mode: str = "random",
    min_site_distance: float = 0.0,
    min_distance: float = 0.8,
    fraction_a: float = 0.5,
    seed: int | None = None,
    max_retries: int = 5000,
) -> System:
    """Build a nanoparticle with polymer chains bonded to surface sites."""

    validate_chain_length(chain_length)
    rng = rng_from_seed(seed)
    nanoparticle = build_nanoparticle(np_radius, np_bead_spacing, surface_only=False)
    surface_indices = np.asarray(nanoparticle.metadata["surface_site_indices"], dtype=int)
    selected_local = select_graft_sites(
        nanoparticle.positions[surface_indices],
        num_grafts,
        mode=graft_mode,
        min_site_distance=min_site_distance,
        seed=seed,
    )
    selected_sites = surface_indices[selected_local]
    if len(np.unique(selected_sites)) != len(selected_sites):
        raise ValueError("duplicate graft sites selected")

    positions = [nanoparticle.positions]
    atom_types = [nanoparticle.atom_types]
    labels = list(nanoparticle.labels or [])
    bonds: list[tuple[int, int]] = []
    bond_types: list[int] = []
    chain_ranges: list[tuple[int, int]] = []
    occupied = nanoparticle.positions.copy()

    for graft_number, site_index in enumerate(selected_sites):
        site = nanoparticle.positions[site_index]
        norm = np.linalg.norm(site)
        direction = site / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])
        start = site + bond_length * direction
        sequence = generate_random_sequence(
            chain_length,
            fraction_a=fraction_a,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )

        chain = _build_chain_with_retries(
            sequence=sequence,
            start=start,
            direction=direction,
            bond_length=bond_length,
            walk=walk,
            min_distance=min_distance,
            forbidden_points=occupied,
            rng=rng,
            max_retries=max_retries,
        )
        offset = sum(len(block) for block in positions)
        positions.append(chain.positions)
        atom_types.append(chain.atom_types)
        labels.extend(chain.labels or [])
        bonds.append((int(site_index), offset))
        bond_types.append(2)
        bonds.extend((offset + a, offset + b) for a, b in chain.bonds)
        bond_types.extend([1] * chain.n_bonds)
        chain_ranges.append((offset, offset + chain.n_atoms - 1))
        occupied = np.vstack((occupied, chain.positions))

    all_positions = np.vstack(positions)
    all_atom_types = np.concatenate(atom_types)
    return System(
        positions=all_positions,
        atom_types=all_atom_types,
        bonds=bonds,
        bond_types=bond_types,
        labels=labels,
        metadata={
            "builder": "grafted_nanoparticle",
            "np_radius": np_radius,
            "np_bead_spacing": np_bead_spacing,
            "num_grafts": num_grafts,
            "chain_length": chain_length,
            "bond_length": bond_length,
            "walk": walk,
            "graft_mode": graft_mode,
            "min_site_distance": min_site_distance,
            "min_distance": min_distance,
            "fraction_a": fraction_a,
            "seed": seed,
            "graft_site_indices": selected_sites.tolist(),
            "chain_ranges": chain_ranges,
        },
    )


def _build_chain_with_retries(
    sequence: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    bond_length: float,
    walk: str,
    min_distance: float,
    forbidden_points: np.ndarray,
    rng: np.random.Generator,
    max_retries: int,
) -> System:
    if walk == "saw":
        return _biased_outward_saw_chain(
            sequence=sequence,
            start=start,
            direction=direction,
            bond_length=bond_length,
            min_distance=min_distance,
            forbidden_points=forbidden_points,
            rng=rng,
            max_retries=max_retries,
        )

    for _ in range(max(1, max_retries)):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        try:
            return build_chain(
                sequence=sequence,
                bond_length=bond_length,
                walk=walk,
                min_distance=min_distance,
                seed=seed,
                max_retries=max_retries,
                start=start,
                first_direction=direction,
                forbidden_points=forbidden_points,
            )
        except RuntimeError:
            continue
    raise RuntimeError("failed to graft chain without severe overlap; increase retries or relax distances")


def _biased_outward_saw_chain(
    sequence: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    bond_length: float,
    min_distance: float,
    forbidden_points: np.ndarray,
    rng: np.random.Generator,
    max_retries: int,
) -> System:
    """Grow a grafted SAW with a mild outward bias from the surface normal."""

    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    start = np.asarray(start, dtype=float)
    forbidden = np.asarray(forbidden_points, dtype=float)

    for _ in range(max(1, max_retries)):
        positions = np.zeros((len(sequence), 3), dtype=float)
        positions[0] = start
        if pairwise_min_distance(forbidden, positions[0]) < min_distance:
            continue
        accepted_chain = True
        for i in range(1, len(sequence)):
            accepted_bead = False
            for _ in range(max_retries):
                trial_direction = direction + 0.25 * random_unit_vector(rng)
                norm = np.linalg.norm(trial_direction)
                if norm == 0.0:
                    continue
                trial_direction = trial_direction / norm
                candidate = positions[i - 1] + bond_length * trial_direction
                if np.dot(trial_direction, direction) < 0.85:
                    continue
                own_ok = pairwise_min_distance(positions[: max(0, i - 1)], candidate) >= min_distance
                forbidden_ok = pairwise_min_distance(forbidden, candidate) >= min_distance
                if own_ok and forbidden_ok:
                    positions[i] = candidate
                    accepted_bead = True
                    break
            if not accepted_bead:
                accepted_chain = False
                break
        if accepted_chain:
            atom_types = np.where(sequence == 1, 1, 2)
            bonds = [(i, i + 1) for i in range(len(sequence) - 1)]
            return System(
                positions=positions,
                atom_types=atom_types,
                bonds=bonds,
                bond_types=[1] * len(bonds),
                labels=sequence_to_labels(sequence),
                metadata={"builder": "grafted_chain", "walk": "saw"},
            )
    straight = _straight_outward_chain(sequence, start, direction, bond_length, min_distance, forbidden)
    if straight is not None:
        return straight
    raise RuntimeError("failed to graft chain without severe overlap; increase retries or relax distances")


def _straight_outward_chain(
    sequence: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    bond_length: float,
    min_distance: float,
    forbidden: np.ndarray,
) -> System | None:
    positions = np.asarray([start + i * bond_length * direction for i in range(len(sequence))])
    for i, candidate in enumerate(positions):
        if pairwise_min_distance(forbidden, candidate) < min_distance:
            return None
        if pairwise_min_distance(positions[: max(0, i - 1)], candidate) < min_distance:
            return None
    atom_types = np.where(sequence == 1, 1, 2)
    bonds = [(i, i + 1) for i in range(len(sequence) - 1)]
    return System(
        positions=positions,
        atom_types=atom_types,
        bonds=bonds,
        bond_types=[1] * len(bonds),
        labels=sequence_to_labels(sequence),
        metadata={"builder": "grafted_chain", "walk": "saw", "fallback": "straight_outward"},
    )
