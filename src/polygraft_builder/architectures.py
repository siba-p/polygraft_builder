"""Nonlinear polymer architecture builders."""

from __future__ import annotations

import numpy as np

from .nanoparticle import fibonacci_sphere
from .sequence import generate_random_sequence
from .topology import System
from .utils import rng_from_seed, validate_chain_length
from .walks import build_chain


def build_polymer(
    architecture: str = "linear",
    chain_length: int | None = None,
    sequence: str | list[int] | np.ndarray | None = None,
    fraction_a: float = 0.5,
    bond_length: float = 1.0,
    walk: str = "random",
    min_distance: float = 0.8,
    seed: int | None = None,
    max_retries: int = 5000,
    num_arms: int = 4,
    arm_length: int | None = None,
    backbone_length: int | None = None,
    side_chain_length: int = 5,
    graft_every: int = 2,
) -> System:
    """Build a linear, star, or bottlebrush polymer architecture."""

    if architecture == "linear":
        return build_chain(
            chain_length=chain_length,
            sequence=sequence,
            fraction_a=fraction_a,
            bond_length=bond_length,
            walk=walk,
            min_distance=min_distance,
            seed=seed,
            max_retries=max_retries,
        )
    if architecture == "star":
        resolved_arm_length = arm_length if arm_length is not None else chain_length
        if resolved_arm_length is None:
            raise ValueError("arm_length or chain_length is required for star polymers")
        return build_star_polymer(
            num_arms=num_arms,
            arm_length=resolved_arm_length,
            fraction_a=fraction_a,
            bond_length=bond_length,
            walk=walk,
            min_distance=min_distance,
            seed=seed,
            max_retries=max_retries,
        )
    if architecture == "brush":
        resolved_backbone_length = backbone_length if backbone_length is not None else chain_length
        if resolved_backbone_length is None:
            raise ValueError("backbone_length or chain_length is required for brush polymers")
        return build_bottlebrush_polymer(
            backbone_length=resolved_backbone_length,
            side_chain_length=side_chain_length,
            graft_every=graft_every,
            fraction_a=fraction_a,
            bond_length=bond_length,
            walk=walk,
            min_distance=min_distance,
            seed=seed,
            max_retries=max_retries,
        )
    raise ValueError("architecture must be 'linear', 'star', or 'brush'")


def build_star_polymer(
    num_arms: int,
    arm_length: int,
    fraction_a: float = 0.5,
    bond_length: float = 1.0,
    walk: str = "random",
    min_distance: float = 0.8,
    seed: int | None = None,
    max_retries: int = 5000,
) -> System:
    """Build a star polymer with a central core and outward arms."""

    if num_arms <= 1:
        raise ValueError("num_arms must be greater than 1")
    validate_chain_length(arm_length)
    rng = rng_from_seed(seed)
    directions = fibonacci_sphere(num_arms)
    positions = [np.zeros((1, 3), dtype=float)]
    atom_types = [np.asarray([4], dtype=int)]
    labels = ["CORE"]
    bonds: list[tuple[int, int]] = []
    bond_types: list[int] = []
    arm_ranges: list[tuple[int, int]] = []
    occupied = positions[0].copy()

    for direction in directions:
        sequence = generate_random_sequence(
            arm_length,
            fraction_a=fraction_a,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        try:
            chain = build_chain(
                sequence=sequence,
                bond_length=bond_length,
                walk=walk,
                min_distance=min_distance,
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                max_retries=max_retries,
                start=bond_length * direction,
                first_direction=direction,
                forbidden_points=occupied if walk == "saw" else None,
            )
        except RuntimeError:
            if walk != "saw":
                raise
            chain = _straight_arm_chain(sequence, direction, bond_length)
        offset = sum(len(block) for block in positions)
        positions.append(chain.positions)
        atom_types.append(chain.atom_types)
        labels.extend(chain.labels or [])
        bonds.append((0, offset))
        bond_types.append(3)
        bonds.extend((offset + a, offset + b) for a, b in chain.bonds)
        bond_types.extend([1] * chain.n_bonds)
        arm_ranges.append((offset, offset + chain.n_atoms - 1))
        occupied = np.vstack((occupied, chain.positions))

    return System(
        positions=np.vstack(positions),
        atom_types=np.concatenate(atom_types),
        bonds=bonds,
        bond_types=bond_types,
        labels=labels,
        metadata={
            "builder": "polymer",
            "architecture": "star",
            "num_arms": num_arms,
            "arm_length": arm_length,
            "bond_length": bond_length,
            "walk": walk,
            "fraction_a": fraction_a,
            "seed": seed,
            "arm_ranges": arm_ranges,
        },
    )


def _straight_arm_chain(sequence: np.ndarray, direction: np.ndarray, bond_length: float) -> System:
    positions = np.asarray([(i + 1) * bond_length * direction for i in range(len(sequence))])
    atom_types = np.where(sequence == 1, 1, 2)
    bonds = [(i, i + 1) for i in range(len(sequence) - 1)]
    labels = ["A" if value == 1 else "B" for value in sequence]
    return System(
        positions=positions,
        atom_types=atom_types,
        bonds=bonds,
        bond_types=[1] * len(bonds),
        labels=labels,
        metadata={"builder": "star_arm", "fallback": "straight_outward"},
    )


def build_bottlebrush_polymer(
    backbone_length: int,
    side_chain_length: int,
    graft_every: int = 2,
    fraction_a: float = 0.5,
    bond_length: float = 1.0,
    walk: str = "random",
    min_distance: float = 0.8,
    seed: int | None = None,
    max_retries: int = 5000,
) -> System:
    """Build a bottlebrush polymer with side chains grafted to a backbone."""

    validate_chain_length(backbone_length)
    validate_chain_length(side_chain_length)
    if graft_every <= 0:
        raise ValueError("graft_every must be positive")
    rng = rng_from_seed(seed)
    backbone_sequence = generate_random_sequence(backbone_length, fraction_a=fraction_a, seed=seed)
    backbone_positions = np.column_stack(
        (
            np.arange(backbone_length, dtype=float) * bond_length,
            np.zeros(backbone_length),
            np.zeros(backbone_length),
        )
    )
    positions = [backbone_positions]
    atom_types = [np.where(backbone_sequence == 1, 1, 2)]
    labels = ["A" if value == 1 else "B" for value in backbone_sequence]
    bonds = [(i, i + 1) for i in range(backbone_length - 1)]
    bond_types = [1] * len(bonds)
    occupied = backbone_positions.copy()
    side_chain_ranges: list[tuple[int, int]] = []
    graft_sites = list(range(0, backbone_length, graft_every))

    for graft_count, site_index in enumerate(graft_sites):
        angle = 2.0 * np.pi * graft_count / max(1, len(graft_sites))
        direction = np.array([0.0, np.cos(angle), np.sin(angle)])
        sequence = generate_random_sequence(
            side_chain_length,
            fraction_a=fraction_a,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        chain = build_chain(
            sequence=sequence,
            bond_length=bond_length,
            walk=walk,
            min_distance=min_distance,
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            max_retries=max_retries,
            start=backbone_positions[site_index] + bond_length * direction,
            first_direction=direction,
            forbidden_points=occupied if walk == "saw" else None,
        )
        offset = sum(len(block) for block in positions)
        positions.append(chain.positions)
        atom_types.append(chain.atom_types)
        labels.extend(chain.labels or [])
        bonds.append((site_index, offset))
        bond_types.append(4)
        bonds.extend((offset + a, offset + b) for a, b in chain.bonds)
        bond_types.extend([1] * chain.n_bonds)
        side_chain_ranges.append((offset, offset + chain.n_atoms - 1))
        occupied = np.vstack((occupied, chain.positions))

    return System(
        positions=np.vstack(positions),
        atom_types=np.concatenate(atom_types),
        bonds=bonds,
        bond_types=bond_types,
        labels=labels,
        metadata={
            "builder": "polymer",
            "architecture": "brush",
            "backbone_length": backbone_length,
            "side_chain_length": side_chain_length,
            "graft_every": graft_every,
            "num_side_chains": len(graft_sites),
            "bond_length": bond_length,
            "walk": walk,
            "fraction_a": fraction_a,
            "seed": seed,
            "graft_sites": graft_sites,
            "side_chain_ranges": side_chain_ranges,
        },
    )
