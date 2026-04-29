"""Random-walk and self-avoiding-walk chain builders."""

from __future__ import annotations

import numpy as np

from .sequence import generate_random_sequence, parse_sequence, sequence_to_labels
from .topology import System
from .utils import pairwise_min_distance, rng_from_seed, validate_chain_length


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly distributed 3D unit vector."""

    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    while norm == 0.0:
        vector = rng.normal(size=3)
        norm = np.linalg.norm(vector)
    return vector / norm


def random_walk(
    chain_length: int,
    bond_length: float = 1.0,
    seed: int | None = None,
    start: np.ndarray | None = None,
    first_direction: np.ndarray | None = None,
) -> np.ndarray:
    """Build an unconstrained 3D random-walk chain."""

    validate_chain_length(chain_length)
    rng = rng_from_seed(seed)
    positions = np.zeros((chain_length, 3), dtype=float)
    if start is not None:
        positions[0] = np.asarray(start, dtype=float)
    for i in range(1, chain_length):
        if i == 1 and first_direction is not None:
            direction = np.asarray(first_direction, dtype=float)
            direction = direction / np.linalg.norm(direction)
        else:
            direction = random_unit_vector(rng)
        positions[i] = positions[i - 1] + bond_length * direction
    return positions


def self_avoiding_walk(
    chain_length: int,
    bond_length: float = 1.0,
    min_distance: float = 0.8,
    seed: int | None = None,
    max_retries: int = 5000,
    start: np.ndarray | None = None,
    first_direction: np.ndarray | None = None,
    forbidden_points: np.ndarray | None = None,
) -> np.ndarray:
    """Build a self-avoiding random walk using rejection sampling."""

    validate_chain_length(chain_length)
    if min_distance <= 0.0:
        raise ValueError("min_distance must be positive")
    rng = rng_from_seed(seed)
    positions = np.zeros((chain_length, 3), dtype=float)
    if start is not None:
        positions[0] = np.asarray(start, dtype=float)
    forbidden = np.empty((0, 3), dtype=float) if forbidden_points is None else np.asarray(forbidden_points, dtype=float)

    for i in range(1, chain_length):
        accepted = False
        for _ in range(max_retries):
            if i == 1 and first_direction is not None:
                direction = np.asarray(first_direction, dtype=float)
                direction = direction / np.linalg.norm(direction)
            else:
                direction = random_unit_vector(rng)
            candidate = positions[i - 1] + bond_length * direction
            own_points = positions[: max(0, i - 1)]
            own_ok = pairwise_min_distance(own_points, candidate) >= min_distance
            forbidden_ok = pairwise_min_distance(forbidden, candidate) >= min_distance
            if own_ok and forbidden_ok:
                positions[i] = candidate
                accepted = True
                break
            if i == 1 and first_direction is not None:
                break
        if not accepted:
            raise RuntimeError(
                f"SAW failed at bead {i + 1}; increase max_retries or lower min_distance"
            )
    return positions


def build_chain(
    chain_length: int | None = None,
    sequence: str | list[int] | np.ndarray | None = None,
    fraction_a: float = 0.5,
    bond_length: float = 1.0,
    walk: str = "random",
    min_distance: float = 0.8,
    seed: int | None = None,
    max_retries: int = 5000,
    start: np.ndarray | None = None,
    first_direction: np.ndarray | None = None,
    forbidden_points: np.ndarray | None = None,
) -> System:
    """Build a single polymer chain system."""

    if sequence is None:
        if chain_length is None:
            raise ValueError("chain_length is required when sequence is not provided")
        seq = generate_random_sequence(chain_length, fraction_a=fraction_a, seed=seed)
    else:
        seq = parse_sequence(sequence)
        if chain_length is not None and chain_length != len(seq):
            raise ValueError("chain_length must match provided sequence length")
    validate_chain_length(len(seq))

    if walk == "random":
        positions = random_walk(len(seq), bond_length, seed, start, first_direction)
    elif walk == "saw":
        positions = self_avoiding_walk(
            len(seq),
            bond_length,
            min_distance,
            seed,
            max_retries,
            start,
            first_direction,
            forbidden_points,
        )
    else:
        raise ValueError("walk must be 'random' or 'saw'")

    atom_types = np.where(seq == 1, 1, 2)
    bonds = [(i, i + 1) for i in range(len(seq) - 1)]
    return System(
        positions=positions,
        atom_types=atom_types,
        bonds=bonds,
        bond_types=[1] * len(bonds),
        labels=sequence_to_labels(seq),
        metadata={
            "builder": "chain",
            "chain_length": len(seq),
            "bond_length": bond_length,
            "walk": walk,
            "sequence": sequence_to_labels(seq),
            "seed": seed,
        },
    )
