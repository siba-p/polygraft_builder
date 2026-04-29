"""Validation and random-number helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def rng_from_seed(seed: int | None) -> np.random.Generator:
    """Create a reproducible numpy random generator."""

    return np.random.default_rng(seed)


def validate_chain_length(chain_length: int) -> None:
    """Validate a polymer chain length."""

    if chain_length <= 1:
        raise ValueError("chain_length must be greater than 1")


def validate_fraction(fraction_a: float) -> None:
    """Validate a binary sequence fraction."""

    if not 0.0 <= fraction_a <= 1.0:
        raise ValueError("fraction_a must be between 0 and 1")


def ensure_parent(path: str | Path) -> Path:
    """Create an output file's parent directory and return the path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def json_ready(value: Any) -> Any:
    """Convert numpy values into JSON-serializable values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def pairwise_min_distance(points: np.ndarray, candidate: np.ndarray) -> float:
    """Return the minimum distance between points and a candidate."""

    if len(points) == 0:
        return float("inf")
    distances = np.linalg.norm(points - candidate, axis=1)
    return float(np.min(distances))
