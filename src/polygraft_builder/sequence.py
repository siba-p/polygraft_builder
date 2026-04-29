"""Binary polymer sequence generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ensure_parent, rng_from_seed, validate_chain_length, validate_fraction


def generate_random_sequence(
    chain_length: int,
    fraction_a: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an exact-composition random +1/-1 sequence.

    ``+1`` represents bead A and ``-1`` represents bead B.
    """

    validate_chain_length(chain_length)
    validate_fraction(fraction_a)
    rng = rng_from_seed(seed)
    n_a = int(round(chain_length * fraction_a))
    sequence = np.full(chain_length, -1, dtype=int)
    if n_a:
        sequence[rng.choice(chain_length, size=n_a, replace=False)] = 1
    return sequence


def generate_ising_sequence(
    chain_length: int,
    fraction_a: float = 0.5,
    coupling: float = 0.8,
    mc_steps: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an Ising-like correlated sequence with fixed composition.

    The update swaps unlike sites, preserving the requested A fraction while
    favoring neighboring equal spins when ``coupling`` is positive.
    """

    validate_chain_length(chain_length)
    validate_fraction(fraction_a)
    if mc_steps < 0:
        raise ValueError("mc_steps must be non-negative")
    rng = rng_from_seed(seed)
    sequence = generate_random_sequence(chain_length, fraction_a, seed)
    if mc_steps == 0 or chain_length < 3:
        return sequence

    for _ in range(mc_steps):
        i, j = rng.choice(chain_length, size=2, replace=False)
        if sequence[i] == sequence[j]:
            continue
        old_energy = _local_energy(sequence, i, coupling) + _local_energy(sequence, j, coupling)
        sequence[i], sequence[j] = sequence[j], sequence[i]
        new_energy = _local_energy(sequence, i, coupling) + _local_energy(sequence, j, coupling)
        delta = new_energy - old_energy
        if delta > 0.0 and rng.random() >= np.exp(-delta):
            sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence


def _local_energy(sequence: np.ndarray, index: int, coupling: float) -> float:
    left = sequence[(index - 1) % len(sequence)]
    right = sequence[(index + 1) % len(sequence)]
    return float(-coupling * sequence[index] * (left + right))


def parse_sequence(sequence: str | list[int] | np.ndarray) -> np.ndarray:
    """Parse A/B, +/- strings, or numeric arrays into +1/-1 values."""

    if isinstance(sequence, np.ndarray):
        values = sequence.astype(int)
    elif isinstance(sequence, list):
        values = np.asarray(sequence, dtype=int)
    else:
        raw = sequence.strip()
        if Path(raw).exists():
            raw = Path(raw).read_text(encoding="utf-8").strip()
        tokens = raw.replace(",", " ").split()
        if len(tokens) == 1 and set(tokens[0].upper()) <= {"A", "B", "+", "-", "1"}:
            chars = list(tokens[0])
            values = np.asarray([_parse_sequence_token(char) for char in chars], dtype=int)
        else:
            values = np.asarray([_parse_sequence_token(token) for token in tokens], dtype=int)
    if values.ndim != 1 or len(values) <= 1:
        raise ValueError("sequence must contain more than one bead")
    if not np.all(np.isin(values, [-1, 1])):
        raise ValueError("sequence values must be A/B or +1/-1")
    return values


def _parse_sequence_token(token: str) -> int:
    token = token.strip().upper()
    if token in {"A", "+", "+1", "1"}:
        return 1
    if token in {"B", "-", "-1"}:
        return -1
    raise ValueError(f"unsupported sequence token: {token!r}")


def sequence_to_labels(sequence: np.ndarray) -> list[str]:
    """Convert +1/-1 sequence values to A/B labels."""

    return ["A" if value == 1 else "B" for value in sequence]


def save_sequence_outputs(sequence: np.ndarray, out: str | Path, metadata: dict) -> dict[str, str]:
    """Save sequence text, npy, and JSON metadata next to ``out``."""

    from .io import write_json

    out_path = ensure_parent(out)
    stem = out_path.with_suffix("")
    txt_path = out_path if out_path.suffix == ".txt" else stem.with_suffix(".txt")
    npy_path = stem.with_suffix(".npy")
    json_path = stem.with_suffix(".json")

    txt_path.write_text("".join(sequence_to_labels(sequence)) + "\n", encoding="utf-8")
    np.save(npy_path, sequence)
    write_json(json_path, {"sequence": sequence_to_labels(sequence), **metadata})
    return {"txt": str(txt_path), "npy": str(npy_path), "json": str(json_path)}
