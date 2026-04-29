"""Small topology container shared by builders and writers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class System:
    """Coarse-grained atoms, bonds, and metadata.

    Atom arrays are 0-based internally. Writers convert to 1-based IDs where
    needed by external formats such as LAMMPS data files.
    """

    positions: np.ndarray
    atom_types: np.ndarray
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_types: list[int] = field(default_factory=list)
    labels: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.atom_types = np.asarray(self.atom_types, dtype=int)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3)")
        if len(self.atom_types) != len(self.positions):
            raise ValueError("atom_types length must match positions")
        if not self.bond_types:
            self.bond_types = [1] * len(self.bonds)
        if len(self.bond_types) != len(self.bonds):
            raise ValueError("bond_types length must match bonds")
        if self.labels is not None and len(self.labels) != len(self.positions):
            raise ValueError("labels length must match positions")

    @property
    def n_atoms(self) -> int:
        """Number of atoms/beads."""

        return int(self.positions.shape[0])

    @property
    def n_bonds(self) -> int:
        """Number of bonds."""

        return len(self.bonds)

    def copy_with_metadata(self, **metadata: Any) -> "System":
        """Return a shallow geometry copy with updated metadata."""

        merged = dict(self.metadata)
        merged.update(metadata)
        return System(
            positions=self.positions.copy(),
            atom_types=self.atom_types.copy(),
            bonds=list(self.bonds),
            bond_types=list(self.bond_types),
            labels=None if self.labels is None else list(self.labels),
            metadata=merged,
        )
