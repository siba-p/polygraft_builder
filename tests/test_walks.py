import numpy as np
import pytest

from polygraft_builder.io import write_system_outputs
from polygraft_builder.walks import build_chain


def test_chain_length_and_bond_count():
    system = build_chain(chain_length=8, walk="random", seed=1)

    assert system.n_atoms == 8
    assert system.n_bonds == 7


def test_saw_has_no_overlapping_nonbonded_beads():
    system = build_chain(chain_length=12, walk="saw", min_distance=0.8, seed=1)

    distances = []
    for i in range(system.n_atoms):
        for j in range(i + 2, system.n_atoms):
            distances.append(np.linalg.norm(system.positions[i] - system.positions[j]))
    assert min(distances) >= 0.8


def test_saw_fails_clearly_when_retries_exceeded():
    with pytest.raises(RuntimeError, match="SAW failed"):
        build_chain(chain_length=5, walk="saw", bond_length=1.0, min_distance=2.0, seed=1, max_retries=2)


def test_chain_output_files_are_written(tmp_path):
    system = build_chain(chain_length=5, seed=1)
    write_system_outputs(system, tmp_path / "polymer.xyz")

    assert (tmp_path / "polymer.xyz").exists()
    assert (tmp_path / "polymer.lammps").exists()
    assert (tmp_path / "polymer.json").exists()
