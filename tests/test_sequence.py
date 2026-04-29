import json

import numpy as np

from polygraft_builder.sequence import (
    generate_ising_sequence,
    generate_random_sequence,
    parse_sequence,
    save_sequence_outputs,
)


def test_random_sequence_has_correct_length_and_fraction():
    sequence = generate_random_sequence(10, fraction_a=0.3, seed=1)

    assert len(sequence) == 10
    assert np.sum(sequence == 1) == 3
    assert set(sequence.tolist()) == {-1, 1}


def test_ising_sequence_preserves_composition():
    sequence = generate_ising_sequence(20, fraction_a=0.5, seed=2, mc_steps=100)

    assert len(sequence) == 20
    assert np.sum(sequence == 1) == 10


def test_parse_sequence_ab_and_numeric():
    assert parse_sequence("AABBA").tolist() == [1, 1, -1, -1, 1]
    assert parse_sequence("+1 -1 1").tolist() == [1, -1, 1]


def test_sequence_outputs_are_written(tmp_path):
    sequence = generate_random_sequence(6, seed=1)
    paths = save_sequence_outputs(sequence, tmp_path / "sequence.txt", {"seed": 1})

    assert (tmp_path / "sequence.txt").exists()
    assert (tmp_path / "sequence.npy").exists()
    assert (tmp_path / "sequence.json").exists()
    metadata = json.loads((tmp_path / "sequence.json").read_text())
    assert metadata["seed"] == 1
    assert set(paths) == {"txt", "npy", "json"}
