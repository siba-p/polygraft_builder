from polygraft_builder.architectures import build_bottlebrush_polymer, build_polymer, build_star_polymer
from polygraft_builder.io import write_system_outputs


def test_star_polymer_atom_and_bond_counts():
    system = build_star_polymer(num_arms=3, arm_length=4, walk="saw", seed=1)

    assert system.n_atoms == 1 + 3 * 4
    assert system.n_bonds == 3 * 4
    assert system.metadata["architecture"] == "star"
    assert len(system.metadata["arm_ranges"]) == 3


def test_bottlebrush_polymer_atom_and_bond_counts():
    system = build_bottlebrush_polymer(
        backbone_length=6,
        side_chain_length=3,
        graft_every=2,
        walk="saw",
        seed=1,
    )

    assert system.n_atoms == 6 + 3 * 3
    assert system.n_bonds == 5 + 3 * 3
    assert system.metadata["architecture"] == "brush"
    assert system.metadata["num_side_chains"] == 3


def test_build_polymer_dispatches_architectures():
    star = build_polymer(architecture="star", arm_length=3, num_arms=2, seed=1)
    brush = build_polymer(architecture="brush", backbone_length=5, side_chain_length=2, seed=1)

    assert star.metadata["architecture"] == "star"
    assert brush.metadata["architecture"] == "brush"


def test_gromacs_outputs_are_written_for_architecture(tmp_path):
    system = build_polymer(architecture="star", arm_length=3, num_arms=3, seed=1)
    write_system_outputs(system, tmp_path / "star.gro")

    assert (tmp_path / "star.xyz").exists()
    assert (tmp_path / "star.lammps").exists()
    assert (tmp_path / "star.gro").exists()
    assert (tmp_path / "star.itp").exists()
    assert (tmp_path / "star.top").exists()
    assert (tmp_path / "star.json").exists()
