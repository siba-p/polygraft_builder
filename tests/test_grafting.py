import pytest

from polygraft_builder.grafting import build_grafted_nanoparticle, select_graft_sites
from polygraft_builder.io import write_system_outputs
from polygraft_builder.nanoparticle import build_nanoparticle


def test_nanoparticle_detects_surface_sites():
    nanoparticle = build_nanoparticle(np_radius=2.0, np_bead_spacing=1.0)

    assert nanoparticle.n_atoms > 0
    assert nanoparticle.metadata["num_surface_sites"] > 0


def test_grafted_nanoparticle_has_correct_number_of_chains():
    system = build_grafted_nanoparticle(
        np_radius=3.0,
        np_bead_spacing=1.0,
        num_grafts=4,
        chain_length=5,
        walk="saw",
        seed=1,
    )

    assert len(system.metadata["chain_ranges"]) == 4
    assert len(system.metadata["graft_site_indices"]) == 4
    assert system.n_bonds >= 4 * 5


def test_num_grafts_cannot_exceed_sites():
    nanoparticle = build_nanoparticle(np_radius=1.0, np_bead_spacing=1.0)
    surface_positions = nanoparticle.positions[nanoparticle.metadata["surface_site_indices"]]

    with pytest.raises(ValueError, match="cannot exceed"):
        select_graft_sites(surface_positions, len(surface_positions) + 1)


def test_grafted_output_files_are_written(tmp_path):
    system = build_grafted_nanoparticle(
        np_radius=2.5,
        num_grafts=3,
        chain_length=4,
        walk="random",
        seed=1,
    )
    write_system_outputs(system, tmp_path / "grafted_np.lammps")

    assert (tmp_path / "grafted_np.xyz").exists()
    assert (tmp_path / "grafted_np.lammps").exists()
    assert (tmp_path / "grafted_np.gro").exists()
    assert (tmp_path / "grafted_np.itp").exists()
    assert (tmp_path / "grafted_np.top").exists()
    assert (tmp_path / "grafted_np.json").exists()
