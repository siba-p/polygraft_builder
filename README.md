# polygraft_builder

`polygraft_builder` builds coarse-grained polymer chains and polymer-grafted
nanoparticles for polymer-surface and polymer-nanoparticle simulation setup.

The code is a clean, minimal rewrite of useful concepts from earlier
ML_surfacepoly and KG_PGNs workflows:

- binary A/B polymer sequences represented as `+1/-1`
- random and Ising-like correlated sequence generation
- random-walk and self-avoiding-walk chain construction
- linear, star, and bottlebrush polymer architectures
- spherical nanoparticle bead generation and surface-site detection
- simple grafting of many chains to nanoparticle surface beads
- XYZ, LAMMPS data, GROMACS, and JSON metadata output

## Install

```bash
cd polygraft_builder
python -m pip install -e ".[test]"
```

## CLI Examples

Generate a correlated binary sequence:

```bash
polygraft sequence --chain-length 40 --fraction-a 0.5 --mode ising --seed 1 --out sequence.txt
```

Build a single random-walk chain:

```bash
polygraft chain --chain-length 40 --walk random --bond-length 1.0 --seed 1 --out polymer.xyz
```

Build a single self-avoiding-walk chain:

```bash
polygraft chain --chain-length 40 --walk saw --bond-length 1.0 --min-distance 0.8 --seed 1 --out polymer_saw.xyz
```

Build a star polymer:

```bash
polygraft chain --architecture star --num-arms 6 --arm-length 12 --walk saw --seed 1 --out star.gro
```

Build a bottlebrush polymer:

```bash
polygraft chain --architecture brush --backbone-length 30 --side-chain-length 6 --graft-every 3 --walk saw --seed 1 --out brush.gro
```

Build a grafted nanoparticle:

```bash
polygraft graft --np-radius 5.0 --num-grafts 60 --chain-length 20 --walk saw --graft-mode random --seed 1 --out grafted_np.lammps
```

Build only the nanoparticle:

```bash
polygraft nanoparticle --np-radius 5.0 --np-bead-spacing 1.0 --surface-only false --out nanoparticle.xyz
```

## Outputs

Builder commands write a complete portable output set:

- `.xyz` for quick visualization
- `.lammps` for a minimal LAMMPS molecular data file
- `.gro` for GROMACS coordinates
- `.itp` for a minimal GROMACS molecule include file
- `.top` for a minimal GROMACS topology that includes the `.itp`
- `.json` for metadata, sequence, atom types, bonds, and selected graft sites

Sequence generation writes:

- `.txt` with A/B labels
- `.npy` with `+1/-1` integer values
- `.json` metadata

## Python API

```python
from polygraft_builder import build_chain, build_grafted_nanoparticle, build_star_polymer
from polygraft_builder.io import write_system_outputs

chain = build_chain(chain_length=40, walk="saw", seed=1)
write_system_outputs(chain, "polymer.xyz")

star = build_star_polymer(num_arms=6, arm_length=12, walk="saw", seed=1)
write_system_outputs(star, "star.gro")

grafted = build_grafted_nanoparticle(
    np_radius=5.0,
    num_grafts=60,
    chain_length=20,
    walk="saw",
    seed=1,
)
write_system_outputs(grafted, "grafted_np.lammps")
```

## Notes

This package intentionally avoids force-field complexity. Atom types are simple:

- `1`: polymer A bead
- `2`: polymer B bead
- `3`: nanoparticle bead
- `4`: polymer core bead, used by star polymers

Bond type `1` is used for polymer backbone bonds, bond type `2` connects a
polymer bead to a nanoparticle graft site, bond type `3` connects star arms to
the core, and bond type `4` connects bottlebrush side chains to the backbone.

The GROMACS files are deliberately minimal coarse-grained starting points:
coordinates in `.gro`, atoms and bonds in `.itp`, and generic atom types in
`.top`. Add your preferred bonded and nonbonded force-field parameters before
production simulation.

## Tests

```bash
cd polygraft_builder
pytest
```
