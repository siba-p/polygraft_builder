# polygraft_builder

`polygraft_builder` builds coarse-grained polymer chains and polymer-grafted
nanoparticles for polymer-surface and polymer-nanoparticle simulation setup.

The code is a clean, minimal rewrite of useful concepts from earlier
ML_surfacepoly and KG_PGNs workflows:

- binary A/B polymer sequences represented as `+1/-1`
- random and Ising-like correlated sequence generation
- random-walk and self-avoiding-walk chain construction
- spherical nanoparticle bead generation and surface-site detection
- simple grafting of many chains to nanoparticle surface beads
- XYZ, LAMMPS data, and JSON metadata output

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

Build a grafted nanoparticle:

```bash
polygraft graft --np-radius 5.0 --num-grafts 60 --chain-length 20 --walk saw --graft-mode random --seed 1 --out grafted_np.lammps
```

Build only the nanoparticle:

```bash
polygraft nanoparticle --np-radius 5.0 --np-bead-spacing 1.0 --surface-only false --out nanoparticle.xyz
```

## Outputs

Builder commands write a complete trio of files:

- `.xyz` for quick visualization
- `.lammps` for a minimal LAMMPS molecular data file
- `.json` for metadata, sequence, atom types, bonds, and selected graft sites

Sequence generation writes:

- `.txt` with A/B labels
- `.npy` with `+1/-1` integer values
- `.json` metadata

## Python API

```python
from polygraft_builder import build_chain, build_grafted_nanoparticle
from polygraft_builder.io import write_system_outputs

chain = build_chain(chain_length=40, walk="saw", seed=1)
write_system_outputs(chain, "polymer.xyz")

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

Bond type `1` is used for polymer backbone bonds, and bond type `2` connects the
first polymer bead to its grafting site.

## Tests

```bash
cd polygraft_builder
pytest
```
