"""Command-line interface for polygraft_builder."""

from __future__ import annotations

import argparse
from pathlib import Path

from .grafting import build_grafted_nanoparticle
from .io import write_system_outputs
from .nanoparticle import build_nanoparticle
from .sequence import generate_ising_sequence, generate_random_sequence, save_sequence_outputs
from .walks import build_chain


def main(argv: list[str] | None = None) -> None:
    """Run the ``polygraft`` command."""

    parser = argparse.ArgumentParser(prog="polygraft")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sequence_parser = subparsers.add_parser("sequence", help="generate a binary polymer sequence")
    sequence_parser.add_argument("--chain-length", type=int, required=True)
    sequence_parser.add_argument("--fraction-a", type=float, default=0.5)
    sequence_parser.add_argument("--mode", choices=["random", "ising"], default="random")
    sequence_parser.add_argument("--coupling", type=float, default=0.8)
    sequence_parser.add_argument("--mc-steps", type=int, default=2000)
    sequence_parser.add_argument("--seed", type=int)
    sequence_parser.add_argument("--out", default="sequence.txt")
    sequence_parser.set_defaults(func=_sequence_command)

    chain_parser = subparsers.add_parser("chain", help="build a single polymer chain")
    chain_parser.add_argument("--chain-length", type=int)
    chain_parser.add_argument("--sequence")
    chain_parser.add_argument("--fraction-a", type=float, default=0.5)
    chain_parser.add_argument("--bond-length", type=float, default=1.0)
    chain_parser.add_argument("--walk", choices=["random", "saw"], default="random")
    chain_parser.add_argument("--min-distance", type=float, default=0.8)
    chain_parser.add_argument("--max-retries", type=int, default=5000)
    chain_parser.add_argument("--seed", type=int)
    chain_parser.add_argument("--out", default="polymer.xyz")
    chain_parser.set_defaults(func=_chain_command)

    np_parser = subparsers.add_parser("nanoparticle", help="build a spherical nanoparticle")
    np_parser.add_argument("--np-radius", type=float, required=True)
    np_parser.add_argument("--np-bead-spacing", type=float, default=1.0)
    np_parser.add_argument("--surface-only", type=_parse_bool, default=False)
    np_parser.add_argument("--out", default="nanoparticle.xyz")
    np_parser.set_defaults(func=_nanoparticle_command)

    graft_parser = subparsers.add_parser("graft", help="build a polymer-grafted nanoparticle")
    graft_parser.add_argument("--np-radius", type=float, required=True)
    graft_parser.add_argument("--np-bead-spacing", type=float, default=1.0)
    graft_parser.add_argument("--num-grafts", type=int, required=True)
    graft_parser.add_argument("--chain-length", type=int, required=True)
    graft_parser.add_argument("--bond-length", type=float, default=1.0)
    graft_parser.add_argument("--walk", choices=["random", "saw"], default="random")
    graft_parser.add_argument("--graft-mode", choices=["random", "uniform"], default="random")
    graft_parser.add_argument("--min-site-distance", type=float, default=0.0)
    graft_parser.add_argument("--min-distance", type=float, default=0.8)
    graft_parser.add_argument("--fraction-a", type=float, default=0.5)
    graft_parser.add_argument("--max-retries", type=int, default=5000)
    graft_parser.add_argument("--seed", type=int)
    graft_parser.add_argument("--out", default="grafted_np.lammps")
    graft_parser.set_defaults(func=_graft_command)

    args = parser.parse_args(argv)
    args.func(args)


def _sequence_command(args: argparse.Namespace) -> None:
    if args.mode == "random":
        sequence = generate_random_sequence(args.chain_length, args.fraction_a, args.seed)
    else:
        sequence = generate_ising_sequence(
            args.chain_length,
            args.fraction_a,
            coupling=args.coupling,
            mc_steps=args.mc_steps,
            seed=args.seed,
        )
    paths = save_sequence_outputs(
        sequence,
        args.out,
        {
            "builder": "sequence",
            "chain_length": args.chain_length,
            "fraction_a": args.fraction_a,
            "mode": args.mode,
            "coupling": args.coupling,
            "mc_steps": args.mc_steps,
            "seed": args.seed,
        },
    )
    print(f"wrote {paths['txt']}, {paths['npy']}, {paths['json']}")


def _chain_command(args: argparse.Namespace) -> None:
    system = build_chain(
        chain_length=args.chain_length,
        sequence=args.sequence,
        fraction_a=args.fraction_a,
        bond_length=args.bond_length,
        walk=args.walk,
        min_distance=args.min_distance,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    paths = write_system_outputs(system, args.out)
    print(f"wrote {paths['xyz']}, {paths['lammps']}, {paths['json']}")


def _nanoparticle_command(args: argparse.Namespace) -> None:
    system = build_nanoparticle(args.np_radius, args.np_bead_spacing, args.surface_only)
    paths = write_system_outputs(system, args.out)
    print(f"wrote {paths['xyz']}, {paths['lammps']}, {paths['json']}")


def _graft_command(args: argparse.Namespace) -> None:
    system = build_grafted_nanoparticle(
        np_radius=args.np_radius,
        np_bead_spacing=args.np_bead_spacing,
        num_grafts=args.num_grafts,
        chain_length=args.chain_length,
        bond_length=args.bond_length,
        walk=args.walk,
        graft_mode=args.graft_mode,
        min_site_distance=args.min_site_distance,
        min_distance=args.min_distance,
        fraction_a=args.fraction_a,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    paths = write_system_outputs(system, args.out)
    print(f"wrote {paths['xyz']}, {paths['lammps']}, {paths['json']}")


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


if __name__ == "__main__":
    main()
