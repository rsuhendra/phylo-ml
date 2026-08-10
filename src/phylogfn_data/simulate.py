"""Generate controlled protein-family alignments with known true gene trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fasta import FastaRecord, write_fasta
from .parsimony import AMINO_ACIDS, TreeNode, tree_to_newick
from .tree_env import TreeState


def random_tree(leaf_names: list[str], generator: np.random.Generator) -> TreeNode:
    state = TreeState.initial(leaf_names)
    while not state.is_terminal:
        actions = state.valid_actions()
        state = state.step(actions[int(generator.integers(len(actions)))])
    return state.terminal_tree()


def simulate_sequences(
    tree: TreeNode,
    *,
    sequence_length: int,
    branch_length: float,
    generator: np.random.Generator,
) -> dict[str, str]:
    if sequence_length < 1 or branch_length <= 0:
        raise ValueError("Sequence length and branch length must be positive")
    alphabet = np.asarray(AMINO_ACIDS)
    decay = np.exp(-(20.0 / 19.0) * branch_length)
    same_probability = 1.0 / 20.0 + 19.0 * decay / 20.0
    root_sequence = generator.integers(0, 20, size=sequence_length)
    leaves: dict[str, str] = {}

    def mutate(parent: np.ndarray) -> np.ndarray:
        child = parent.copy()
        changes = generator.random(sequence_length) >= same_probability
        for position in np.flatnonzero(changes):
            choices = np.concatenate((np.arange(child[position]), np.arange(child[position] + 1, 20)))
            child[position] = generator.choice(choices)
        return child

    def visit(node: TreeNode, sequence: np.ndarray) -> None:
        if node.is_leaf:
            assert node.name is not None
            leaves[node.name] = "".join(alphabet[sequence])
            return
        for child in node.children:
            visit(child, mutate(sequence))

    visit(tree, root_sequence)
    return leaves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-families", type=int, default=100)
    parser.add_argument("--min-leaves", type=int, default=8)
    parser.add_argument("--max-leaves", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--min-branch-length", type=float, default=0.02)
    parser.add_argument("--max-branch-length", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_families < 1 or not 3 <= args.min_leaves <= args.max_leaves:
        raise SystemExit("Require positive families and 3 <= min leaves <= max leaves")
    generator = np.random.default_rng(args.seed)
    aligned_dir = args.output_dir / "aligned"
    raw_dir = args.output_dir / "raw"
    trees_dir = args.output_dir / "trees"
    manifest = []
    for family_index in range(args.num_families):
        family_id = f"sim_{family_index:06d}"
        num_leaves = int(generator.integers(args.min_leaves, args.max_leaves + 1))
        branch_length = float(
            np.exp(
                generator.uniform(
                    np.log(args.min_branch_length), np.log(args.max_branch_length)
                )
            )
        )
        names = [f"taxon_{index:04d}" for index in range(num_leaves)]
        tree = random_tree(names, generator)
        sequences = simulate_sequences(
            tree,
            sequence_length=args.sequence_length,
            branch_length=branch_length,
            generator=generator,
        )
        records = [FastaRecord(name, name, sequences[name]) for name in sorted(sequences)]
        alignment_path = aligned_dir / f"{family_id}.fasta"
        raw_path = raw_dir / f"{family_id}.fasta"
        tree_path = trees_dir / f"{family_id}.nwk"
        write_fasta(records, alignment_path)
        write_fasta(records, raw_path)
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_path.write_text(tree_to_newick(tree) + "\n", encoding="utf-8")
        manifest.append(
            {
                "family_id": family_id,
                "num_sequences": num_leaves,
                "alignment_length": args.sequence_length,
                "branch_length": branch_length,
                "alignment": str(alignment_path),
                "raw": str(raw_path),
                "true_tree": str(tree_path),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
