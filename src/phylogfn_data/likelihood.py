"""A simple Poisson-20 protein likelihood oracle for topology experiments.

This is a controlled baseline with one shared branch length, not a replacement
for an LG+Gamma implementation such as IQ-TREE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fasta import GAP_CHARS, STANDARD_AMINO_ACIDS, read_fasta
from .parsimony import AMBIGUOUS_STATES, TreeNode, leaf_names, parse_newick


AMINO_ACIDS = tuple(sorted(STANDARD_AMINO_ACIDS))
AMINO_ACID_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def transition_matrix(branch_length: float) -> np.ndarray:
    if branch_length <= 0:
        raise ValueError("Branch length must be positive")
    decay = np.exp(-(20.0 / 19.0) * branch_length)
    different = (1.0 - decay) / 20.0
    same = 1.0 / 20.0 + 19.0 * decay / 20.0
    matrix = np.full((20, 20), different, dtype=np.float64)
    np.fill_diagonal(matrix, same)
    return matrix


def _leaf_partials(sequence: str) -> np.ndarray:
    partials = np.zeros((len(sequence), 20), dtype=np.float64)
    for column, character in enumerate(sequence.upper()):
        if character in GAP_CHARS or character in {"?", "X"}:
            partials[column] = 1.0
            continue
        states = AMBIGUOUS_STATES.get(character, frozenset((character,)))
        for state in states:
            partials[column, AMINO_ACID_INDEX[state]] = 1.0
    return partials


def poisson_log_likelihood(
    tree: TreeNode, sequences: dict[str, str], branch_length: float
) -> float:
    names = leaf_names(tree)
    if set(names) != set(sequences) or len(names) != len(sequences):
        raise ValueError("Tree and alignment must contain identical unique leaf sets")
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ValueError("Alignment sequences have inconsistent lengths")
    transition = transition_matrix(branch_length)

    def partials(node: TreeNode) -> tuple[np.ndarray, np.ndarray]:
        if node.is_leaf:
            assert node.name is not None
            sequence_partials = _leaf_partials(sequences[node.name])
            return sequence_partials, np.zeros(sequence_partials.shape[0], dtype=np.float64)
        combined = np.ones((next(iter(lengths)), 20), dtype=np.float64)
        log_scale = np.zeros(next(iter(lengths)), dtype=np.float64)
        for child in node.children:
            child_partials, child_scale = partials(child)
            combined *= child_partials @ transition.T
            log_scale += child_scale
        scale = combined.sum(axis=1)
        if np.any(scale <= 0):
            return combined, np.full(combined.shape[0], -np.inf)
        combined /= scale[:, None]
        return combined, log_scale + np.log(scale)

    root_partials, scale = partials(tree)
    site_likelihood = root_partials.mean(axis=1)
    return float(np.sum(np.log(site_likelihood) + scale))


def optimize_shared_branch_length(
    tree: TreeNode,
    sequences: dict[str, str],
    candidates: np.ndarray | None = None,
) -> tuple[float, float]:
    if candidates is None:
        candidates = np.geomspace(1e-3, 3.0, 48)
    scored = [
        (float(length), poisson_log_likelihood(tree, sequences, float(length)))
        for length in candidates
    ]
    return max(scored, key=lambda item: item[1])


def normalized_poisson_log_reward(
    tree: TreeNode, sequences: dict[str, str], *, beta: float = 1.0
) -> float:
    _, log_likelihood = optimize_shared_branch_length(tree, sequences)
    observations = len(sequences) * len(next(iter(sequences.values())))
    return beta * log_likelihood / observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    records = read_fasta(args.alignment)
    sequences = {record.identifier: record.sequence for record in records}
    tree = parse_newick(args.tree.read_text(encoding="utf-8"))
    branch_length, log_likelihood = optimize_shared_branch_length(tree, sequences)
    result = {
        "model": "Poisson20-shared-branch-length",
        "branch_length": branch_length,
        "log_likelihood": log_likelihood,
        "mean_log_likelihood_per_observation": log_likelihood
        / (len(records) * len(records[0].sequence)),
    }
    print(json.dumps(result) if args.json else json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
