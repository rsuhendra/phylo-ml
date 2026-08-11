"""Distance-based Neighbor-Joining baseline for aligned proteins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data.fasta import GAP_CHARS, read_fasta
from .parsimony import TreeNode, tree_to_newick


def p_distance(first: str, second: str) -> float:
    """Compute mismatch fraction over comparable nongap, non-unknown sites."""

    comparable = [
        (left, right)
        for left, right in zip(first, second)
        if left not in GAP_CHARS and right not in GAP_CHARS and left != "X" and right != "X"
    ]
    if not comparable:
        return 1.0
    return sum(left != right for left, right in comparable) / len(comparable)


def neighbor_joining(names: list[str], distances: np.ndarray) -> TreeNode:
    """Construct a deterministic Neighbor-Joining topology from a distance matrix."""

    if len(names) < 3 or distances.shape != (len(names), len(names)):
        raise ValueError("Neighbor Joining requires an n-by-n matrix for at least three leaves")
    active = list(range(len(names)))
    nodes = {index: TreeNode(name=name) for index, name in enumerate(names)}
    matrix = {
        (min(i, j), max(i, j)): float(distances[i, j])
        for i in active
        for j in active
        if i != j
    }
    next_index = len(names)

    def distance(i: int, j: int) -> float:
        """Read a symmetric active-cluster distance."""

        return 0.0 if i == j else matrix[(min(i, j), max(i, j))]

    while len(active) > 2:
        count = len(active)
        totals = {i: sum(distance(i, j) for j in active if j != i) for i in active}
        first, second = min(
            ((i, j) for position, i in enumerate(active) for j in active[position + 1 :]),
            key=lambda pair: (
                (count - 2) * distance(*pair) - totals[pair[0]] - totals[pair[1]],
                pair,
            ),
        )
        new_distances = {
            other: 0.5
            * (distance(first, other) + distance(second, other) - distance(first, second))
            for other in active
            if other not in (first, second)
        }
        nodes[next_index] = TreeNode(children=[nodes[first], nodes[second]])
        active = [item for item in active if item not in (first, second)]
        for other, value in new_distances.items():
            matrix[(min(other, next_index), max(other, next_index))] = value
        active.append(next_index)
        next_index += 1
    return TreeNode(children=[nodes[active[0]], nodes[active[1]]])


def main(argv: list[str] | None = None) -> int:
    """Build and write a Neighbor-Joining baseline from an aligned FASTA."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = read_fasta(args.alignment)
    distances = np.zeros((len(records), len(records)), dtype=np.float64)
    for i, first in enumerate(records):
        for j in range(i + 1, len(records)):
            distances[i, j] = distances[j, i] = p_distance(first.sequence, records[j].sequence)
    tree = neighbor_joining([record.identifier for record in records], distances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tree_to_newick(tree) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
