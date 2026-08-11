"""Root-invariant split and Robinson--Foulds metrics for gene trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .parsimony import TreeNode, leaf_names, parse_newick


Split = tuple[str, ...]


def unrooted_splits(tree: TreeNode) -> frozenset[Split]:
    """Extract canonical nontrivial bipartitions independent of root placement."""

    leaves = frozenset(leaf_names(tree))
    if len(leaves) < 3:
        raise ValueError("Tree must contain at least three leaves")
    splits: set[Split] = set()

    def visit(node: TreeNode) -> frozenset[str]:
        """Collect descendant leaves and record the corresponding edge split."""

        if node.is_leaf:
            assert node.name is not None
            return frozenset((node.name,))
        descendants = frozenset().union(*(visit(child) for child in node.children))
        if node is not tree:
            complement = leaves - descendants
            if len(descendants) >= 2 and len(complement) >= 2:
                left = tuple(sorted(descendants))
                right = tuple(sorted(complement))
                splits.add(min(left, right))
        return descendants

    observed = visit(tree)
    if observed != leaves:
        raise RuntimeError("Tree traversal lost leaves")
    return frozenset(splits)


def robinson_foulds(first: TreeNode, second: TreeNode) -> tuple[int, float]:
    """Return raw and normalized unrooted Robinson--Foulds distance."""

    first_leaves = set(leaf_names(first))
    second_leaves = set(leaf_names(second))
    if first_leaves != second_leaves:
        raise ValueError("RF distance requires identical leaf sets")
    first_splits = unrooted_splits(first)
    second_splits = unrooted_splits(second)
    distance = len(first_splits ^ second_splits)
    denominator = len(first_splits) + len(second_splits)
    return distance, distance / denominator if denominator else 0.0


def main(argv: list[str] | None = None) -> int:
    """Compare a predicted topology with a reference Newick tree."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    tree = parse_newick(args.tree.read_text(encoding="utf-8"))
    reference = parse_newick(args.reference.read_text(encoding="utf-8"))
    raw, normalized = robinson_foulds(tree, reference)
    result = {"rf_distance": raw, "normalized_rf": normalized}
    print(json.dumps(result) if args.json else f"RF={raw} normalized_RF={normalized:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
