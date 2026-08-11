from __future__ import annotations

import unittest

from phylogfn.phylo.parsimony import parse_newick
from phylogfn.phylo.tree_metrics import robinson_foulds


class TreeMetricTests(unittest.TestCase):
    def test_rf_is_root_invariant(self) -> None:
        first = parse_newick("((a,b),(c,d));")
        rerooted = parse_newick("(a,(b,(c,d)));")
        self.assertEqual(robinson_foulds(first, rerooted), (0, 0.0))

    def test_rf_distinguishes_quartets(self) -> None:
        first = parse_newick("((a,b),(c,d));")
        second = parse_newick("((a,c),(b,d));")
        self.assertEqual(robinson_foulds(first, second), (2, 1.0))


if __name__ == "__main__":
    unittest.main()
