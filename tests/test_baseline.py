from __future__ import annotations

import unittest

import numpy as np

from phylogfn_data.baseline import neighbor_joining
from phylogfn_data.parsimony import leaf_names


class BaselineTests(unittest.TestCase):
    def test_neighbor_joining_preserves_leaf_set(self) -> None:
        names = ["a", "b", "c", "d"]
        distances = np.array(
            [[0, 1, 5, 5], [1, 0, 5, 5], [5, 5, 0, 1], [5, 5, 1, 0]],
            dtype=float,
        )
        tree = neighbor_joining(names, distances)
        self.assertEqual(set(leaf_names(tree)), set(names))


if __name__ == "__main__":
    unittest.main()
