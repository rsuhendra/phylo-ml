from __future__ import annotations

import unittest

from phylogfn_data.likelihood import optimize_shared_branch_length, poisson_log_likelihood
from phylogfn_data.parsimony import parse_newick


class LikelihoodTests(unittest.TestCase):
    def test_supported_topology_has_better_likelihood(self) -> None:
        sequences = {"a": "AAAA", "b": "AAAA", "c": "CCCC", "d": "CCCC"}
        supported = parse_newick("((a,b),(c,d));")
        unsupported = parse_newick("((a,c),(b,d));")
        self.assertGreater(
            poisson_log_likelihood(supported, sequences, 0.1),
            poisson_log_likelihood(unsupported, sequences, 0.1),
        )
        branch_length, score = optimize_shared_branch_length(supported, sequences)
        self.assertGreater(branch_length, 0)
        self.assertLess(score, 0)


if __name__ == "__main__":
    unittest.main()
