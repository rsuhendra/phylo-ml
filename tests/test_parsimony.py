from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phylogfn.phylo.parsimony import (
    normalized_parsimony_score,
    parse_newick,
    parsimony_log_reward,
    parsimony_score,
    score_files,
)


class ParsimonyTests(unittest.TestCase):
    def test_scores_informative_binary_character(self) -> None:
        tree = parse_newick("((a,b),(c,d));")
        self.assertEqual(parsimony_score(tree, {"a": "A", "b": "A", "c": "C", "d": "C"}), 1)

    def test_distinguishes_supported_and_unsupported_splits(self) -> None:
        sequences = {"a": "A", "b": "A", "c": "C", "d": "C"}
        supported = parsimony_score(parse_newick("((a,b),(c,d));"), sequences)
        unsupported = parsimony_score(parse_newick("((a,c),(b,d));"), sequences)
        self.assertEqual((supported, unsupported), (1, 2))

    def test_normalized_reward_is_comparable_across_repeated_sites(self) -> None:
        tree = parse_newick("((a,b),(c,d));")
        short = {"a": "A", "b": "A", "c": "C", "d": "C"}
        long = {name: sequence * 10 for name, sequence in short.items()}
        self.assertAlmostEqual(
            normalized_parsimony_score(tree, short),
            normalized_parsimony_score(tree, long),
        )
        self.assertAlmostEqual(
            parsimony_log_reward(tree, short),
            -normalized_parsimony_score(tree, short),
        )

    def test_missing_and_ambiguous_states_do_not_force_changes(self) -> None:
        tree = parse_newick("(a,b,c);")
        self.assertEqual(parsimony_score(tree, {"a": "A-", "b": "AX", "c": "AA"}), 0)
        self.assertEqual(parsimony_score(tree, {"a": "D", "b": "B", "c": "N"}), 1)

    def test_accepts_branch_lengths_internal_labels_and_quotes(self) -> None:
        tree = parse_newick("(('a one':0.1,b:2)e:0.5,c:1.0)root;")
        self.assertEqual(parsimony_score(tree, {"a one": "A", "b": "A", "c": "C"}), 1)

    def test_rejects_leaf_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "leaf mismatch"):
            parsimony_score(parse_newick("(a,c);"), {"a": "A", "b": "A"})

    def test_scores_fasta_and_newick_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alignment = root / "alignment.fasta"
            tree = root / "tree.nwk"
            alignment.write_text(">a\nAA\n>b\nAA\n>c\nCC\n", encoding="utf-8")
            tree.write_text("(a,b,c);\n", encoding="utf-8")
            self.assertEqual(score_files(alignment, tree), 2)


if __name__ == "__main__":
    unittest.main()
