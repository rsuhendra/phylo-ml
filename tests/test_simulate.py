from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phylogfn.phylo.parsimony import leaf_names
from phylogfn.data.simulate import main, random_tree, simulate_sequences


class SimulationTests(unittest.TestCase):
    def test_simulation_has_known_tree_and_aligned_leaves(self) -> None:
        generator = np.random.default_rng(3)
        names = ["a", "b", "c", "d", "e"]
        tree = random_tree(names, generator)
        sequences = simulate_sequences(
            tree, sequence_length=30, branch_length=0.1, generator=generator
        )
        self.assertEqual(set(leaf_names(tree)), set(names))
        self.assertEqual(set(sequences), set(names))
        self.assertEqual({len(sequence) for sequence in sequences.values()}, {30})

    def test_cli_layout_has_unique_family_filenames(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            self.assertEqual(
                main(
                    [
                        "--output-dir",
                        str(output_dir),
                        "--num-families",
                        "2",
                        "--min-leaves",
                        "4",
                        "--max-leaves",
                        "4",
                        "--sequence-length",
                        "12",
                    ]
                ),
                0,
            )
            for family_id in ("sim_000000", "sim_000001"):
                self.assertTrue((output_dir / "aligned" / f"{family_id}.fasta").is_file())
                self.assertTrue((output_dir / "raw" / f"{family_id}.fasta").is_file())
                self.assertTrue((output_dir / "trees" / f"{family_id}.nwk").is_file())


if __name__ == "__main__":
    unittest.main()
