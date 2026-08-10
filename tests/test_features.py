from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phylogfn_data.features import GAP_INDEX, load_aligned_esm2, load_pooled_esm2


class FeatureTests(unittest.TestCase):
    @staticmethod
    def _write_family(root: Path) -> None:
        np.save(
            root / "embeddings.npy",
            np.array([[1, 3], [3, 5], [8, 10]], dtype=np.float16),
        )
        metadata = {
            "family_id": "tiny",
            "records": [
                {
                    "id": "a",
                    "embedding_start": 0,
                    "embedding_stop": 2,
                    "aligned_sequence": "A-C",
                    "ungapped_to_aligned": [0, 2],
                },
                {
                    "id": "b",
                    "embedding_start": 2,
                    "embedding_stop": 3,
                    "aligned_sequence": "--D",
                    "ungapped_to_aligned": [2],
                },
            ],
        }
        (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_mean_pools_residue_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_family(root)
            identifiers, features = load_pooled_esm2(root)
            self.assertEqual(identifiers, ["a", "b"])
            np.testing.assert_allclose(features, [[2, 4], [8, 10]])
            self.assertEqual(features.dtype, np.float32)

    def test_scatter_embeddings_back_to_msa_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_family(root)
            family = load_aligned_esm2(root)
            self.assertEqual(family.identifiers, ("a", "b"))
            self.assertEqual(family.residue_embeddings.shape, (2, 3, 2))
            np.testing.assert_array_equal(family.residue_mask, [[True, False, True], [False, False, True]])
            np.testing.assert_allclose(family.residue_embeddings[0, 1], [0, 0])
            self.assertEqual(family.amino_acid_indices[0, 1], GAP_INDEX)


if __name__ == "__main__":
    unittest.main()
