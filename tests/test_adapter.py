from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class AdapterTests(unittest.TestCase):
    def test_residue_adapter_preserves_leaf_axis_and_learns_site_weights(self) -> None:
        from phylogfn_data.adapter import ResidueAwareAdapter

        adapter = ResidueAwareAdapter(esm_dim=6, hidden_dim=8, num_heads=2, dropout=0.0)
        embeddings = torch.randn(4, 5, 6)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1],
            ],
            dtype=torch.bool,
        )
        amino_acids = torch.zeros(4, 5, dtype=torch.long)
        leaf_features, context, weights = adapter(embeddings, mask, amino_acids)
        self.assertEqual(tuple(leaf_features.shape), (4, 8))
        self.assertEqual(tuple(context.shape), (8,))
        self.assertEqual(tuple(weights.shape), (5,))
        self.assertAlmostEqual(float(weights.sum().detach()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
