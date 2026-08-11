from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class AdapterTests(unittest.TestCase):
    def test_adapter_returns_symmetric_pair_evidence_without_leaf_pooling(self) -> None:
        from phylogfn.model.adapter import ResiduePairAdapter

        adapter = ResiduePairAdapter(
            esm_dim=6, hidden_dim=8, pair_dim=5, num_heads=2, dropout=0.0
        )
        self.assertFalse(hasattr(adapter, "position_embedding"))
        self.assertFalse(hasattr(adapter, "sequence_encoder"))
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
        pairs, context, residues = adapter(embeddings, mask, amino_acids)
        self.assertEqual(tuple(pairs.shape), (4, 4, 5))
        self.assertEqual(tuple(context.shape), (8,))
        self.assertEqual(tuple(residues.shape), (4, 5, 8))
        torch.testing.assert_close(pairs, pairs.transpose(0, 1))
        torch.testing.assert_close(torch.diagonal(pairs), torch.zeros(5, 4))
        self.assertTrue(torch.equal(residues[1, 2], torch.zeros(8)))


if __name__ == "__main__":
    unittest.main()
