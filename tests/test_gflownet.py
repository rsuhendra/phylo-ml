from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class GFlowNetTests(unittest.TestCase):
    def test_trajectory_balance_loss_is_finite_and_differentiable(self) -> None:
        from phylogfn.model.gflownet import ConditionalPhyloGFN

        identifiers = ("a", "b", "c", "d")
        model = ConditionalPhyloGFN(
            esm_dim=6,
            adapter_dim=8,
            policy_dim=12,
            num_heads=2,
            dropout=0.0,
        )
        embeddings = torch.randn(4, 5, 6)
        mask = torch.ones(4, 5, dtype=torch.bool)
        amino_acids = torch.zeros(4, 5, dtype=torch.long)
        sequences = {"a": "AAAAA", "b": "AAAAA", "c": "CCCCC", "d": "CCCCC"}
        pair_matrix, log_z = model.encode_family(
            identifiers, embeddings, mask, amino_acids
        )
        trajectory = model.sample_trajectory(
            identifiers, pair_matrix, log_z, sequences, beta=2.0
        )
        self.assertEqual(trajectory.num_actions, 2)
        self.assertIsNone(trajectory.terminal_state.anchor.fitch)
        self.assertIsNone(trajectory.terminal_state.forest[0].fitch)
        self.assertTrue(math.isfinite(float(trajectory.loss.detach())))
        trajectory.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertIsNotNone(model.adapter.pair_site_encoder[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
