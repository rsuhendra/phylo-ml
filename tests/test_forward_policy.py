from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class ForwardPolicyTests(unittest.TestCase):
    def test_scores_and_samples_all_unordered_pairs(self) -> None:
        from phylogfn_data.forward_policy import ForwardPolicy
        from phylogfn_data.tree_env import TreeState

        state = TreeState.initial(["a", "b", "c", "d"])
        features = {name: torch.randn(5) for name in ("a", "b", "c", "d")}
        policy = ForwardPolicy(input_dim=5, hidden_dim=8)
        logits, actions = policy.logits(state, features)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(len(actions), 3)
        action, log_probability = policy.sample_action(state, features)
        self.assertIn(action, actions)
        self.assertEqual(log_probability.ndim, 0)


if __name__ == "__main__":
    unittest.main()
