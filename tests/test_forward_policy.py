from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class ForwardPolicyTests(unittest.TestCase):
    def test_scores_and_samples_all_unordered_pairs(self) -> None:
        from phylogfn.model.forward_policy import ForwardPolicy
        from phylogfn.phylo.tree_env import TreeState

        identifiers = ("a", "b", "c", "d")
        sequences = {"a": "AAAA", "b": "AACA", "c": "CCCC", "d": "CCAC"}
        state = TreeState.initial(identifiers, sequences)
        raw = torch.randn(4, 4, 5)
        pairs = (raw + raw.transpose(0, 1)) / 2
        pairs[torch.arange(4), torch.arange(4)] = 0
        policy = ForwardPolicy(pair_dim=5, hidden_dim=8, fitch_dim=4)
        logits, actions = policy.logits(state, pairs, identifiers)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(len(actions), 3)
        action, log_probability = policy.sample_action(state, pairs, identifiers)
        self.assertIn(action, actions)
        self.assertEqual(log_probability.ndim, 0)

    def test_incremental_cache_matches_rebuilt_child_state(self) -> None:
        from phylogfn.model.forward_policy import ForwardPolicy
        from phylogfn.phylo.tree_env import TreeState

        identifiers = ("a", "b", "c", "d", "e")
        sequences = {
            "a": "AA-CA",
            "b": "AACCA",
            "c": "CCACA",
            "d": "CC-AC",
            "e": "CAACC",
        }
        state = TreeState.initial(identifiers, sequences)
        raw = torch.randn(5, 5, 6, requires_grad=True)
        pairs = (raw + raw.transpose(0, 1)) / 2
        policy = ForwardPolicy(pair_dim=6, hidden_dim=9, fitch_dim=5)

        cache = policy.initialize_state_cache(state, pairs, identifiers)
        policy.logits(state, pairs, identifiers, cache=cache)
        child = state.step((1, 3))
        incremental = policy.advance_state_cache(cache, (1, 3), child)
        rebuilt = policy.initialize_state_cache(child, pairs, identifiers)

        self.assertEqual(incremental.subtree_leaves, rebuilt.subtree_leaves)
        torch.testing.assert_close(incremental.sizes, rebuilt.sizes)
        torch.testing.assert_close(incremental.pair_sums, rebuilt.pair_sums)
        torch.testing.assert_close(incremental.anchor_sums, rebuilt.anchor_sums)
        torch.testing.assert_close(incremental.fitch_states, rebuilt.fitch_states)
        torch.testing.assert_close(incremental.fitch_valid, rebuilt.fitch_valid)
        torch.testing.assert_close(incremental.fitch_scores, rebuilt.fitch_scores)
        incremental_logits, incremental_actions = policy.logits(
            child, pairs, identifiers, cache=incremental
        )
        rebuilt_logits, rebuilt_actions = policy.logits(
            child, pairs, identifiers, cache=rebuilt
        )
        self.assertEqual(incremental_actions, rebuilt_actions)
        torch.testing.assert_close(incremental_logits, rebuilt_logits)

    def test_vectorized_action_order_matches_tree_environment(self) -> None:
        from phylogfn.model.forward_policy import ForwardPolicy
        from phylogfn.phylo.tree_env import TreeState

        identifiers = ("a", "b", "c", "d", "e", "f")
        sequences = {identifier: "ACDE" for identifier in identifiers}
        state = TreeState.initial(identifiers, sequences)
        pairs = torch.randn(6, 6, 3)
        policy = ForwardPolicy(pair_dim=3, hidden_dim=8, fitch_dim=4)
        cache = policy.initialize_state_cache(state, pairs, identifiers)
        _, left, right = policy._cached_logits(cache)
        tensor_actions = tuple(zip(left.tolist(), right.tolist()))
        self.assertEqual(tensor_actions, state.valid_actions())

    def test_only_new_candidate_logits_are_scored_after_each_merge(self) -> None:
        from phylogfn.model.forward_policy import ForwardPolicy
        from phylogfn.phylo.tree_env import TreeState

        identifiers = tuple("abcdefg")
        sequences = {identifier: "ACDEFG" for identifier in identifiers}
        state = TreeState.initial(identifiers)
        pairs = torch.randn(7, 7, 4)
        policy = ForwardPolicy(pair_dim=4, hidden_dim=8, fitch_dim=4)
        cache = policy.initialize_state_cache(
            state, pairs, identifiers, sequences=sequences
        )
        scored_rows: list[int] = []

        def count_rows(_module, inputs, _output):
            scored_rows.append(inputs[0].shape[0])

        handle = policy.action_head.register_forward_hook(count_rows)
        try:
            while not state.is_terminal:
                policy.logits(state, pairs, identifiers, cache=cache)
                # A second query of the same state must be a pure cache hit.
                policy.logits(state, pairs, identifiers, cache=cache)
                child = state.step((0, 1))
                cache = policy.advance_state_cache(cache, (0, 1), child)
                state = child
        finally:
            handle.remove()

        self.assertEqual(scored_rows, [15, 4, 3, 2, 1])
        self.assertEqual(sum(scored_rows), 25)

    def test_tensor_fitch_score_matches_independent_cpu_tree_state(self) -> None:
        from phylogfn.model.forward_policy import ForwardPolicy
        from phylogfn.phylo.tree_env import TreeState

        identifiers = tuple("abcde")
        sequences = {
            "a": "AA-CA",
            "b": "AACCA",
            "c": "CCACA",
            "d": "CC-AC",
            "e": "CAACC",
        }
        topology_state = TreeState.initial(identifiers)
        cpu_fitch_state = TreeState.initial(identifiers, sequences)
        pairs = torch.randn(5, 5, 3)
        policy = ForwardPolicy(pair_dim=3, hidden_dim=8, fitch_dim=4)
        cache = policy.initialize_state_cache(
            topology_state, pairs, identifiers, sequences=sequences
        )
        while not topology_state.is_terminal:
            action = (0, 1)
            topology_child = topology_state.step(action)
            cache = policy.advance_state_cache(cache, action, topology_child)
            topology_state = topology_child
            cpu_fitch_state = cpu_fitch_state.step(action)

        self.assertIsNone(topology_state.anchor.fitch)
        self.assertIsNone(topology_state.forest[0].fitch)
        self.assertEqual(
            policy.terminal_fitch_score(cache), cpu_fitch_state.terminal_fitch().score
        )


if __name__ == "__main__":
    unittest.main()
