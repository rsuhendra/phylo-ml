from __future__ import annotations

import unittest

from phylogfn_data.parsimony import leaf_names, parsimony_score
from phylogfn_data.tree_env import Subtree, TreeState


class TreeEnvironmentTests(unittest.TestCase):
    def test_initial_state_has_every_unordered_pair_action(self) -> None:
        state = TreeState.initial(["d", "b", "a", "c"])
        self.assertEqual(state.anchor.leaves, ("a",))
        self.assertEqual(len(state.valid_actions()), 3)
        self.assertEqual(state.valid_actions()[0], (0, 1))

    def test_trajectory_builds_terminal_tree(self) -> None:
        state = TreeState.initial(["a", "b", "c", "d"])
        state = state.step((1, 2))
        state = state.step((0, 1))
        self.assertTrue(state.is_terminal)
        self.assertEqual(set(leaf_names(state.terminal_tree())), {"a", "b", "c", "d"})
        self.assertEqual(
            parsimony_score(state.terminal_tree(), {"a": "A", "b": "A", "c": "C", "d": "C"}),
            1,
        )

    def test_backward_action_reverses_merge(self) -> None:
        state = TreeState.initial(["a", "b", "c", "d"])
        child, reverse_index = state.step_with_reverse((1, 2))
        self.assertEqual(child.backward_step(reverse_index), state)

    def test_merge_and_newick_are_invariant_to_pair_order(self) -> None:
        a, b = Subtree.leaf("a"), Subtree.leaf("b")
        self.assertEqual(Subtree.merge(a, b), Subtree.merge(b, a))
        self.assertEqual(Subtree.merge(b, a).to_newick(), "(a,b);")

    def test_rejects_invalid_actions(self) -> None:
        state = TreeState.initial(["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "Invalid merge action"):
            state.step((1, 1))


if __name__ == "__main__":
    unittest.main()
