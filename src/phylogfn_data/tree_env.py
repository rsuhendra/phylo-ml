"""Canonical leaf-rooted construction of unrooted binary phylogenies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .parsimony import TreeNode


Action = tuple[int, int]


@dataclass(frozen=True)
class Subtree:
    """An immutable partial tree with canonical child ordering."""

    leaves: tuple[str, ...]
    left: "Subtree | None" = None
    right: "Subtree | None" = None

    @classmethod
    def leaf(cls, name: str) -> "Subtree":
        if not name:
            raise ValueError("Leaf names must not be empty")
        return cls((name,))

    @classmethod
    def merge(cls, first: "Subtree", second: "Subtree") -> "Subtree":
        if set(first.leaves) & set(second.leaves):
            raise ValueError("Cannot merge subtrees with overlapping leaves")
        left, right = sorted((first, second), key=lambda subtree: subtree.leaves)
        return cls(tuple(sorted(left.leaves + right.leaves)), left, right)

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def to_tree_node(self) -> TreeNode:
        if self.is_leaf:
            return TreeNode(name=self.leaves[0])
        assert self.left is not None and self.right is not None
        return TreeNode(children=[self.left.to_tree_node(), self.right.to_tree_node()])

    def to_newick(self, terminate: bool = True) -> str:
        def quote(name: str) -> str:
            if any(character in name for character in "(),:;[]'\t\r\n "):
                return "'" + name.replace("'", "''") + "'"
            return name

        if self.is_leaf:
            text = quote(self.leaves[0])
        else:
            assert self.left is not None and self.right is not None
            text = f"({self.left.to_newick(False)},{self.right.to_newick(False)})"
        return text + (";" if terminate else "")


@dataclass(frozen=True)
class TreeState:
    """A forest rooted canonically at the lexicographically first leaf.

    Rooting every unrooted topology on the pendant edge of the same anchor leaf
    gives a one-to-one representation without changing an unrooted tree score.
    The anchor is never merged; all other leaves are joined into the opposite
    side of that root edge.
    """

    anchor: Subtree
    forest: tuple[Subtree, ...]

    @classmethod
    def initial(cls, leaf_names: list[str] | tuple[str, ...]) -> "TreeState":
        if len(leaf_names) < 3:
            raise ValueError("Phylogenetic tree construction requires at least three leaves")
        if len(set(leaf_names)) != len(leaf_names):
            raise ValueError("Leaf names must be unique")
        ordered = sorted(leaf_names)
        return cls(Subtree.leaf(ordered[0]), tuple(Subtree.leaf(name) for name in ordered[1:]))

    @property
    def is_terminal(self) -> bool:
        return len(self.forest) == 1

    def valid_actions(self) -> tuple[Action, ...]:
        if self.is_terminal:
            return ()
        return tuple(combinations(range(len(self.forest)), 2))

    def step(self, action: Action) -> "TreeState":
        if self.is_terminal:
            raise ValueError("Cannot step a terminal tree state")
        i, j = action
        if not (0 <= i < j < len(self.forest)):
            raise ValueError(f"Invalid merge action {action} for {len(self.forest)} subtrees")
        merged = Subtree.merge(self.forest[i], self.forest[j])
        remaining = [subtree for index, subtree in enumerate(self.forest) if index not in action]
        remaining.append(merged)
        return TreeState(self.anchor, tuple(sorted(remaining, key=lambda subtree: subtree.leaves)))

    def step_with_reverse(self, action: Action) -> tuple["TreeState", int]:
        """Apply a merge and return the child state plus its reverse split index."""
        child = self.step(action)
        merged_leaves = tuple(sorted(self.forest[action[0]].leaves + self.forest[action[1]].leaves))
        reverse_index = next(
            index for index, subtree in enumerate(child.forest) if subtree.leaves == merged_leaves
        )
        return child, reverse_index

    def valid_backward_actions(self) -> tuple[int, ...]:
        """Indices of top-level internal subtrees that can be split."""
        return tuple(index for index, subtree in enumerate(self.forest) if not subtree.is_leaf)

    def backward_step(self, subtree_index: int) -> "TreeState":
        if subtree_index not in self.valid_backward_actions():
            raise ValueError(f"Invalid backward action {subtree_index}")
        subtree = self.forest[subtree_index]
        assert subtree.left is not None and subtree.right is not None
        remaining = [
            candidate for index, candidate in enumerate(self.forest) if index != subtree_index
        ]
        remaining.extend((subtree.left, subtree.right))
        return TreeState(
            self.anchor,
            tuple(sorted(remaining, key=lambda candidate: candidate.leaves)),
        )

    def terminal_tree(self) -> TreeNode:
        if not self.is_terminal:
            raise ValueError("State is not terminal")
        return TreeNode(children=[self.anchor.to_tree_node(), self.forest[0].to_tree_node()])

    def terminal_newick(self) -> str:
        if not self.is_terminal:
            raise ValueError("State is not terminal")
        anchor = self.anchor.to_newick(False)
        remainder = self.forest[0].to_newick(False)
        return f"({anchor},{remainder});"
