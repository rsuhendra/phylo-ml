"""Canonical leaf-rooted construction with exact partial-tree Fitch states."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..data.fasta import GAP_CHARS
from .parsimony import AMINO_ACID_INDEX, AMINO_ACIDS, TreeNode, allowed_states


Action = tuple[int, int]
ALL_STATE_BITS = (1 << len(AMINO_ACIDS)) - 1


def _character_bits(character: str) -> tuple[int, bool]:
    """Return possible amino-acid bits and whether the observation is present."""
    character = character.upper()
    if character in GAP_CHARS or character == "?":
        return ALL_STATE_BITS, False
    bits = 0
    for amino_acid in allowed_states(character):
        bits |= 1 << AMINO_ACID_INDEX[amino_acid]
    return bits, True


@dataclass(frozen=True)
class FitchFeature:
    """Minimum-cost possible root states at every site of a partial tree."""

    states: tuple[int, ...]
    valid: tuple[bool, ...]
    score: int = 0

    @classmethod
    def from_sequence(cls, sequence: str) -> "FitchFeature":
        """Initialize per-site possible states for one observed leaf sequence."""

        encoded = [_character_bits(character) for character in sequence]
        return cls(
            states=tuple(state for state, _ in encoded),
            valid=tuple(present for _, present in encoded),
        )

    @classmethod
    def merge(cls, left: "FitchFeature", right: "FitchFeature") -> "FitchFeature":
        """Apply the Fitch intersection/union recurrence across all sites."""

        if len(left.states) != len(right.states):
            raise ValueError("Cannot merge Fitch features with different alignment lengths")
        states: list[int] = []
        valid: list[bool] = []
        added_cost = 0
        for left_state, right_state, left_valid, right_valid in zip(
            left.states, right.states, left.valid, right.valid
        ):
            if not left_valid and not right_valid:
                states.append(ALL_STATE_BITS)
                valid.append(False)
                continue
            if not left_valid:
                states.append(right_state)
                valid.append(True)
                continue
            if not right_valid:
                states.append(left_state)
                valid.append(True)
                continue
            intersection = left_state & right_state
            if intersection:
                states.append(intersection)
            else:
                states.append(left_state | right_state)
                added_cost += 1
            valid.append(True)
        return cls(
            states=tuple(states),
            valid=tuple(valid),
            score=left.score + right.score + added_cost,
        )

    def incremental_cost(self, other: "FitchFeature") -> int:
        """Count mutations introduced by merging with another partial tree."""

        if len(self.states) != len(other.states):
            raise ValueError("Fitch features have different alignment lengths")
        return sum(
            1
            for left, right, left_valid, right_valid in zip(
                self.states, other.states, self.valid, other.valid
            )
            if left_valid and right_valid and not (left & right)
        )


@dataclass(frozen=True)
class Subtree:
    """An immutable partial tree with canonical children and a Fitch state."""

    leaves: tuple[str, ...]
    left: "Subtree | None" = None
    right: "Subtree | None" = None
    fitch: FitchFeature | None = None

    @classmethod
    def leaf(cls, name: str, sequence: str | None = None) -> "Subtree":
        """Create a singleton topology, optionally annotated with Fitch state."""

        if not name:
            raise ValueError("Leaf names must not be empty")
        return cls((name,), fitch=FitchFeature.from_sequence(sequence) if sequence else None)

    @classmethod
    def merge(cls, first: "Subtree", second: "Subtree") -> "Subtree":
        """Join disjoint subtrees with canonical child ordering."""

        if set(first.leaves) & set(second.leaves):
            raise ValueError("Cannot merge subtrees with overlapping leaves")
        left, right = sorted((first, second), key=lambda subtree: subtree.leaves)
        if (left.fitch is None) != (right.fitch is None):
            raise ValueError("Cannot merge one Fitch-annotated subtree with one unannotated subtree")
        fitch = (
            FitchFeature.merge(left.fitch, right.fitch)
            if left.fitch is not None and right.fitch is not None
            else None
        )
        return cls(tuple(sorted(left.leaves + right.leaves)), left, right, fitch)

    @property
    def is_leaf(self) -> bool:
        """Return whether this partial subtree is a singleton leaf."""

        return self.left is None

    def to_tree_node(self) -> TreeNode:
        """Convert the immutable construction subtree to a generic tree node."""

        if self.is_leaf:
            return TreeNode(name=self.leaves[0])
        assert self.left is not None and self.right is not None
        return TreeNode(children=[self.left.to_tree_node(), self.right.to_tree_node()])

    def to_newick(self, terminate: bool = True) -> str:
        """Serialize this canonical partial subtree to Newick."""

        def quote(name: str) -> str:
            """Quote labels containing syntax-significant characters."""

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
    """A forest rooted canonically at the lexicographically first leaf."""

    anchor: Subtree
    forest: tuple[Subtree, ...]

    @classmethod
    def initial(
        cls,
        leaf_names: list[str] | tuple[str, ...],
        sequences: dict[str, str] | None = None,
    ) -> "TreeState":
        """Create singleton components with a fixed lexicographic anchor leaf."""

        if len(leaf_names) < 3:
            raise ValueError("Phylogenetic tree construction requires at least three leaves")
        if len(set(leaf_names)) != len(leaf_names):
            raise ValueError("Leaf names must be unique")
        if sequences is not None:
            if set(sequences) != set(leaf_names):
                raise ValueError("Sequence identifiers must exactly match leaf names")
            lengths = {len(sequence) for sequence in sequences.values()}
            if len(lengths) != 1:
                raise ValueError("Aligned sequences have inconsistent lengths")
        ordered = sorted(leaf_names)

        def leaf(name: str) -> Subtree:
            """Construct one initial component with optional sequence state."""

            return Subtree.leaf(name, sequences[name] if sequences is not None else None)

        return cls(leaf(ordered[0]), tuple(leaf(name) for name in ordered[1:]))

    @property
    def is_terminal(self) -> bool:
        """Return whether all non-anchor leaves form one subtree."""

        return len(self.forest) == 1

    def valid_actions(self) -> tuple[Action, ...]:
        """Enumerate every unordered pair of mergeable forest components."""

        if self.is_terminal:
            return ()
        return tuple(combinations(range(len(self.forest)), 2))

    def step(self, action: Action) -> "TreeState":
        """Apply one merge and restore canonical forest ordering."""

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
        """Apply a merge and identify its inverse split in the child state."""

        child = self.step(action)
        merged_leaves = tuple(sorted(self.forest[action[0]].leaves + self.forest[action[1]].leaves))
        reverse_index = next(
            index for index, subtree in enumerate(child.forest) if subtree.leaves == merged_leaves
        )
        return child, reverse_index

    def valid_backward_actions(self) -> tuple[int, ...]:
        """Return top-level internal subtrees that may be split backward."""

        return tuple(index for index, subtree in enumerate(self.forest) if not subtree.is_leaf)

    def backward_step(self, subtree_index: int) -> "TreeState":
        """Undo one top-level merge by restoring its two children."""

        if subtree_index not in self.valid_backward_actions():
            raise ValueError(f"Invalid backward action {subtree_index}")
        subtree = self.forest[subtree_index]
        assert subtree.left is not None and subtree.right is not None
        remaining = [
            candidate for index, candidate in enumerate(self.forest) if index != subtree_index
        ]
        remaining.extend((subtree.left, subtree.right))
        return TreeState(self.anchor, tuple(sorted(remaining, key=lambda item: item.leaves)))

    def terminal_fitch(self) -> FitchFeature:
        """Combine anchor and remainder to obtain the complete-tree Fitch score."""

        if not self.is_terminal:
            raise ValueError("State is not terminal")
        if self.anchor.fitch is None or self.forest[0].fitch is None:
            raise ValueError("State does not carry Fitch features")
        return FitchFeature.merge(self.anchor.fitch, self.forest[0].fitch)

    def terminal_tree(self) -> TreeNode:
        """Return the complete topology rooted at the anchor pendant edge."""

        if not self.is_terminal:
            raise ValueError("State is not terminal")
        return TreeNode(children=[self.anchor.to_tree_node(), self.forest[0].to_tree_node()])

    def terminal_newick(self) -> str:
        """Serialize the complete canonical topology to Newick."""

        if not self.is_terminal:
            raise ValueError("State is not terminal")
        anchor = self.anchor.to_newick(False)
        remainder = self.forest[0].to_newick(False)
        return f"({anchor},{remainder});"
