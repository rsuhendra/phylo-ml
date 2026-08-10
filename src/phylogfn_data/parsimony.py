"""Score protein phylogenies with unordered Fitch/Sankoff parsimony."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fasta import GAP_CHARS, STANDARD_AMINO_ACIDS, read_fasta


AMINO_ACIDS = tuple(sorted(STANDARD_AMINO_ACIDS))
AMINO_ACID_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
AMBIGUOUS_STATES = {
    "B": frozenset("DN"),
    "Z": frozenset("EQ"),
    "J": frozenset("IL"),
    "X": STANDARD_AMINO_ACIDS,
    "U": frozenset("C"),  # Selenocysteine is treated as cysteine for this baseline.
    "O": frozenset("K"),  # Pyrrolysine is treated as lysine for this baseline.
}


@dataclass
class TreeNode:
    name: str | None = None
    children: list["TreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class _NewickParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> TreeNode:
        node = self._subtree()
        self._space_and_comments()
        if self._peek() == ";":
            self.position += 1
            self._space_and_comments()
        if self.position != len(self.text):
            raise ValueError(f"Unexpected Newick text at position {self.position}")
        return node

    def _subtree(self) -> TreeNode:
        self._space_and_comments()
        children: list[TreeNode] = []
        if self._peek() == "(":
            self.position += 1
            while True:
                children.append(self._subtree())
                self._space_and_comments()
                token = self._peek()
                if token == ",":
                    self.position += 1
                    continue
                if token == ")":
                    self.position += 1
                    break
                raise ValueError(f"Expected ',' or ')' at Newick position {self.position}")
        name = self._label()
        self._space_and_comments()
        if self._peek() == ":":
            self.position += 1
            self._branch_length()
        if not children and not name:
            raise ValueError(f"Missing leaf name at Newick position {self.position}")
        return TreeNode(name=name or None, children=children)

    def _label(self) -> str:
        self._space_and_comments()
        if self._peek() in {"'", '"'}:
            quote = self._peek()
            self.position += 1
            parts: list[str] = []
            while self.position < len(self.text):
                character = self.text[self.position]
                self.position += 1
                if character == quote:
                    if quote == "'" and self._peek() == "'":
                        parts.append("'")
                        self.position += 1
                        continue
                    return "".join(parts)
                parts.append(character)
            raise ValueError("Unterminated quoted Newick label")
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in "(),:;[]\t\r\n ":
            self.position += 1
        return self.text[start : self.position]

    def _branch_length(self) -> None:
        self._space_and_comments()
        start = self.position
        while self.position < len(self.text) and self.text[self.position] not in ",);[\t\r\n ":
            self.position += 1
        if start == self.position:
            raise ValueError(f"Missing branch length at Newick position {self.position}")
        try:
            float(self.text[start : self.position])
        except ValueError as error:
            raise ValueError(f"Invalid branch length {self.text[start:self.position]!r}") from error

    def _space_and_comments(self) -> None:
        while True:
            while self.position < len(self.text) and self.text[self.position].isspace():
                self.position += 1
            if self._peek() != "[":
                return
            end = self.text.find("]", self.position + 1)
            if end < 0:
                raise ValueError("Unterminated Newick comment")
            self.position = end + 1

    def _peek(self) -> str:
        return self.text[self.position] if self.position < len(self.text) else ""


def parse_newick(text: str) -> TreeNode:
    return _NewickParser(text).parse()


def leaf_names(tree: TreeNode) -> list[str]:
    if tree.is_leaf:
        assert tree.name is not None
        return [tree.name]
    names: list[str] = []
    for child in tree.children:
        names.extend(leaf_names(child))
    return names


def tree_to_newick(tree: TreeNode, terminate: bool = True) -> str:
    def quote(name: str) -> str:
        if any(character in name for character in "(),:;[]'\t\r\n "):
            return "'" + name.replace("'", "''") + "'"
        return name

    if tree.is_leaf:
        assert tree.name is not None
        text = quote(tree.name)
    else:
        text = "(" + ",".join(tree_to_newick(child, False) for child in tree.children) + ")"
    return text + (";" if terminate else "")


def _allowed_states(character: str) -> frozenset[str]:
    if character in GAP_CHARS or character == "?":
        return STANDARD_AMINO_ACIDS
    if character in STANDARD_AMINO_ACIDS:
        return frozenset((character,))
    if character in AMBIGUOUS_STATES:
        return AMBIGUOUS_STATES[character]
    raise ValueError(f"Unsupported alignment character {character!r}")


def parsimony_score(tree: TreeNode, sequences: dict[str, str]) -> int:
    """Return exact unordered-state parsimony cost across alignment columns."""
    if not sequences:
        raise ValueError("Alignment is empty")
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ValueError("Alignment sequences have inconsistent lengths")

    names = leaf_names(tree)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Tree contains duplicate leaves: {', '.join(duplicates)}")
    missing = sorted(set(sequences) - set(names))
    extra = sorted(set(names) - set(sequences))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing from tree: {', '.join(missing)}")
        if extra:
            details.append(f"missing from alignment: {', '.join(extra)}")
        raise ValueError("Tree/alignment leaf mismatch (" + "; ".join(details) + ")")

    alignment_length = lengths.pop()
    infinity = np.int32(len(sequences) + 1)

    def costs(node: TreeNode) -> np.ndarray:
        if node.is_leaf:
            assert node.name is not None
            result = np.full((alignment_length, len(AMINO_ACIDS)), infinity, dtype=np.int32)
            for column, character in enumerate(sequences[node.name].upper()):
                for amino_acid in _allowed_states(character):
                    result[column, AMINO_ACID_INDEX[amino_acid]] = 0
            return result

        result = np.zeros((alignment_length, len(AMINO_ACIDS)), dtype=np.int32)
        for child in node.children:
            child_costs = costs(child)
            cheapest_change = child_costs.min(axis=1, keepdims=True) + 1
            result += np.minimum(child_costs, cheapest_change)
        return result

    return int(costs(tree).min(axis=1).sum())


def normalized_parsimony_score(tree: TreeNode, sequences: dict[str, str]) -> float:
    """Scale parsimony to approximately ``[0, 1]`` across family sizes."""
    raw_score = parsimony_score(tree, sequences)
    alignment_length = len(next(iter(sequences.values())))
    maximum_unit_cost = alignment_length * max(1, len(sequences) - 1)
    return raw_score / maximum_unit_cost


def parsimony_log_reward(
    tree: TreeNode,
    sequences: dict[str, str],
    *,
    beta: float = 1.0,
    normalized: bool = True,
) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    score = (
        normalized_parsimony_score(tree, sequences)
        if normalized
        else float(parsimony_score(tree, sequences))
    )
    return -beta * score


def score_files(alignment_path: Path, tree_path: Path) -> int:
    records = read_fasta(alignment_path)
    sequences = {record.identifier: record.sequence for record in records}
    if len(sequences) != len(records):
        raise ValueError("Alignment contains duplicate sequence identifiers")
    tree = parse_newick(tree_path.read_text(encoding="utf-8"))
    return parsimony_score(tree, sequences)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True, help="Aligned protein FASTA")
    parser.add_argument("--tree", type=Path, required=True, help="Candidate tree in Newick format")
    parser.add_argument("--beta", type=float, default=1.0, help="Reward inverse temperature")
    parser.add_argument(
        "--normalization",
        choices=("per-site-leaf", "raw"),
        default="per-site-leaf",
        help="Normalize reward scale across alignment lengths and leaf counts",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.beta <= 0:
        raise SystemExit("--beta must be positive")
    score = score_files(args.alignment, args.tree)
    records = read_fasta(args.alignment)
    sequences = {record.identifier: record.sequence for record in records}
    normalized_score = score / (len(records[0].sequence) * max(1, len(records) - 1))
    reward_score = normalized_score if args.normalization == "per-site-leaf" else float(score)
    result = {
        "parsimony_score": score,
        "normalized_parsimony": normalized_score,
        "normalization": args.normalization,
        "beta": args.beta,
        "log_reward": -args.beta * reward_score,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"parsimony_score={score}")
        print(f"log_reward={result['log_reward']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
