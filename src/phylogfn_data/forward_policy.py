"""Permutation-invariant forward merge policy for tree construction."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .tree_env import Action, Subtree, TreeState


class ForwardPolicy(nn.Module):
    """Score every valid unordered subtree merge from pooled leaf features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.leaf_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.merge_encoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _symmetric_pair(first: Tensor, second: Tensor) -> Tensor:
        return torch.cat((first + second, torch.abs(first - second), first * second), dim=-1)

    def _encode_subtree(self, subtree: Subtree, leaf_features: dict[str, Tensor]) -> Tensor:
        if subtree.is_leaf:
            try:
                feature = leaf_features[subtree.leaves[0]]
            except KeyError as error:
                raise ValueError(f"Missing feature for leaf {subtree.leaves[0]!r}") from error
            if feature.ndim != 1 or feature.shape[0] != self.input_dim:
                raise ValueError(
                    f"Leaf feature must have shape ({self.input_dim},), got {tuple(feature.shape)}"
                )
            return self.leaf_encoder(feature)
        assert subtree.left is not None and subtree.right is not None
        left = self._encode_subtree(subtree.left, leaf_features)
        right = self._encode_subtree(subtree.right, leaf_features)
        return self.merge_encoder(self._symmetric_pair(left, right))

    def logits(self, state: TreeState, leaf_features: dict[str, Tensor]) -> tuple[Tensor, tuple[Action, ...]]:
        actions = state.valid_actions()
        if not actions:
            raise ValueError("A terminal state has no forward actions")
        representations = torch.stack(
            [self._encode_subtree(subtree, leaf_features) for subtree in state.forest]
        )
        anchor_representation = self._encode_subtree(state.anchor, leaf_features)
        context = torch.cat((representations, anchor_representation.unsqueeze(0))).mean(dim=0)
        action_features = []
        for i, j in actions:
            pair = self._symmetric_pair(representations[i], representations[j])
            action_features.append(torch.cat((pair, context), dim=-1))
        return self.action_head(torch.stack(action_features)).squeeze(-1), actions

    def distribution(
        self, state: TreeState, leaf_features: dict[str, Tensor], temperature: float = 1.0
    ) -> tuple[torch.distributions.Categorical, tuple[Action, ...]]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits, actions = self.logits(state, leaf_features)
        return torch.distributions.Categorical(logits=logits / temperature), actions

    def sample_action(
        self,
        state: TreeState,
        leaf_features: dict[str, Tensor],
        *,
        temperature: float = 1.0,
        generator: Any = None,
    ) -> tuple[Action, Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits, actions = self.logits(state, leaf_features)
        probabilities = torch.softmax(logits / temperature, dim=0)
        index = torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
        return actions[int(index.item())], torch.log(probabilities[index])
