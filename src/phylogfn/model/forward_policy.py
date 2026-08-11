"""Vectorized forward policy with incremental ESM and Fitch subtree state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..phylo.parsimony import AMINO_ACIDS
from ..phylo.tree_env import ALL_STATE_BITS, Action, FitchFeature, TreeState


@dataclass
class PolicyStateCache:
    """Tensorized sufficient statistics carried between tree-construction steps.

    ``pair_sums[a,b]`` is the sum of leaf-pair ESM evidence between partial
    subtrees ``a`` and ``b``. Fitch states and validity masks have shape
    ``[subtrees, alignment columns]``. All rows follow ``TreeState.forest``.

    A finite off-diagonal entry in ``action_logits`` is reusable while both
    corresponding subtrees survive: every policy feature depends on those two
    subtrees and their fixed leaf-set complement, not on how the complement is
    partitioned into other forest components. NaNs mark pairs involving the
    newly created subtree that still need to be scored.
    """

    subtree_leaves: tuple[tuple[str, ...], ...]
    total_leaves: int
    sizes: Tensor
    pair_sums: Tensor
    anchor_sums: Tensor
    fitch_states: Tensor
    fitch_valid: Tensor
    fitch_scores: Tensor
    anchor_fitch_states: Tensor
    anchor_fitch_valid: Tensor
    anchor_fitch_score: Tensor
    action_logits: Tensor | None = None
    action_logits_complete: bool = False

    @property
    def num_subtrees(self) -> int:
        """Number of currently mergeable components in the forest."""

        return len(self.subtree_leaves)


class ForwardPolicy(nn.Module):
    """Assign logits to subtree merges from pairwise ESM and exact Fitch state."""

    def __init__(
        self,
        pair_dim: int = 64,
        hidden_dim: int = 256,
        fitch_dim: int = 64,
        *,
        fitch_chunk_size: int = 128,
    ) -> None:
        """Configure Fitch-site and candidate-action scoring networks."""

        super().__init__()
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.fitch_dim = fitch_dim
        self.fitch_chunk_size = fitch_chunk_size
        self.fitch_site_encoder = nn.Sequential(
            nn.Linear(3 * len(AMINO_ACIDS), fitch_dim),
            nn.GELU(),
            nn.Linear(fitch_dim, fitch_dim),
            nn.LayerNorm(fitch_dim),
        )
        # Six ESM-pair summaries, one Fitch summary, one immediate cost,
        # and three symmetric subtree-size values.
        action_dim = 6 * pair_dim + fitch_dim + 4
        self.action_head = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initialize_state_cache(
        self,
        state: TreeState,
        pair_matrix: Tensor,
        identifiers: tuple[str, ...] | list[str],
        sequences: dict[str, str] | None = None,
    ) -> PolicyStateCache:
        """Build tensorized subtree statistics once at trajectory initialization."""
        if pair_matrix.ndim != 3 or pair_matrix.shape[0] != pair_matrix.shape[1]:
            raise ValueError("pair_matrix must have shape [sequences, sequences, features]")
        if pair_matrix.shape[0] != len(identifiers) or pair_matrix.shape[2] != self.pair_dim:
            raise ValueError("pair_matrix shape does not match identifiers or policy pair_dim")
        index = {identifier: position for position, identifier in enumerate(identifiers)}
        if len(index) != len(identifiers):
            raise ValueError("Identifiers must be unique")
        state_leaves = set(state.anchor.leaves)
        for subtree in state.forest:
            state_leaves.update(subtree.leaves)
        if state_leaves != set(index):
            raise ValueError("Tree state and pair-feature identifiers do not match")
        if sequences is not None:
            if set(sequences) != set(identifiers):
                raise ValueError("Sequence identifiers must exactly match policy identifiers")
            if len({len(sequence) for sequence in sequences.values()}) != 1:
                raise ValueError("Aligned sequences have inconsistent lengths")

        def fitch_feature(subtree) -> FitchFeature:
            """Resolve stored Fitch state or construct it for an initial leaf."""

            if subtree.fitch is not None:
                return subtree.fitch
            if sequences is not None and subtree.is_leaf:
                return FitchFeature.from_sequence(sequences[subtree.leaves[0]])
            raise ValueError(
                "Forward policy requires Fitch-annotated subtrees or initial leaf sequences"
            )

        forest_fitch = [fitch_feature(subtree) for subtree in state.forest]
        anchor_fitch = fitch_feature(state.anchor)

        membership = pair_matrix.new_zeros((len(state.forest), len(identifiers)))
        for row, subtree in enumerate(state.forest):
            membership[row, [index[name] for name in subtree.leaves]] = 1
        anchor = pair_matrix.new_zeros((len(identifiers),))
        anchor[[index[name] for name in state.anchor.leaves]] = 1
        pair_sums = torch.einsum("an,nmq,bm->abq", membership, pair_matrix, membership)
        anchor_sums = torch.einsum("n,nmq,am->aq", anchor, pair_matrix, membership)
        fitch_states = torch.tensor(
            [feature.states for feature in forest_fitch],
            device=pair_matrix.device,
            dtype=torch.long,
        )
        fitch_valid = torch.tensor(
            [feature.valid for feature in forest_fitch],
            device=pair_matrix.device,
            dtype=torch.bool,
        )
        return PolicyStateCache(
            subtree_leaves=tuple(subtree.leaves for subtree in state.forest),
            total_leaves=len(identifiers),
            sizes=membership.sum(dim=1),
            pair_sums=pair_sums,
            anchor_sums=anchor_sums,
            fitch_states=fitch_states,
            fitch_valid=fitch_valid,
            fitch_scores=torch.tensor(
                [feature.score for feature in forest_fitch],
                device=pair_matrix.device,
                dtype=torch.long,
            ),
            anchor_fitch_states=torch.tensor(
                anchor_fitch.states, device=pair_matrix.device, dtype=torch.long
            ),
            anchor_fitch_valid=torch.tensor(
                anchor_fitch.valid, device=pair_matrix.device, dtype=torch.bool
            ),
            anchor_fitch_score=torch.tensor(
                anchor_fitch.score, device=pair_matrix.device, dtype=torch.long
            ),
        )

    @staticmethod
    def _merge_fitch_rows(
        cache: PolicyStateCache, action: Action
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the exact Fitch recurrence to one selected subtree pair."""

        left_index, right_index = action
        left_states = cache.fitch_states[left_index]
        right_states = cache.fitch_states[right_index]
        left_valid = cache.fitch_valid[left_index]
        right_valid = cache.fitch_valid[right_index]
        intersection = left_states & right_states
        both_missing = ~left_valid & ~right_valid
        merged_states = torch.where(
            both_missing,
            torch.full_like(left_states, ALL_STATE_BITS),
            torch.where(
                ~left_valid,
                right_states,
                torch.where(
                    ~right_valid,
                    left_states,
                    torch.where(intersection != 0, intersection, left_states | right_states),
                ),
            ),
        )
        added_cost = (left_valid & right_valid & (intersection == 0)).sum()
        merged_score = cache.fitch_scores[left_index] + cache.fitch_scores[right_index]
        return merged_states, left_valid | right_valid, merged_score + added_cost

    def advance_state_cache(
        self,
        cache: PolicyStateCache,
        action: Action,
        child_state: TreeState,
    ) -> PolicyStateCache:
        """Update subtree statistics from one merge without revisiting leaves."""
        i, j = action
        if not 0 <= i < j < cache.num_subtrees:
            raise ValueError(f"Invalid merge action {action} for policy cache")
        if len(child_state.forest) != cache.num_subtrees - 1:
            raise ValueError("Child tree state does not follow one cache merge")

        merged_leaves = tuple(sorted(cache.subtree_leaves[i] + cache.subtree_leaves[j]))
        survivor_indices = [
            index for index in range(cache.num_subtrees) if index not in action
        ]
        survivor_tensor = torch.tensor(
            survivor_indices, device=cache.pair_sums.device, dtype=torch.long
        )
        survivor_pairs = cache.pair_sums[survivor_tensor][:, survivor_tensor]
        survivor_to_merged = (
            cache.pair_sums[survivor_tensor, i]
            + cache.pair_sums[survivor_tensor, j]
        )
        merged_to_survivor = (
            cache.pair_sums[i, survivor_tensor]
            + cache.pair_sums[j, survivor_tensor]
        )
        merged_within = (
            cache.pair_sums[i, i]
            + cache.pair_sums[i, j]
            + cache.pair_sums[j, i]
            + cache.pair_sums[j, j]
        )
        top = torch.cat((survivor_pairs, survivor_to_merged.unsqueeze(1)), dim=1)
        bottom = torch.cat((merged_to_survivor, merged_within.unsqueeze(0)), dim=0)
        pair_sums = torch.cat((top, bottom.unsqueeze(0)), dim=0)
        sizes = torch.cat(
            (cache.sizes[survivor_tensor], (cache.sizes[i] + cache.sizes[j]).unsqueeze(0))
        )
        anchor_sums = torch.cat(
            (
                cache.anchor_sums[survivor_tensor],
                (cache.anchor_sums[i] + cache.anchor_sums[j]).unsqueeze(0),
            )
        )

        merged_states, merged_valid, merged_score = self._merge_fitch_rows(cache, action)
        fitch_states = torch.cat(
            (cache.fitch_states[survivor_tensor], merged_states.unsqueeze(0))
        )
        fitch_valid = torch.cat(
            (cache.fitch_valid[survivor_tensor], merged_valid.unsqueeze(0))
        )
        fitch_scores = torch.cat(
            (cache.fitch_scores[survivor_tensor], merged_score.unsqueeze(0))
        )

        action_logits = None
        if cache.action_logits is not None:
            # Every survivor-survivor logit is unchanged. Add one unknown row
            # and column for the new subtree; _cached_logits fills only these.
            survivor_logits = cache.action_logits[survivor_tensor][:, survivor_tensor]
            missing_column = survivor_logits.new_full(
                (len(survivor_indices), 1), torch.nan
            )
            missing_row = survivor_logits.new_full(
                (1, len(survivor_indices) + 1), torch.nan
            )
            action_logits = torch.cat(
                (torch.cat((survivor_logits, missing_column), dim=1), missing_row),
                dim=0,
            )

        intermediate_leaves = tuple(
            cache.subtree_leaves[index] for index in survivor_indices
        ) + (merged_leaves,)
        position = {leaves: index for index, leaves in enumerate(intermediate_leaves)}
        try:
            order = [position[subtree.leaves] for subtree in child_state.forest]
        except KeyError as error:
            raise ValueError("Child tree state and policy cache are inconsistent") from error
        order_tensor = torch.tensor(order, device=cache.pair_sums.device, dtype=torch.long)
        pair_sums = pair_sums[order_tensor][:, order_tensor]
        sizes = sizes[order_tensor]
        anchor_sums = anchor_sums[order_tensor]
        fitch_states = fitch_states[order_tensor]
        fitch_valid = fitch_valid[order_tensor]
        fitch_scores = fitch_scores[order_tensor]
        if action_logits is not None:
            action_logits = action_logits[order_tensor][:, order_tensor]

        return PolicyStateCache(
            subtree_leaves=tuple(subtree.leaves for subtree in child_state.forest),
            total_leaves=cache.total_leaves,
            sizes=sizes,
            pair_sums=pair_sums,
            anchor_sums=anchor_sums,
            fitch_states=fitch_states,
            fitch_valid=fitch_valid,
            fitch_scores=fitch_scores,
            anchor_fitch_states=cache.anchor_fitch_states,
            anchor_fitch_valid=cache.anchor_fitch_valid,
            anchor_fitch_score=cache.anchor_fitch_score,
            action_logits=action_logits,
            action_logits_complete=False,
        )

    @staticmethod
    def _action_indices(cache: PolicyStateCache) -> tuple[Tensor, Tensor]:
        """Return vectorized indices for all unordered candidate merges."""

        return torch.triu_indices(
            cache.num_subtrees,
            cache.num_subtrees,
            offset=1,
            device=cache.pair_sums.device,
        )

    def _esm_action_features(
        self,
        cache: PolicyStateCache,
        left_index: Tensor,
        right_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate cross, within, anchor, complement, and size evidence."""

        sizes = cache.sizes
        left_sizes = sizes[left_index]
        right_sizes = sizes[right_index]
        union_sizes = left_sizes + right_sizes

        cross = cache.pair_sums[left_index, right_index]
        cross = cross / (left_sizes * right_sizes).unsqueeze(-1).clamp_min(1.0)

        diagonal = torch.arange(cache.num_subtrees, device=sizes.device)
        within = cache.pair_sums[diagonal, diagonal]
        within = within / (sizes * (sizes - 1)).unsqueeze(-1).clamp_min(1.0)
        left_within = within[left_index]
        right_within = within[right_index]
        within_features = torch.cat(
            (
                left_within + right_within,
                torch.abs(left_within - right_within),
                left_within * right_within,
            ),
            dim=-1,
        )

        anchor_union = (
            cache.anchor_sums[left_index] + cache.anchor_sums[right_index]
        ) / union_sizes.unsqueeze(-1).clamp_min(1.0)

        row_sums = cache.pair_sums.sum(dim=1)
        rest_sums = (
            row_sums[left_index]
            + row_sums[right_index]
            - cache.pair_sums[left_index, left_index]
            - cache.pair_sums[left_index, right_index]
            - cache.pair_sums[right_index, left_index]
            - cache.pair_sums[right_index, right_index]
        )
        rest_sizes = sizes.sum() - union_sizes
        rest_means = rest_sums / (union_sizes * rest_sizes).unsqueeze(-1).clamp_min(1.0)

        total_leaves = float(cache.total_leaves)
        size_features = torch.stack(
            (
                union_sizes / total_leaves,
                torch.abs(left_sizes - right_sizes) / total_leaves,
                (left_sizes * right_sizes) / (total_leaves * total_leaves),
            ),
            dim=-1,
        )
        return torch.cat(
            (cross, within_features, anchor_union, rest_means), dim=-1
        ), size_features

    def _fitch_action_features(
        self,
        cache: PolicyStateCache,
        left_index: Tensor,
        right_index: Tensor,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode ancestral-state compatibility and immediate mutation cost."""

        bits = 1 << torch.arange(len(AMINO_ACIDS), device=reference.device, dtype=torch.long)
        feature_chunks: list[Tensor] = []
        cost_chunks: list[Tensor] = []
        for start in range(0, left_index.numel(), self.fitch_chunk_size):
            chunk_left = left_index[start : start + self.fitch_chunk_size]
            chunk_right = right_index[start : start + self.fitch_chunk_size]
            left_states = cache.fitch_states[chunk_left]
            right_states = cache.fitch_states[chunk_right]
            left_valid = cache.fitch_valid[chunk_left]
            right_valid = cache.fitch_valid[chunk_right]
            pair_valid = left_valid & right_valid
            left_binary = ((left_states.unsqueeze(-1) & bits) != 0).to(reference.dtype)
            right_binary = ((right_states.unsqueeze(-1) & bits) != 0).to(reference.dtype)
            site_input = torch.cat(
                (
                    left_binary + right_binary,
                    torch.abs(left_binary - right_binary),
                    left_binary * right_binary,
                ),
                dim=-1,
            )
            encoded = self.fitch_site_encoder(site_input)
            valid_float = pair_valid.unsqueeze(-1).to(reference.dtype)
            pooled = (encoded * valid_float).sum(dim=1)
            pooled = pooled / valid_float.sum(dim=1).clamp_min(1.0)
            incremental = pair_valid & ((left_states & right_states) == 0)
            normalized_cost = incremental.sum(dim=1, keepdim=True).to(reference.dtype)
            normalized_cost = normalized_cost / max(1, left_states.shape[1])
            feature_chunks.append(pooled)
            cost_chunks.append(normalized_cost)
        return torch.cat(feature_chunks), torch.cat(cost_chunks)

    def _score_action_indices(
        self,
        cache: PolicyStateCache,
        left_index: Tensor,
        right_index: Tensor,
    ) -> Tensor:
        """Run selected candidate pairs through both feature branches and the head."""

        esm_features, size_features = self._esm_action_features(
            cache, left_index, right_index
        )
        fitch_features, immediate_cost = self._fitch_action_features(
            cache, left_index, right_index, cache.pair_sums
        )
        action_features = torch.cat(
            (esm_features, fitch_features, immediate_cost, size_features), dim=-1
        )
        return self.action_head(action_features).squeeze(-1)

    def _cached_logits(self, cache: PolicyStateCache) -> tuple[Tensor, Tensor, Tensor]:
        """Fill missing candidate logits and return all currently valid scores."""

        if cache.num_subtrees < 2:
            raise ValueError("A terminal state has no forward actions")
        left_index, right_index = self._action_indices(cache)
        if not cache.action_logits_complete:
            if cache.action_logits is None:
                matrix = cache.pair_sums.new_full(
                    (cache.num_subtrees, cache.num_subtrees), torch.nan
                )
                missing_left = left_index
                missing_right = right_index
            else:
                matrix = cache.action_logits
                missing = torch.isnan(matrix[left_index, right_index])
                missing_left = left_index[missing]
                missing_right = right_index[missing]
            # Initially this scores all pairs. After a merge it scores only
            # pairs incident to the new subtree, reducing neural evaluations
            # across one trajectory from cubic to quadratic in leaf count.
            missing_logits = self._score_action_indices(
                cache, missing_left, missing_right
            )
            matrix = matrix.index_put((missing_left, missing_right), missing_logits)
            matrix = matrix.index_put((missing_right, missing_left), missing_logits)
            cache.action_logits = matrix
            cache.action_logits_complete = True
        assert cache.action_logits is not None
        return cache.action_logits[left_index, right_index], left_index, right_index

    @staticmethod
    def terminal_fitch_score(cache: PolicyStateCache) -> int:
        """Return exact terminal parsimony without rebuilding Fitch state on CPU."""
        if cache.num_subtrees != 1:
            raise ValueError("Policy cache is not terminal")
        intersection = cache.anchor_fitch_states & cache.fitch_states[0]
        added_cost = (
            cache.anchor_fitch_valid
            & cache.fitch_valid[0]
            & (intersection == 0)
        ).sum()
        score = cache.anchor_fitch_score + cache.fitch_scores[0] + added_cost
        return int(score.item())

    def logits(
        self,
        state: TreeState,
        pair_matrix: Tensor,
        identifiers: tuple[str, ...] | list[str],
        *,
        cache: PolicyStateCache | None = None,
    ) -> tuple[Tensor, tuple[Action, ...]]:
        """Return logits in the same order as ``TreeState.valid_actions``."""
        if cache is None:
            cache = self.initialize_state_cache(state, pair_matrix, identifiers)
        elif cache.subtree_leaves != tuple(subtree.leaves for subtree in state.forest):
            raise ValueError("Policy cache does not match tree state")
        logits, _, _ = self._cached_logits(cache)
        return logits, state.valid_actions()

    def distribution(
        self,
        state: TreeState,
        pair_matrix: Tensor,
        identifiers: tuple[str, ...] | list[str],
        temperature: float = 1.0,
        *,
        cache: PolicyStateCache | None = None,
    ) -> tuple[torch.distributions.Categorical, tuple[Action, ...]]:
        """Construct the temperature-scaled categorical forward policy."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits, actions = self.logits(
            state, pair_matrix, identifiers, cache=cache
        )
        return torch.distributions.Categorical(logits=logits / temperature), actions

    def sample_action(
        self,
        state: TreeState,
        pair_matrix: Tensor,
        identifiers: tuple[str, ...] | list[str],
        *,
        temperature: float = 1.0,
        generator: Any = None,
        cache: PolicyStateCache | None = None,
    ) -> tuple[Action, Tensor]:
        """Sample one valid merge and return its differentiable log-probability."""

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if cache is None:
            cache = self.initialize_state_cache(state, pair_matrix, identifiers)
        elif cache.subtree_leaves != tuple(subtree.leaves for subtree in state.forest):
            raise ValueError("Policy cache does not match tree state")
        logits, left_index, right_index = self._cached_logits(cache)
        probabilities = torch.softmax(logits / temperature, dim=0)
        selected = torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
        action = (int(left_index[selected].item()), int(right_index[selected].item()))
        return action, torch.log(probabilities[selected])
