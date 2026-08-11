"""Conditional Trajectory-Balance GFlowNet for protein gene-tree topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .adapter import ResiduePairAdapter
from .forward_policy import ForwardPolicy
from ..phylo.tree_env import TreeState


@dataclass(frozen=True)
class TrajectoryResult:
    """Terminal state and differentiable statistics for one sampled trajectory."""

    terminal_state: TreeState
    log_forward: Tensor
    log_backward: Tensor
    log_reward: float
    loss: Tensor
    num_actions: int


class ConditionalPhyloGFN(nn.Module):
    """Residue-aware encoder and conditional tree-construction GFlowNet.

    ESM-derived pair evidence conditions the forward policy. The terminal
    phylogenetic reward is non-differentiable; Trajectory Balance propagates its
    signal through sampled forward log-probabilities and the learned ``log Z``.
    """

    def __init__(
        self,
        esm_dim: int,
        *,
        adapter_dim: int = 128,
        pair_dim: int = 64,
        policy_dim: int = 256,
        fitch_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Assemble pair adapter, merge policy, and conditional partition head."""

        super().__init__()
        self.config = {
            "esm_dim": esm_dim,
            "adapter_dim": adapter_dim,
            "pair_dim": pair_dim,
            "policy_dim": policy_dim,
            "fitch_dim": fitch_dim,
            "num_heads": num_heads,
            "dropout": dropout,
        }
        self.adapter = ResiduePairAdapter(
            esm_dim,
            adapter_dim,
            pair_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.forward_policy = ForwardPolicy(pair_dim, policy_dim, fitch_dim)
        self.log_z_head = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, 1),
        )

    def encode_family(
        self,
        identifiers: tuple[str, ...] | list[str],
        residue_embeddings: Tensor,
        residue_mask: Tensor,
        amino_acid_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert aligned ESM residues into pair evidence and family ``log Z``."""

        pair_matrix, family_context, _ = self.adapter(
            residue_embeddings, residue_mask, amino_acid_indices
        )
        if len(identifiers) != pair_matrix.shape[0]:
            raise ValueError("Identifier count does not match adapter output")
        log_z = self.log_z_head(family_context).squeeze(-1)
        return pair_matrix, log_z

    def sample_trajectory(
        self,
        identifiers: tuple[str, ...] | list[str],
        pair_matrix: Tensor,
        log_z: Tensor,
        sequences: dict[str, str],
        *,
        beta: float = 10.0,
        reward: str = "parsimony",
        temperature: float = 1.0,
        generator: Any = None,
    ) -> TrajectoryResult:
        """Construct one topology and evaluate its Trajectory Balance residual."""

        if beta <= 0:
            raise ValueError("beta must be positive")
        # The Python environment carries topology only. Tensorized Fitch state
        # and accumulated parsimony live in the policy cache, avoiding the same
        # per-site merge calculation on both CPU and accelerator.
        state = TreeState.initial(tuple(identifiers))
        policy_cache = self.forward_policy.initialize_state_cache(
            state, pair_matrix, identifiers, sequences=sequences
        )
        zero = log_z.new_zeros(())
        log_forward = zero
        log_backward = zero
        num_actions = 0

        while not state.is_terminal:
            action, log_probability = self.forward_policy.sample_action(
                state,
                pair_matrix,
                identifiers,
                temperature=temperature,
                generator=generator,
                cache=policy_cache,
            )
            child, reverse_index = state.step_with_reverse(action)
            backward_actions = child.valid_backward_actions()
            if reverse_index not in backward_actions:
                raise RuntimeError("Forward merge did not produce a valid reverse action")
            log_forward = log_forward + log_probability
            log_backward = log_backward - math.log(len(backward_actions))
            policy_cache = self.forward_policy.advance_state_cache(
                policy_cache, action, child
            )
            state = child
            num_actions += 1

        if reward == "parsimony":
            raw_score = self.forward_policy.terminal_fitch_score(policy_cache)
            observations = len(next(iter(sequences.values()))) * max(1, len(sequences) - 1)
            log_reward = -beta * raw_score / observations
        elif reward == "poisson":
            from ..phylo.likelihood import normalized_poisson_log_reward

            log_reward = normalized_poisson_log_reward(
                state.terminal_tree(), sequences, beta=beta
            )
        else:
            raise ValueError(f"Unknown reward {reward!r}")
        target = log_z.new_tensor(log_reward)
        residual = log_z + log_forward - log_backward - target
        return TrajectoryResult(
            terminal_state=state,
            log_forward=log_forward,
            log_backward=log_backward,
            log_reward=log_reward,
            loss=residual.square(),
            num_actions=num_actions,
        )
