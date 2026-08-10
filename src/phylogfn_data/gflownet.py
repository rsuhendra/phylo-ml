"""Conditional Trajectory-Balance GFlowNet for protein gene-tree topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .adapter import ResidueAwareAdapter
from .forward_policy import ForwardPolicy
from .parsimony import parsimony_log_reward
from .tree_env import TreeState


@dataclass(frozen=True)
class TrajectoryResult:
    terminal_state: TreeState
    log_forward: Tensor
    log_backward: Tensor
    log_reward: float
    loss: Tensor
    num_actions: int


class ConditionalPhyloGFN(nn.Module):
    """Residue-aware family encoder plus a conditional tree-construction GFlowNet."""

    def __init__(
        self,
        esm_dim: int,
        *,
        adapter_dim: int = 128,
        policy_dim: int = 256,
        num_heads: int = 4,
        sequence_layers: int = 1,
        dropout: float = 0.1,
        max_alignment_length: int = 2048,
    ) -> None:
        super().__init__()
        self.config = {
            "esm_dim": esm_dim,
            "adapter_dim": adapter_dim,
            "policy_dim": policy_dim,
            "num_heads": num_heads,
            "sequence_layers": sequence_layers,
            "dropout": dropout,
            "max_alignment_length": max_alignment_length,
        }
        self.adapter = ResidueAwareAdapter(
            esm_dim,
            adapter_dim,
            num_heads=num_heads,
            sequence_layers=sequence_layers,
            dropout=dropout,
            max_alignment_length=max_alignment_length,
        )
        self.forward_policy = ForwardPolicy(adapter_dim, policy_dim)
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
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        leaf_matrix, family_context, site_weights = self.adapter(
            residue_embeddings, residue_mask, amino_acid_indices
        )
        if len(identifiers) != leaf_matrix.shape[0]:
            raise ValueError("Identifier count does not match adapter output")
        leaf_features = {
            identifier: leaf_matrix[index] for index, identifier in enumerate(identifiers)
        }
        log_z = self.log_z_head(family_context).squeeze(-1)
        return leaf_features, log_z, site_weights

    def sample_trajectory(
        self,
        identifiers: tuple[str, ...] | list[str],
        leaf_features: dict[str, Tensor],
        log_z: Tensor,
        sequences: dict[str, str],
        *,
        beta: float = 10.0,
        reward: str = "parsimony",
        temperature: float = 1.0,
        generator: Any = None,
    ) -> TrajectoryResult:
        state = TreeState.initial(tuple(identifiers))
        zero = log_z.new_zeros(())
        log_forward = zero
        log_backward = zero
        num_actions = 0

        while not state.is_terminal:
            action, log_probability = self.forward_policy.sample_action(
                state,
                leaf_features,
                temperature=temperature,
                generator=generator,
            )
            child, reverse_index = state.step_with_reverse(action)
            backward_actions = child.valid_backward_actions()
            if reverse_index not in backward_actions:
                raise RuntimeError("Forward merge did not produce a valid reverse action")
            log_forward = log_forward + log_probability
            log_backward = log_backward - math.log(len(backward_actions))
            state = child
            num_actions += 1

        if reward == "parsimony":
            log_reward = parsimony_log_reward(
                state.terminal_tree(), sequences, beta=beta, normalized=True
            )
        elif reward == "poisson":
            from .likelihood import normalized_poisson_log_reward

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
