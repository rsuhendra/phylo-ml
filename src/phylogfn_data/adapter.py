"""Residue-aware adaptation of aligned protein-language-model embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import AMINO_ACID_VOCAB_SIZE


class ResidueAwareAdapter(nn.Module):
    """Create leaf features only after cross-sequence, site-aware adaptation."""

    def __init__(
        self,
        esm_dim: int,
        hidden_dim: int = 128,
        *,
        num_heads: int = 4,
        sequence_layers: int = 1,
        dropout: float = 0.1,
        max_alignment_length: int = 2048,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.max_alignment_length = max_alignment_length
        self.esm_projection = nn.Linear(esm_dim, hidden_dim)
        self.amino_acid_embedding = nn.Embedding(AMINO_ACID_VOCAB_SIZE, hidden_dim)
        self.position_embedding = nn.Embedding(max_alignment_length, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.cross_sequence_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_sequence_norm = nn.LayerNorm(hidden_dim)
        sequence_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            sequence_layer, num_layers=sequence_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.site_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        residue_embeddings: Tensor,
        residue_mask: Tensor,
        amino_acid_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return leaf features, family context, and learned MSA-site weights.

        Inputs have shapes ``[N, L, D]``, ``[N, L]``, and ``[N, L]``.
        """
        if residue_embeddings.ndim != 3:
            raise ValueError("residue_embeddings must have shape [sequences, columns, features]")
        num_sequences, alignment_length, esm_dim = residue_embeddings.shape
        if esm_dim != self.esm_dim:
            raise ValueError(f"Expected ESM dimension {self.esm_dim}, received {esm_dim}")
        if residue_mask.shape != (num_sequences, alignment_length):
            raise ValueError("residue_mask shape does not match residue_embeddings")
        if amino_acid_indices.shape != (num_sequences, alignment_length):
            raise ValueError("amino_acid_indices shape does not match residue_embeddings")
        if alignment_length > self.max_alignment_length:
            raise ValueError(
                f"Alignment has {alignment_length} columns; adapter limit is "
                f"{self.max_alignment_length}"
            )

        mask = residue_mask.bool()
        positions = torch.arange(alignment_length, device=residue_embeddings.device)
        hidden = self.esm_projection(residue_embeddings)
        hidden = hidden + self.amino_acid_embedding(amino_acid_indices)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.input_norm(hidden)

        # Treat MSA columns as a batch and attend across sequences/taxa per site.
        by_site = hidden.transpose(0, 1)
        site_padding = ~mask.transpose(0, 1)
        all_gap_sites = site_padding.all(dim=1)
        if all_gap_sites.any():
            # MultiheadAttention cannot softmax a row whose every key is masked.
            site_padding = site_padding.clone()
            site_padding[all_gap_sites, 0] = False
        attended, _ = self.cross_sequence_attention(
            by_site,
            by_site,
            by_site,
            key_padding_mask=site_padding,
            need_weights=False,
        )
        hidden = self.cross_sequence_norm(hidden + attended.transpose(0, 1))

        # Then model ordered context along each protein's aligned sequence.
        hidden = self.sequence_encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)
        hidden = hidden * mask.unsqueeze(-1)

        site_counts = mask.sum(dim=0).clamp_min(1).unsqueeze(-1)
        site_context = hidden.sum(dim=0) / site_counts
        site_logits = self.site_scorer(site_context).squeeze(-1)
        valid_sites = mask.any(dim=0)
        site_logits = site_logits.masked_fill(~valid_sites, torch.finfo(site_logits.dtype).min)
        site_weights = torch.softmax(site_logits, dim=0)

        per_leaf_weights = site_weights.unsqueeze(0) * mask
        per_leaf_weights = per_leaf_weights / per_leaf_weights.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        leaf_features = torch.einsum("nl,nlh->nh", per_leaf_weights, hidden)
        family_context = leaf_features.mean(dim=0)
        return leaf_features, family_context, site_weights
