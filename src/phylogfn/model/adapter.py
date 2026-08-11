"""Aligned ESM-2 adaptation and pairwise sequence evidence."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.features import AMINO_ACID_VOCAB_SIZE


class ResiduePairAdapter(nn.Module):
    """Compare aligned residues before pooling them into leaf-pair features.

    There is deliberately no absolute MSA-position embedding and no
    within-sequence Transformer. ESM-2 has already contextualized each protein;
    this adapter supplies the missing cross-sequence comparison at homologous
    alignment columns.
    """

    def __init__(
        self,
        esm_dim: int,
        hidden_dim: int = 128,
        pair_dim: int = 64,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        pair_chunk_size: int = 256,
    ) -> None:
        """Configure cross-taxon attention and masked residue-pair encoding."""

        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if pair_dim < 1 or pair_chunk_size < 1:
            raise ValueError("pair_dim and pair_chunk_size must be positive")
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim
        self.pair_dim = pair_dim
        self.pair_chunk_size = pair_chunk_size

        self.esm_projection = nn.Linear(esm_dim, hidden_dim)
        self.amino_acid_embedding = nn.Embedding(AMINO_ACID_VOCAB_SIZE, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.cross_sequence_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_sequence_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pair_site_encoder = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pair_dim),
            nn.LayerNorm(pair_dim),
        )

    def _residue_features(
        self,
        residue_embeddings: Tensor,
        residue_mask: Tensor,
        amino_acid_indices: Tensor,
    ) -> Tensor:
        """Adapt ESM residues by attending across taxa independently per MSA site."""

        if residue_embeddings.ndim != 3:
            raise ValueError("residue_embeddings must have shape [sequences, columns, features]")
        num_sequences, alignment_length, esm_dim = residue_embeddings.shape
        if esm_dim != self.esm_dim:
            raise ValueError(f"Expected ESM dimension {self.esm_dim}, received {esm_dim}")
        if residue_mask.shape != (num_sequences, alignment_length):
            raise ValueError("residue_mask shape does not match residue_embeddings")
        if amino_acid_indices.shape != (num_sequences, alignment_length):
            raise ValueError("amino_acid_indices shape does not match residue_embeddings")

        mask = residue_mask.bool()
        hidden = self.esm_projection(residue_embeddings)
        hidden = hidden + self.amino_acid_embedding(amino_acid_indices)
        hidden = self.input_norm(hidden)

        # MSA columns are independent batches; taxa are the attention tokens.
        by_site = hidden.transpose(0, 1)
        site_padding = ~mask.transpose(0, 1)
        all_gap_sites = site_padding.all(dim=1)
        if all_gap_sites.any():
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
        hidden = self.output_norm(hidden) * mask.unsqueeze(-1)
        return hidden

    def _pair_features(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Pool symmetric site comparisons into a dense leaf-pair evidence matrix."""

        num_sequences, _, _ = hidden.shape
        rows, columns = torch.triu_indices(
            num_sequences, num_sequences, offset=1, device=hidden.device
        )
        chunks: list[Tensor] = []
        for start in range(0, rows.numel(), self.pair_chunk_size):
            chunk_rows = rows[start : start + self.pair_chunk_size]
            chunk_columns = columns[start : start + self.pair_chunk_size]
            left = hidden[chunk_rows]
            right = hidden[chunk_columns]
            valid = mask[chunk_rows] & mask[chunk_columns]
            pair_sites = torch.cat(
                (left + right, torch.abs(left - right), left * right), dim=-1
            )
            encoded = self.pair_site_encoder(pair_sites)
            valid_float = valid.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * valid_float).sum(dim=1)
            pooled = pooled / valid_float.sum(dim=1).clamp_min(1.0)
            chunks.append(pooled)

        pair_values = torch.cat(chunks, dim=0)
        pair_matrix = pair_values.new_zeros((num_sequences, num_sequences, self.pair_dim))
        pair_matrix = pair_matrix.index_put((rows, columns), pair_values)
        return pair_matrix + pair_matrix.transpose(0, 1)

    def forward(
        self,
        residue_embeddings: Tensor,
        residue_mask: Tensor,
        amino_acid_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return pair matrix, family context, and aligned residue features.

        Shapes are ``[N,N,Q]``, ``[H]``, and ``[N,L,H]``. Pair-site pooling
        includes only columns at which both leaves have residues.
        """
        mask = residue_mask.bool()
        hidden = self._residue_features(residue_embeddings, mask, amino_acid_indices)
        pair_matrix = self._pair_features(hidden, mask)
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        family_context = (hidden * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
        return pair_matrix, family_context, hidden
