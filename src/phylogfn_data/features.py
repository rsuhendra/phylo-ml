"""Load aligned residue-level conditioning features from ESM-2 output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fasta import GAP_CHARS, STANDARD_AMINO_ACIDS


AMINO_ACIDS = tuple(sorted(STANDARD_AMINO_ACIDS))
AMINO_ACID_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
GAP_INDEX = len(AMINO_ACIDS)
UNKNOWN_INDEX = GAP_INDEX + 1
AMINO_ACID_VOCAB_SIZE = UNKNOWN_INDEX + 1


@dataclass(frozen=True)
class AlignedFamilyFeatures:
    """One family represented on a shared MSA coordinate system."""

    family_id: str
    identifiers: tuple[str, ...]
    aligned_sequences: tuple[str, ...]
    residue_embeddings: np.ndarray  # [sequences, columns, embedding_dim]
    residue_mask: np.ndarray  # [sequences, columns]
    amino_acid_indices: np.ndarray  # [sequences, columns]

    @property
    def num_sequences(self) -> int:
        return len(self.identifiers)

    @property
    def alignment_length(self) -> int:
        return int(self.residue_mask.shape[1])

    @property
    def embedding_dim(self) -> int:
        return int(self.residue_embeddings.shape[2])


def _amino_acid_index(character: str) -> int:
    if character in GAP_CHARS:
        return GAP_INDEX
    return AMINO_ACID_INDEX.get(character, UNKNOWN_INDEX)


def load_aligned_esm2(family_dir: Path) -> AlignedFamilyFeatures:
    """Scatter ungapped ESM-2 residues back onto their MSA columns."""
    metadata_path = family_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records", [])
    if not records:
        raise ValueError(f"No sequence records in {metadata_path}")

    embeddings = np.load(family_dir / "embeddings.npy", mmap_mode="r")
    aligned_sequences = tuple(str(record["aligned_sequence"]).upper() for record in records)
    alignment_lengths = {len(sequence) for sequence in aligned_sequences}
    if len(alignment_lengths) != 1:
        raise ValueError(f"Inconsistent aligned sequence lengths in {metadata_path}")
    alignment_length = alignment_lengths.pop()
    embedding_dim = int(embeddings.shape[1])

    aligned_embeddings = np.zeros(
        (len(records), alignment_length, embedding_dim), dtype=np.float32
    )
    residue_mask = np.zeros((len(records), alignment_length), dtype=np.bool_)
    amino_acid_indices = np.empty((len(records), alignment_length), dtype=np.int64)
    identifiers: list[str] = []

    for sequence_index, (record, aligned_sequence) in enumerate(zip(records, aligned_sequences)):
        identifier = str(record["id"])
        start = int(record["embedding_start"])
        stop = int(record["embedding_stop"])
        ungapped_to_aligned = np.asarray(record["ungapped_to_aligned"], dtype=np.int64)
        if not 0 <= start < stop <= len(embeddings):
            raise ValueError(f"Invalid embedding slice [{start}:{stop}] for {identifier!r}")
        if stop - start != len(ungapped_to_aligned):
            raise ValueError(f"Embedding/map length mismatch for {identifier!r}")
        if len(ungapped_to_aligned) and (
            ungapped_to_aligned.min() < 0 or ungapped_to_aligned.max() >= alignment_length
        ):
            raise ValueError(f"Out-of-range MSA mapping for {identifier!r}")

        aligned_embeddings[sequence_index, ungapped_to_aligned] = np.asarray(
            embeddings[start:stop], dtype=np.float32
        )
        residue_mask[sequence_index, ungapped_to_aligned] = True
        amino_acid_indices[sequence_index] = np.fromiter(
            (_amino_acid_index(character) for character in aligned_sequence),
            dtype=np.int64,
            count=alignment_length,
        )
        expected_mask = np.fromiter(
            (character not in GAP_CHARS for character in aligned_sequence),
            dtype=np.bool_,
            count=alignment_length,
        )
        if not np.array_equal(residue_mask[sequence_index], expected_mask):
            raise ValueError(f"MSA mapping disagrees with aligned sequence for {identifier!r}")
        identifiers.append(identifier)

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Embedding metadata contains duplicate sequence identifiers")
    return AlignedFamilyFeatures(
        family_id=str(metadata.get("family_id", family_dir.name)),
        identifiers=tuple(identifiers),
        aligned_sequences=aligned_sequences,
        residue_embeddings=aligned_embeddings,
        residue_mask=residue_mask,
        amino_acid_indices=amino_acid_indices,
    )


def load_pooled_esm2(family_dir: Path, pooling: str = "mean") -> tuple[list[str], np.ndarray]:
    """Mean-pooled baseline; the main model should use :func:`load_aligned_esm2`."""
    if pooling != "mean":
        raise ValueError("Only mean residue pooling is currently supported")
    metadata = json.loads((family_dir / "metadata.json").read_text(encoding="utf-8"))
    embeddings = np.load(family_dir / "embeddings.npy", mmap_mode="r")
    identifiers: list[str] = []
    pooled: list[np.ndarray] = []
    for record in metadata["records"]:
        start = int(record["embedding_start"])
        stop = int(record["embedding_stop"])
        if not 0 <= start < stop <= len(embeddings):
            raise ValueError(f"Invalid embedding slice [{start}:{stop}] for {record['id']!r}")
        identifiers.append(record["id"])
        pooled.append(np.asarray(embeddings[start:stop], dtype=np.float32).mean(axis=0))
    if not pooled:
        raise ValueError(f"No sequence records in {family_dir / 'metadata.json'}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Embedding metadata contains duplicate sequence identifiers")
    return identifiers, np.stack(pooled)
