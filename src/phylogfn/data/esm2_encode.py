"""Encode ungapped MSA records with frozen ESM-2 and retain alignment maps."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .fasta import alignment_maps, discover_fastas, read_fasta, ungap, validate_protein


def choose_device(torch: Any, requested: str) -> str:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU in priority order."""

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Partition a list into consecutive inference batches."""

    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def encode_family(
    fasta_path: Path,
    output_root: Path,
    tokenizer: Any,
    model: Any,
    torch: Any,
    *,
    device: str,
    batch_size: int,
    storage_dtype: str,
    model_name: str,
    overwrite: bool,
) -> None:
    """Encode one aligned family and atomically save residues plus metadata.

    ESM-2 sees ungapped proteins. The output metadata records exact inverse maps
    so residue embeddings can later be scattered back onto the original MSA.
    """

    family_id = fasta_path.stem
    family_dir = output_root / family_id
    embeddings_path = family_dir / "embeddings.npy"
    metadata_path = family_dir / "metadata.json"
    if embeddings_path.exists() and metadata_path.exists() and not overwrite:
        print(f"Skipping existing {family_id}", file=sys.stderr)
        return

    records = read_fasta(fasta_path)
    alignment_lengths = {len(record.sequence) for record in records}
    if len(alignment_lengths) != 1:
        raise ValueError(f"Input is not an MSA (inconsistent lengths): {fasta_path}")

    prepared: list[dict[str, Any]] = []
    for record in records:
        sequence = ungap(record.sequence)
        if not validate_protein(sequence):
            raise ValueError(f"Invalid protein sequence {record.identifier!r} in {fasta_path}")
        if len(sequence) > 1022:
            raise ValueError(
                f"{record.identifier!r} has {len(sequence)} residues; ESM-2 supports at most 1022 here"
            )
        aligned_to_ungapped, ungapped_to_aligned = alignment_maps(record.sequence)
        prepared.append(
            {
                "id": record.identifier,
                "description": record.description,
                "sequence": sequence,
                "aligned_sequence": record.sequence,
                "aligned_to_ungapped": aligned_to_ungapped,
                "ungapped_to_aligned": ungapped_to_aligned,
            }
        )

    arrays: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    offset = 0
    autocast_context = nullcontext
    if device == "cuda":
        autocast_context = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)

    for group in batches(prepared, batch_size):
        sequences = [item["sequence"] for item in group]
        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode(), autocast_context():
            hidden = model(**encoded).last_hidden_state

        for index, item in enumerate(group):
            length = len(item["sequence"])
            # Hugging Face ESM tokenization adds BOS at position zero and EOS after the sequence.
            residue_embeddings = hidden[index, 1 : length + 1].detach().float().cpu().numpy()
            if residue_embeddings.shape[0] != length:
                raise RuntimeError(f"Token/residue mismatch for {item['id']!r}")
            if storage_dtype == "float16":
                residue_embeddings = residue_embeddings.astype(np.float16)
            else:
                residue_embeddings = residue_embeddings.astype(np.float32)
            arrays.append(residue_embeddings)
            entry = dict(item)
            entry["embedding_start"] = offset
            entry["embedding_stop"] = offset + length
            entries.append(entry)
            offset += length

    concatenated = np.concatenate(arrays, axis=0)
    family_dir.mkdir(parents=True, exist_ok=True)
    temporary_embeddings = embeddings_path.with_suffix(".npy.part")
    with temporary_embeddings.open("wb") as handle:
        np.save(handle, concatenated, allow_pickle=False)
    temporary_embeddings.replace(embeddings_path)

    metadata = {
        "family_id": family_id,
        "source_fasta": str(fasta_path),
        "model": model_name,
        "embedding_shape": list(concatenated.shape),
        "embedding_dtype": str(concatenated.dtype),
        "alignment_length": alignment_lengths.pop(),
        "records": entries,
    }
    temporary_metadata = metadata_path.with_suffix(".json.part")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    print(f"Encoded {family_id}: {len(entries)} proteins, {offset} residues", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Define command-line options for frozen ESM-2 feature extraction."""

    parser = argparse.ArgumentParser(
        description="Encode ungapped proteins from aligned FASTA files with frozen ESM-2."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-families", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load ESM-2 once and encode every discovered aligned protein family."""

    args = build_parser().parse_args(argv)
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install ESM dependencies with: python -m pip install -e '.[esm]'") from error

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    device = choose_device(torch, args.device)
    print(f"Loading {args.model} on {device}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=False)
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    fastas = list(discover_fastas(Path(args.input_dir)))
    if args.max_families is not None:
        fastas = fastas[: args.max_families]
    if not fastas:
        raise SystemExit(f"No aligned FASTA files found under {args.input_dir}")

    output_root = Path(args.output_dir)
    for fasta_path in fastas:
        encode_family(
            fasta_path,
            output_root,
            tokenizer,
            model,
            torch,
            device=device,
            batch_size=args.batch_size,
            storage_dtype=args.storage_dtype,
            model_name=args.model,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
