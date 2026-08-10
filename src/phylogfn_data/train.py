"""Train the residue-aware conditional phylogenetic GFlowNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import AlignedFamilyFeatures, load_aligned_esm2


@dataclass(frozen=True)
class FamilyExample:
    family_id: str
    embedding_dir: Path


def deterministic_split(family_id: str, validation_fraction: float, test_fraction: float) -> str:
    value = int(hashlib.sha1(family_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def discover_examples(root: Path) -> list[FamilyExample]:
    examples = [
        FamilyExample(path.name, path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "metadata.json").is_file() and (path / "embeddings.npy").is_file()
    ]
    if not examples:
        raise ValueError(f"No encoded family directories found under {root}")
    return examples


def family_tensors(family: AlignedFamilyFeatures, torch, device: str):
    embeddings = torch.from_numpy(np.asarray(family.residue_embeddings)).to(device)
    mask = torch.from_numpy(np.asarray(family.residue_mask)).to(device)
    amino_acids = torch.from_numpy(np.asarray(family.amino_acid_indices)).to(device)
    sequences = dict(zip(family.identifiers, family.aligned_sequences))
    return embeddings, mask, amino_acids, sequences


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trajectories-per-family", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--reward", choices=("parsimony", "poisson"), default="parsimony")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--policy-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--sequence-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.trajectories_per_family < 1:
        raise SystemExit("epochs, batch size, and trajectories per family must be positive")
    if not 0 <= args.validation_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise SystemExit("split fractions must be between zero and one")
    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("validation and test fractions must sum to less than one")

    try:
        import torch
        from torch.nn.utils import clip_grad_norm_
    except ImportError as error:
        raise SystemExit("Install training dependencies with: python -m pip install -e '.[gfn]'") from error
    from .gflownet import ConditionalPhyloGFN

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(torch, args.device)
    examples = discover_examples(args.embeddings_dir)
    split_rows = [
        {
            "family_id": example.family_id,
            "split": deterministic_split(
                example.family_id, args.validation_fraction, args.test_fraction
            ),
        }
        for example in examples
    ]
    train_ids = {row["family_id"] for row in split_rows if row["split"] == "train"}
    train_examples = [example for example in examples if example.family_id in train_ids]
    if args.max_families is not None:
        train_examples = train_examples[: args.max_families]
    if not train_examples:
        raise SystemExit("No training families remain after splitting")

    first_family = load_aligned_esm2(train_examples[0].embedding_dir)
    model = ConditionalPhyloGFN(
        first_family.embedding_dim,
        adapter_dim=args.adapter_dim,
        policy_dim=args.policy_dim,
        num_heads=args.num_heads,
        sequence_layers=args.sequence_layers,
        dropout=args.dropout,
        max_alignment_length=max(2048, first_family.alignment_length),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits.json").write_text(
        json.dumps(split_rows, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            random.shuffle(train_examples)
            epoch_losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            pending_losses = []
            for family_index, example in enumerate(train_examples, start=1):
                family = load_aligned_esm2(example.embedding_dir)
                if family.embedding_dim != first_family.embedding_dim:
                    raise ValueError(f"Embedding dimension changed in family {family.family_id}")
                embeddings, mask, amino_acids, sequences = family_tensors(family, torch, device)
                leaf_features, log_z, _ = model.encode_family(
                    family.identifiers, embeddings, mask, amino_acids
                )
                trajectory_losses = [
                    model.sample_trajectory(
                        family.identifiers,
                        leaf_features,
                        log_z,
                        sequences,
                        beta=args.beta,
                        reward=args.reward,
                        temperature=args.temperature,
                    ).loss
                    for _ in range(args.trajectories_per_family)
                ]
                family_loss = torch.stack(trajectory_losses).mean()
                pending_losses.append(family_loss)

                end_of_batch = len(pending_losses) == args.batch_size
                end_of_epoch = family_index == len(train_examples)
                if end_of_batch or end_of_epoch:
                    loss = torch.stack(pending_losses).mean()
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    epoch_losses.append(float(loss.detach().cpu()))
                    pending_losses.clear()

            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "num_train_families": len(train_examples),
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), file=sys.stderr)
            checkpoint = {
                "epoch": epoch,
                "model_config": model.config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "training_args": vars(args),
            }
            torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
