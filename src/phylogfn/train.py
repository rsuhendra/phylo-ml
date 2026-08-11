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

from .data.features import AlignedFamilyFeatures, load_aligned_esm2
from .evaluation import EvaluationCache, evaluate_model


@dataclass(frozen=True)
class FamilyExample:
    """Location and stable identifier of one encoded protein family."""

    family_id: str
    embedding_dir: Path


def deterministic_split(family_id: str, validation_fraction: float, test_fraction: float) -> str:
    """Assign a family reproducibly using a hash independent of input ordering."""

    value = int(hashlib.sha1(family_id.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def discover_examples(root: Path) -> list[FamilyExample]:
    """Find directories containing both ESM residue arrays and metadata."""

    examples = [
        FamilyExample(path.name, path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "metadata.json").is_file() and (path / "embeddings.npy").is_file()
    ]
    if not examples:
        raise ValueError(f"No encoded family directories found under {root}")
    return examples


def family_tensors(family: AlignedFamilyFeatures, torch, device: str):
    """Move one aligned family to a device and expose sequences for rewards."""

    embeddings = torch.from_numpy(np.asarray(family.residue_embeddings)).to(device)
    mask = torch.from_numpy(np.asarray(family.residue_mask)).to(device)
    amino_acids = torch.from_numpy(np.asarray(family.amino_acid_indices)).to(device)
    sequences = dict(zip(family.identifiers, family.aligned_sequences))
    return embeddings, mask, amino_acids, sequences


def choose_device(torch, requested: str) -> str:
    """Resolve ``auto`` to CUDA, Apple MPS, or CPU in priority order."""

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    """Define data, optimization, reward, and architecture training options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trajectories-per-family", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--reward", choices=("parsimony",), default="parsimony")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--pair-dim", type=int, default=64)
    parser.add_argument("--policy-dim", type=int, default=256)
    parser.add_argument("--fitch-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument("--validation-trajectories-per-family", type=int, default=4)
    parser.add_argument("--max-validation-families", type=int)
    parser.add_argument("--reference-trees-dir", type=Path)
    parser.add_argument(
        "--selection-metric",
        choices=("normalized-tb", "mean-parsimony", "modal-rf"),
        default="normalized-tb",
        help="Validation metric minimized when saving best_checkpoint.pt",
    )
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train across families with on-policy Trajectory Balance trajectories."""

    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.trajectories_per_family < 1:
        raise SystemExit("epochs, batch size, and trajectories per family must be positive")
    if args.validation_every < 1 or args.validation_trajectories_per_family < 1:
        raise SystemExit("validation interval and trajectories must be positive")
    if args.max_validation_families is not None and args.max_validation_families < 1:
        raise SystemExit("--max-validation-families must be positive")
    if args.selection_metric == "modal-rf" and args.reference_trees_dir is None:
        raise SystemExit("--selection-metric modal-rf requires --reference-trees-dir")
    if min(args.adapter_dim, args.pair_dim, args.policy_dim, args.fitch_dim) < 1:
        raise SystemExit("adapter, pair, policy, and Fitch dimensions must be positive")
    if not 0 <= args.validation_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise SystemExit("split fractions must be between zero and one")
    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("validation and test fractions must sum to less than one")

    try:
        import torch
        from torch.nn.utils import clip_grad_norm_
    except ImportError as error:
        raise SystemExit("Install training dependencies with: python -m pip install -e '.[gfn]'") from error
    from .model.gflownet import ConditionalPhyloGFN

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
    validation_ids = {
        row["family_id"] for row in split_rows if row["split"] == "validation"
    }
    train_examples = [example for example in examples if example.family_id in train_ids]
    validation_examples = [
        example for example in examples if example.family_id in validation_ids
    ]
    if args.max_families is not None:
        train_examples = train_examples[: args.max_families]
    if args.max_validation_families is not None:
        validation_examples = validation_examples[: args.max_validation_families]
    if not train_examples:
        raise SystemExit("No training families remain after splitting")

    first_family = load_aligned_esm2(train_examples[0].embedding_dir)
    model = ConditionalPhyloGFN(
        first_family.embedding_dim,
        adapter_dim=args.adapter_dim,
        pair_dim=args.pair_dim,
        policy_dim=args.policy_dim,
        fitch_dim=args.fitch_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits.json").write_text(
        json.dumps(split_rows, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    evaluation_cache = EvaluationCache()
    selection_keys = {
        "normalized-tb": "validation_normalized_tb_loss",
        "mean-parsimony": "validation_mean_normalized_parsimony",
        "modal-rf": "validation_modal_normalized_rf",
    }
    selection_key = selection_keys[args.selection_metric]
    best_selection_value = float("inf")
    best_epoch: int | None = None
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
                pair_matrix, log_z = model.encode_family(
                    family.identifiers, embeddings, mask, amino_acids
                )
                trajectory_losses = [
                    model.sample_trajectory(
                        family.identifiers,
                        pair_matrix,
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
            should_validate = (
                bool(validation_examples) and epoch % args.validation_every == 0
            )
            if should_validate:
                validation_metrics = evaluate_model(
                    model,
                    [example.embedding_dir for example in validation_examples],
                    torch=torch,
                    device=device,
                    trajectories_per_family=args.validation_trajectories_per_family,
                    beta=args.beta,
                    reward=args.reward,
                    temperature=args.temperature,
                    seed=args.seed + epoch,
                    reference_trees_dir=args.reference_trees_dir,
                    cache=evaluation_cache,
                )
                row.update(validation_metrics)
                raw_selection_value = row[selection_key]
                selection_value = (
                    float(raw_selection_value)
                    if raw_selection_value is not None
                    else None
                )
                if (
                    selection_value is not None
                    and np.isfinite(selection_value)
                    and selection_value < best_selection_value
                ):
                    best_selection_value = selection_value
                    best_epoch = epoch
                    is_best = True
                else:
                    is_best = False
                row.update(
                    {
                        "selection_metric": selection_key,
                        "selection_value": selection_value,
                        "best_epoch": best_epoch,
                        "best_selection_value": (
                            best_selection_value if best_epoch is not None else None
                        ),
                    }
                )
            else:
                is_best = False
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            print(json.dumps(row), file=sys.stderr)
            checkpoint = {
                "architecture": "pair_fitch_v1",
                "epoch": epoch,
                "model_config": model.config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "training_args": vars(args),
                "validation_metrics": {
                    key: value
                    for key, value in row.items()
                    if key.startswith("validation_")
                },
                "selection_metric": selection_key,
                "best_epoch": best_epoch,
                "best_selection_value": (
                    best_selection_value if best_epoch is not None else None
                ),
            }
            torch.save(checkpoint, args.output_dir / "checkpoint.pt")
            if is_best:
                torch.save(checkpoint, args.output_dir / "best_checkpoint.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
