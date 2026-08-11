"""Held-out evaluation for conditional gene-tree topology distributions."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data.features import load_aligned_esm2
from .phylo.baseline import neighbor_joining, p_distance
from .phylo.parsimony import normalized_parsimony_score, parse_newick
from .phylo.tree_metrics import robinson_foulds


@dataclass
class EvaluationCache:
    """Epoch-invariant baselines and reference trees loaded lazily by family."""

    nj_parsimony: dict[str, float] = field(default_factory=dict)
    reference_trees: dict[str, Any | None] = field(default_factory=dict)


def _mean(values: list[float]) -> float:
    """Return a JSON-safe mean, using NaN when no observation is available."""

    return float(np.mean(values)) if values else float("nan")


def _taxon_bin(num_taxa: int) -> str:
    """Group families into interpretable taxon-count ranges."""

    if num_taxa <= 16:
        return "3-16"
    if num_taxa <= 32:
        return "17-32"
    return "33-64+"


def _nj_parsimony(
    family_id: str,
    identifiers: tuple[str, ...],
    sequences: dict[str, str],
    cache: EvaluationCache,
) -> float:
    """Compute and cache normalized parsimony of the Neighbor-Joining baseline."""

    if family_id not in cache.nj_parsimony:
        count = len(identifiers)
        distances = np.zeros((count, count), dtype=np.float64)
        for i, first in enumerate(identifiers):
            for j in range(i + 1, count):
                value = p_distance(sequences[first], sequences[identifiers[j]])
                distances[i, j] = distances[j, i] = value
        tree = neighbor_joining(list(identifiers), distances)
        cache.nj_parsimony[family_id] = normalized_parsimony_score(tree, sequences)
    return cache.nj_parsimony[family_id]


def _reference_tree(
    family_id: str,
    reference_trees_dir: Path | None,
    cache: EvaluationCache,
):
    """Load ``<family>.nwk`` once, returning ``None`` when no reference exists."""

    if reference_trees_dir is None:
        return None
    if family_id not in cache.reference_trees:
        path = reference_trees_dir / f"{family_id}.nwk"
        cache.reference_trees[family_id] = (
            parse_newick(path.read_text(encoding="utf-8")) if path.is_file() else None
        )
    return cache.reference_trees[family_id]


def evaluate_model(
    model,
    family_dirs: Iterable[Path],
    *,
    torch,
    device: str,
    trajectories_per_family: int,
    beta: float,
    reward: str,
    temperature: float,
    seed: int,
    reference_trees_dir: Path | None = None,
    cache: EvaluationCache | None = None,
    prefix: str = "validation",
) -> dict[str, Any]:
    """Sample held-out families and aggregate flow, reward, diversity, and RF metrics.

    Evaluation is inference-only: it neither updates parameters nor contributes
    to the training objective. Raw TB loss is reported alongside its per-action
    normalization because trajectory length grows with the number of taxa.
    """

    if trajectories_per_family < 1:
        raise ValueError("Evaluation requires at least one trajectory per family")
    cache = cache or EvaluationCache()
    generator = None
    if device != "mps":
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        torch.manual_seed(seed)

    tb_losses: list[float] = []
    normalized_tb_losses: list[float] = []
    mean_parsimony: list[float] = []
    best_parsimony: list[float] = []
    modal_parsimony: list[float] = []
    nj_parsimony: list[float] = []
    unique_fractions: list[float] = []
    normalized_entropies: list[float] = []
    expected_rf: list[float] = []
    modal_rf: list[float] = []
    by_taxa: dict[str, dict[str, list[float] | int]] = {}
    evaluated_families = 0
    reference_families = 0

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for family_dir in family_dirs:
                family = load_aligned_esm2(family_dir)
                embeddings = torch.from_numpy(
                    np.asarray(family.residue_embeddings)
                ).to(device)
                mask = torch.from_numpy(np.asarray(family.residue_mask)).to(device)
                amino_acids = torch.from_numpy(
                    np.asarray(family.amino_acid_indices)
                ).to(device)
                sequences = dict(zip(family.identifiers, family.aligned_sequences))
                pair_matrix, log_z = model.encode_family(
                    family.identifiers, embeddings, mask, amino_acids
                )

                trajectories = [
                    model.sample_trajectory(
                        family.identifiers,
                        pair_matrix,
                        log_z,
                        sequences,
                        beta=beta,
                        reward=reward,
                        temperature=temperature,
                        generator=generator,
                    )
                    for _ in range(trajectories_per_family)
                ]
                family_tb = [float(item.loss.cpu()) for item in trajectories]
                family_normalized_tb = [
                    loss / max(1, item.num_actions) ** 2
                    for loss, item in zip(family_tb, trajectories)
                ]
                family_parsimony = [-item.log_reward / beta for item in trajectories]
                newicks = [item.terminal_state.terminal_newick() for item in trajectories]
                counts = Counter(newicks)
                modal_newick = counts.most_common(1)[0][0]
                parsimony_by_tree = dict(zip(newicks, family_parsimony))
                probabilities = np.asarray(list(counts.values()), dtype=np.float64)
                probabilities /= probabilities.sum()
                entropy = float(-np.sum(probabilities * np.log(probabilities)))
                normalized_entropy = (
                    entropy / math.log(trajectories_per_family)
                    if trajectories_per_family > 1
                    else 0.0
                )

                family_mean_parsimony = _mean(family_parsimony)
                family_best_parsimony = min(family_parsimony)
                family_modal_parsimony = parsimony_by_tree[modal_newick]
                family_nj = _nj_parsimony(
                    family.family_id, family.identifiers, sequences, cache
                )
                tb_losses.extend(family_tb)
                normalized_tb_losses.extend(family_normalized_tb)
                mean_parsimony.append(family_mean_parsimony)
                best_parsimony.append(family_best_parsimony)
                modal_parsimony.append(family_modal_parsimony)
                nj_parsimony.append(family_nj)
                unique_fractions.append(len(counts) / trajectories_per_family)
                normalized_entropies.append(normalized_entropy)

                bin_name = _taxon_bin(family.num_sequences)
                bin_metrics = by_taxa.setdefault(
                    bin_name,
                    {"families": 0, "normalized_tb": [], "mean_parsimony": []},
                )
                bin_metrics["families"] += 1
                bin_metrics["normalized_tb"].extend(family_normalized_tb)
                bin_metrics["mean_parsimony"].append(family_mean_parsimony)

                reference = _reference_tree(
                    family.family_id, reference_trees_dir, cache
                )
                if reference is not None:
                    try:
                        trajectory_rf = [
                            robinson_foulds(item.terminal_state.terminal_tree(), reference)[1]
                            for item in trajectories
                        ]
                        modal_tree = next(
                            item.terminal_state.terminal_tree()
                            for item, newick in zip(trajectories, newicks)
                            if newick == modal_newick
                        )
                        expected_rf.append(_mean(trajectory_rf))
                        modal_rf.append(robinson_foulds(modal_tree, reference)[1])
                        reference_families += 1
                    except ValueError as error:
                        warnings.warn(
                            f"Skipping reference RF for {family.family_id}: {error}",
                            stacklevel=2,
                        )
                evaluated_families += 1
    finally:
        model.train(was_training)

    summarized_bins = {
        name: {
            "families": values["families"],
            "normalized_tb_loss": _mean(values["normalized_tb"]),
            "mean_normalized_parsimony": _mean(values["mean_parsimony"]),
        }
        for name, values in sorted(by_taxa.items())
    }
    return {
        f"{prefix}_families": evaluated_families,
        f"{prefix}_trajectories": evaluated_families * trajectories_per_family,
        f"{prefix}_tb_loss": _mean(tb_losses),
        f"{prefix}_normalized_tb_loss": _mean(normalized_tb_losses),
        f"{prefix}_mean_normalized_parsimony": _mean(mean_parsimony),
        f"{prefix}_best_normalized_parsimony": _mean(best_parsimony),
        f"{prefix}_modal_normalized_parsimony": _mean(modal_parsimony),
        f"{prefix}_nj_normalized_parsimony": _mean(nj_parsimony),
        f"{prefix}_unique_topology_fraction": _mean(unique_fractions),
        f"{prefix}_normalized_entropy": _mean(normalized_entropies),
        f"{prefix}_reference_families": reference_families,
        f"{prefix}_expected_normalized_rf": _mean(expected_rf) if expected_rf else None,
        f"{prefix}_modal_normalized_rf": _mean(modal_rf) if modal_rf else None,
        f"{prefix}_by_taxa": summarized_bins,
    }


def build_parser() -> argparse.ArgumentParser:
    """Define standalone validation/test checkpoint evaluation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--trajectories-per-family", type=int, default=100)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reference-trees-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate one checkpoint on a named held-out split and emit JSON metrics."""

    args = build_parser().parse_args(argv)
    if args.trajectories_per_family < 1 or args.beta <= 0 or args.temperature <= 0:
        raise SystemExit("trajectories, beta, and temperature must be positive")
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install evaluation dependencies with: python -m pip install -e '.[gfn]'") from error
    from .model.gflownet import ConditionalPhyloGFN

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    split_rows = json.loads(args.splits.read_text(encoding="utf-8"))
    family_ids = [
        str(row["family_id"])
        for row in split_rows
        if row.get("split") == args.split
    ]
    family_dirs = [args.embeddings_dir / family_id for family_id in family_ids]
    missing = [path.name for path in family_dirs if not path.is_dir()]
    if missing:
        raise SystemExit(f"Missing encoded families: {', '.join(missing[:10])}")
    if not family_dirs:
        raise SystemExit(f"No families assigned to split {args.split!r}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "pair_fitch_v1":
        raise SystemExit("Checkpoint is not compatible with the pairwise-ESM/Fitch model")
    model = ConditionalPhyloGFN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate_model(
        model,
        family_dirs,
        torch=torch,
        device=device,
        trajectories_per_family=args.trajectories_per_family,
        beta=args.beta,
        reward="parsimony",
        temperature=args.temperature,
        seed=args.seed,
        reference_trees_dir=args.reference_trees_dir,
        prefix=args.split,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        **metrics,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
