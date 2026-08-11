#!/usr/bin/env python3
"""Estimate the random-merge parsimony baseline for a saved data split.

This script deliberately reads only ``metadata.json`` files. It does not load
ESM embeddings or a model checkpoint, so it can run cheaply on CPU and does
not require reinstalling an editable checkout after pulling new code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# Prefer the current checkout over an older installed copy of the package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from phylogfn.phylo.tree_env import TreeState  # noqa: E402


def family_seed(seed: int, family_id: str) -> int:
    """Derive a stable seed that is independent of family processing order."""

    digest = hashlib.sha256(f"{seed}:{family_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def load_sequences(family_dir: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    """Read aligned sequences directly from one encoded family's metadata."""

    metadata_path = family_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records", [])
    identifiers = tuple(str(record["id"]) for record in records)
    sequences = {
        str(record["id"]): str(record["aligned_sequence"]) for record in records
    }
    if len(identifiers) < 3:
        raise ValueError(f"{family_dir.name} has fewer than three sequences")
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ValueError(f"{family_dir.name} contains inconsistent alignment lengths")
    return identifiers, sequences


def score_family(work: tuple[Path, str, int, int]) -> dict[str, float | int | str]:
    """Sample uniform merge trajectories and return one family's statistics."""

    family_dir, family_id, trees_per_family, seed = work
    identifiers, sequences = load_sequences(family_dir)
    generator = random.Random(family_seed(seed, family_id))
    alignment_length = len(next(iter(sequences.values())))
    denominator = alignment_length * max(1, len(identifiers) - 1)
    scores: list[float] = []

    for _ in range(trees_per_family):
        state = TreeState.initial(identifiers, sequences=sequences)
        while not state.is_terminal:
            actions = state.valid_actions()
            state = state.step(actions[generator.randrange(len(actions))])
        scores.append(state.terminal_fitch().score / denominator)

    return {
        "family_id": family_id,
        "num_taxa": len(identifiers),
        "alignment_length": alignment_length,
        "mean": statistics.fmean(scores),
        "best": min(scores),
    }


def build_parser() -> argparse.ArgumentParser:
    """Define the standalone random-baseline command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--trees-per-family", type=int, default=100)
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score the requested split and print aggregate JSON."""

    args = build_parser().parse_args(argv)
    if args.trees_per_family < 1 or args.workers < 1:
        raise SystemExit("--trees-per-family and --workers must be positive")
    if args.max_families is not None and args.max_families < 1:
        raise SystemExit("--max-families must be positive")

    split_rows = json.loads(args.splits.read_text(encoding="utf-8"))
    family_ids = [
        str(row["family_id"])
        for row in split_rows
        if row.get("split") == args.split
    ]
    if args.max_families is not None:
        family_ids = family_ids[: args.max_families]
    if not family_ids:
        raise SystemExit(f"No families found in split {args.split!r}")

    missing = [
        family_id
        for family_id in family_ids
        if not (args.embeddings_dir / family_id / "metadata.json").is_file()
    ]
    if missing:
        raise SystemExit(f"Missing family metadata: {', '.join(missing[:10])}")

    work = [
        (args.embeddings_dir / family_id, family_id, args.trees_per_family, args.seed)
        for family_id in family_ids
    ]
    results: list[dict[str, float | int | str]] = []
    if args.workers == 1:
        iterator = map(score_family, work)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(score_family, work)

    try:
        for index, result in enumerate(iterator, start=1):
            results.append(result)
            if index == 1 or index % 10 == 0 or index == len(work):
                running_mean = statistics.fmean(float(row["mean"]) for row in results)
                print(
                    f"Scored {index}/{len(work)} families; "
                    f"running random mean={running_mean:.6f}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    family_means = [float(result["mean"]) for result in results]
    family_bests = [float(result["best"]) for result in results]
    standard_deviation = statistics.stdev(family_means) if len(family_means) > 1 else 0.0
    summary = {
        "split": args.split,
        "families": len(results),
        "trees_per_family": args.trees_per_family,
        "total_random_trees": len(results) * args.trees_per_family,
        "random_mean_normalized_parsimony": statistics.fmean(family_means),
        "random_best_of_n_normalized_parsimony": statistics.fmean(family_bests),
        "random_family_mean_standard_deviation": standard_deviation,
        "random_family_mean_standard_error": standard_deviation / math.sqrt(len(family_means)),
        "seed": args.seed,
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
