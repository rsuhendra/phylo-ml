"""Sample gene-tree topologies from a trained conditional PhyloGFN."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .features import load_aligned_esm2
from .train import choose_device, family_tensors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--family-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--reward", choices=("parsimony", "poisson"), default="parsimony")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install training dependencies with: python -m pip install -e '.[gfn]'") from error
    from .gflownet import ConditionalPhyloGFN

    device = choose_device(torch, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ConditionalPhyloGFN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    family = load_aligned_esm2(args.family_dir)
    embeddings, mask, amino_acids, sequences = family_tensors(family, torch, device)
    torch.manual_seed(args.seed)
    generator = None
    if device != "mps":
        generator = torch.Generator(device=device).manual_seed(args.seed)

    counts: Counter[str] = Counter()
    rewards: dict[str, float] = {}
    with torch.inference_mode():
        leaf_features, log_z, site_weights = model.encode_family(
            family.identifiers, embeddings, mask, amino_acids
        )
        for _ in range(args.num_samples):
            trajectory = model.sample_trajectory(
                family.identifiers,
                leaf_features,
                log_z,
                sequences,
                beta=args.beta,
                reward=args.reward,
                temperature=args.temperature,
                generator=generator,
            )
            newick = trajectory.terminal_state.terminal_newick()
            counts[newick] += 1
            rewards[newick] = trajectory.log_reward

    result = {
        "family_id": family.family_id,
        "num_samples": args.num_samples,
        "conditional_log_z": float(log_z.cpu()),
        "site_weights": np.asarray(site_weights.cpu()).tolist(),
        "trees": [
            {
                "newick": newick,
                "count": count,
                "frequency": count / args.num_samples,
                "log_reward": rewards[newick],
            }
            for newick, count in counts.most_common()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
