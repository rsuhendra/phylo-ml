from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised in lightweight environments
    torch = None

from phylogfn.evaluation import EvaluationCache, evaluate_model
from phylogfn.train import deterministic_split, main as train_main


def write_encoded_family(root: Path, family_id: str) -> Path:
    """Create a tiny residue-embedding family for evaluation integration tests."""

    family_dir = root / family_id
    family_dir.mkdir(parents=True)
    sequences = ("ACDEFG", "ACDFFG", "VCDEYG", "VCDFYG")
    embeddings = np.random.default_rng(4).normal(
        size=(sum(map(len, sequences)), 6)
    ).astype(np.float32)
    np.save(family_dir / "embeddings.npy", embeddings)
    records = []
    start = 0
    for index, sequence in enumerate(sequences):
        stop = start + len(sequence)
        records.append(
            {
                "id": f"leaf_{index}",
                "aligned_sequence": sequence,
                "embedding_start": start,
                "embedding_stop": stop,
                "ungapped_to_aligned": list(range(len(sequence))),
            }
        )
        start = stop
    (family_dir / "metadata.json").write_text(
        json.dumps({"family_id": family_id, "records": records}),
        encoding="utf-8",
    )
    return family_dir


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class EvaluationTests(unittest.TestCase):
    def test_evaluation_reports_flow_reward_diversity_baseline_and_rf(self) -> None:
        from phylogfn.model.gflownet import ConditionalPhyloGFN

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            family_dir = write_encoded_family(root / "embeddings", "family_eval")
            reference_dir = root / "references"
            reference_dir.mkdir()
            (reference_dir / "family_eval.nwk").write_text(
                "((leaf_0,leaf_1),(leaf_2,leaf_3));\n", encoding="utf-8"
            )
            model = ConditionalPhyloGFN(
                esm_dim=6,
                adapter_dim=8,
                pair_dim=5,
                policy_dim=12,
                fitch_dim=4,
                num_heads=2,
                dropout=0.0,
            )
            metrics = evaluate_model(
                model,
                [family_dir],
                torch=torch,
                device="cpu",
                trajectories_per_family=3,
                beta=2.0,
                reward="parsimony",
                temperature=1.0,
                seed=9,
                reference_trees_dir=reference_dir,
                cache=EvaluationCache(),
            )
            self.assertTrue(model.training)
            self.assertEqual(metrics["validation_families"], 1)
            self.assertEqual(metrics["validation_trajectories"], 3)
            self.assertGreaterEqual(metrics["validation_normalized_tb_loss"], 0.0)
            self.assertGreaterEqual(metrics["validation_unique_topology_fraction"], 1 / 3)
            self.assertEqual(metrics["validation_reference_families"], 1)
            self.assertIsNotNone(metrics["validation_modal_normalized_rf"])
            self.assertIn("3-16", metrics["validation_by_taxa"])

    def test_training_print_metrics_are_saved_and_best_checkpoint_is_selected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            embeddings_root = root / "embeddings"
            validation_fraction = 0.5
            train_id = next(
                f"family_{index}"
                for index in range(100)
                if deterministic_split(f"family_{index}", validation_fraction, 0.0)
                == "train"
            )
            validation_id = next(
                f"family_{index}"
                for index in range(100)
                if deterministic_split(f"family_{index}", validation_fraction, 0.0)
                == "validation"
            )
            write_encoded_family(embeddings_root, train_id)
            write_encoded_family(embeddings_root, validation_id)
            output_dir = root / "run"
            self.assertEqual(
                train_main(
                    [
                        "--embeddings-dir",
                        str(embeddings_root),
                        "--output-dir",
                        str(output_dir),
                        "--epochs",
                        "1",
                        "--trajectories-per-family",
                        "1",
                        "--validation-trajectories-per-family",
                        "2",
                        "--adapter-dim",
                        "8",
                        "--pair-dim",
                        "5",
                        "--policy-dim",
                        "12",
                        "--fitch-dim",
                        "4",
                        "--num-heads",
                        "2",
                        "--dropout",
                        "0",
                        "--validation-fraction",
                        str(validation_fraction),
                        "--test-fraction",
                        "0",
                        "--device",
                        "cpu",
                    ]
                ),
                0,
            )
            row = json.loads(
                (output_dir / "metrics.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(row["validation_families"], 1)
            self.assertIn("validation_normalized_tb_loss", row)
            self.assertEqual(row["best_epoch"], 1)
            self.assertTrue((output_dir / "best_checkpoint.pt").is_file())


if __name__ == "__main__":
    unittest.main()
