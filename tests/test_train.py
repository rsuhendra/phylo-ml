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

from phylogfn.train import main
from phylogfn.sample import main as sample_main


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class TrainingTests(unittest.TestCase):
    def test_one_epoch_writes_a_loadable_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            family_dir = root / "embeddings" / "family_a"
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
                json.dumps({"family_id": "family_a", "records": records}),
                encoding="utf-8",
            )

            output_dir = root / "run"
            self.assertEqual(
                main(
                    [
                        "--embeddings-dir",
                        str(root / "embeddings"),
                        "--output-dir",
                        str(output_dir),
                        "--epochs",
                        "1",
                        "--trajectories-per-family",
                        "1",
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
                        "0",
                        "--test-fraction",
                        "0",
                        "--device",
                        "cpu",
                    ]
                ),
                0,
            )
            checkpoint = torch.load(
                output_dir / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["architecture"], "pair_fitch_v1")
            self.assertEqual(checkpoint["model_config"]["esm_dim"], 6)
            self.assertEqual(checkpoint["model_config"]["pair_dim"], 5)
            self.assertTrue((output_dir / "metrics.jsonl").is_file())
            self.assertTrue((output_dir / "splits.json").is_file())

            samples_path = output_dir / "samples.json"
            self.assertEqual(
                sample_main(
                    [
                        "--checkpoint",
                        str(output_dir / "checkpoint.pt"),
                        "--family-dir",
                        str(family_dir),
                        "--output",
                        str(samples_path),
                        "--num-samples",
                        "3",
                        "--device",
                        "cpu",
                    ]
                ),
                0,
            )
            samples = json.loads(samples_path.read_text(encoding="utf-8"))
            self.assertEqual(samples["num_samples"], 3)
            self.assertEqual(sum(tree["count"] for tree in samples["trees"]), 3)


if __name__ == "__main__":
    unittest.main()
