import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from phylogfn_data.fasta import read_fasta
from phylogfn_data.prepare import deduplicate_records, prepare_families, select_records


FIXTURES = Path(__file__).parent / "fixtures" / "families"


class PrepareTests(unittest.TestCase):
    def test_select_records(self) -> None:
        records = read_fasta(FIXTURES / "synthetic_family.fasta")
        selected = select_records(
            records,
            min_taxa=4,
            max_taxa=5,
            min_median_length=10,
            max_median_length=100,
            max_length_ratio=2.0,
            max_sequence_length=1022,
            seed=7,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(len(selected or []), 5)

    def test_deduplicate_exact_sequences(self) -> None:
        records = read_fasta(FIXTURES / "synthetic_family.fasta")
        self.assertEqual(len(deduplicate_records(records + [records[0]])), len(records))

    def test_prepare_pipeline_with_fake_aligner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aligner = root / "fake_mafft.py"
            aligner.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "print(Path(sys.argv[-1]).read_text(), end='')\n",
                encoding="utf-8",
            )
            os.chmod(aligner, 0o755)
            output = root / "processed"
            args = Namespace(
                source_dir=str(FIXTURES),
                output_dir=str(output),
                min_taxa=4,
                max_taxa=16,
                min_median_length=10,
                max_median_length=100,
                max_length_ratio=2.0,
                max_sequence_length=1022,
                max_families=None,
                seed=7,
                threads=1,
                mafft=str(aligner),
                strict=True,
            )
            self.assertEqual(prepare_families(args), 0)
            manifest = json.loads((output / "manifest.jsonl").read_text())
            self.assertEqual(manifest["num_sequences"], 6)
            self.assertEqual(manifest["alignment_length"], 37)
            self.assertTrue(Path(manifest["aligned_fasta"]).exists())


if __name__ == "__main__":
    unittest.main()
