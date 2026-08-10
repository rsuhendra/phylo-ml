import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from phylogfn_data.fasta import FastaRecord, read_fasta
from phylogfn_data.prepare import (
    alignment_statistics,
    deduplicate_records,
    passes_alignment_filters,
    prepare_families,
    select_records,
)


FIXTURES = Path(__file__).parent / "fixtures" / "families"


class PrepareTests(unittest.TestCase):
    def test_alignment_statistics_and_filters(self) -> None:
        records = [
            FastaRecord("a", "a", "AC-D"),
            FastaRecord("b", "b", "A--D"),
        ]
        length, gap_fraction, min_coverage, mean_coverage = alignment_statistics(records)
        self.assertEqual(length, 4)
        self.assertAlmostEqual(gap_fraction, 3 / 8)
        self.assertAlmostEqual(min_coverage, 0.5)
        self.assertAlmostEqual(mean_coverage, 0.625)
        self.assertTrue(
            passes_alignment_filters(
                alignment_length=length,
                gap_fraction=gap_fraction,
                min_sequence_coverage=min_coverage,
                max_alignment_length=512,
                min_gap_fraction=0.1,
                max_gap_fraction=0.5,
                min_sequence_coverage_required=0.5,
            )
        )

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
        identical_new_leaf = FastaRecord("new_taxon", "new_taxon", records[0].sequence)
        self.assertEqual(len(deduplicate_records(records + [identical_new_leaf])), len(records) + 1)

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
            self.assertEqual(manifest["gap_fraction"], 0.0)
            self.assertTrue(Path(manifest["aligned_fasta"]).exists())

    def test_rank_by_gap_keeps_only_selected_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "families"
            source.mkdir()
            fixture_text = (FIXTURES / "synthetic_family.fasta").read_text()
            (source / "low_gap.fasta").write_text(fixture_text)
            (source / "high_gap.fasta").write_text(fixture_text)

            aligner = root / "fake_mafft.py"
            aligner.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "path = Path(sys.argv[-1])\n"
                "gaps = '-' * (5 if 'high_gap' in path.name else 1)\n"
                "for line in path.read_text().splitlines():\n"
                "    print(line if line.startswith('>') else line + gaps)\n",
                encoding="utf-8",
            )
            os.chmod(aligner, 0o755)
            output = root / "processed"
            args = Namespace(
                source_dir=str(source),
                output_dir=str(output),
                min_taxa=4,
                max_taxa=16,
                min_median_length=10,
                max_median_length=100,
                max_length_ratio=2.0,
                max_sequence_length=1022,
                max_alignment_length=512,
                min_gap_fraction=0.0,
                max_gap_fraction=1.0,
                min_sequence_coverage=0.0,
                rank_by_gap=True,
                candidate_pool_size=2,
                max_families=1,
                seed=7,
                threads=1,
                mafft=str(aligner),
                strict=True,
            )
            self.assertEqual(prepare_families(args), 0)
            manifest = json.loads((output / "manifest.jsonl").read_text())
            self.assertEqual(manifest["family_id"], "low_gap")
            self.assertEqual(manifest["selection_policy"], "lowest_gap")
            self.assertEqual(manifest["eligible_candidates_considered"], 2)
            self.assertEqual(
                [path.name for path in (output / "aligned").glob("*.fasta")],
                ["low_gap.fasta"],
            )


if __name__ == "__main__":
    unittest.main()
