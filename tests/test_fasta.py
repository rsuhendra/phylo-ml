import unittest
from pathlib import Path

from phylogfn_data.fasta import alignment_maps, read_fasta, ungap, validate_protein


FIXTURES = Path(__file__).parent / "fixtures" / "families"


class FastaTests(unittest.TestCase):
    def test_read_and_ungap(self) -> None:
        records = read_fasta(FIXTURES / "synthetic_family.fasta")
        self.assertEqual(len(records), 6)
        self.assertEqual(ungap("AC-D.E"), "ACDE")
        self.assertTrue(all(validate_protein(record.sequence) for record in records))

    def test_alignment_maps(self) -> None:
        aligned_to_ungapped, ungapped_to_aligned = alignment_maps("AC-D.E")
        self.assertEqual(aligned_to_ungapped, [0, 1, -1, 2, -1, 3])
        self.assertEqual(ungapped_to_aligned, [0, 1, 3, 5])


if __name__ == "__main__":
    unittest.main()
