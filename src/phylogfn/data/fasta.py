"""Small, dependency-free FASTA parsing and alignment-coordinate utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


FASTA_SUFFIXES = {".fa", ".faa", ".fas", ".fasta"}
GAP_CHARS = frozenset("-.")
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_AMINO_ACIDS = STANDARD_AMINO_ACIDS | frozenset("BXZJUO")


@dataclass(frozen=True)
class FastaRecord:
    """One normalized FASTA record with identifier, full header, and sequence."""

    identifier: str
    description: str
    sequence: str


def normalize_sequence(sequence: str) -> str:
    """Remove whitespace and terminal-stop markers and normalize case."""

    return "".join(sequence.split()).upper().replace("*", "")


def read_fasta(path: Path) -> list[FastaRecord]:
    """Parse a FASTA file and reject blank headers or empty records."""

    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def emit() -> None:
        """Finalize the record accumulated since the preceding header."""

        if header is None:
            return
        sequence = normalize_sequence("".join(sequence_parts))
        if not sequence:
            raise ValueError(f"Empty FASTA record {header!r} in {path}")
        identifier = header.split(maxsplit=1)[0]
        records.append(FastaRecord(identifier, header, sequence))

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                emit()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Blank FASTA header in {path}:{line_number}")
                sequence_parts = []
            elif header is None:
                raise ValueError(f"Sequence before first header in {path}:{line_number}")
            else:
                sequence_parts.append(line)
    emit()
    return records


def write_fasta(records: Iterable[FastaRecord], path: Path, width: int = 80) -> None:
    """Write records to FASTA, creating parent directories when necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.description}\n")
            for start in range(0, len(record.sequence), width):
                handle.write(record.sequence[start : start + width] + "\n")


def discover_fastas(root: Path) -> Iterator[Path]:
    """Yield supported FASTA files recursively in deterministic path order."""

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in FASTA_SUFFIXES:
            yield path


def ungap(sequence: str) -> str:
    """Remove alignment gap characters while preserving residue order."""

    return "".join(character for character in sequence if character not in GAP_CHARS)


def validate_protein(sequence: str) -> bool:
    """Return whether a nonempty sequence contains only supported protein codes."""

    raw = ungap(sequence)
    return bool(raw) and set(raw) <= ALLOWED_AMINO_ACIDS


def alignment_maps(aligned_sequence: str) -> tuple[list[int], list[int]]:
    """Build inverse mappings between MSA columns and ungapped residue indices."""

    aligned_to_ungapped: list[int] = []
    ungapped_to_aligned: list[int] = []
    residue_index = 0
    for column_index, character in enumerate(aligned_sequence):
        if character in GAP_CHARS:
            aligned_to_ungapped.append(-1)
        else:
            aligned_to_ungapped.append(residue_index)
            ungapped_to_aligned.append(column_index)
            residue_index += 1
    return aligned_to_ungapped, ungapped_to_aligned
