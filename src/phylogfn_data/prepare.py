from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from .fasta import FastaRecord, discover_fastas, read_fasta, ungap, validate_protein, write_fasta


PANTHER_VERSION = "19.0"
PANTHER_FASTA_URL = (
    "https://data.pantherdb.org/ftp/panther_library/19.0/PANTHER19.0_fasta.tgz"
)


@dataclass(frozen=True)
class FamilyManifest:
    family_id: str
    source_path: str
    raw_fasta: str
    aligned_fasta: str
    num_sequences: int
    alignment_length: int
    min_ungapped_length: int
    median_ungapped_length: float
    max_ungapped_length: int
    gap_fraction: float
    min_sequence_coverage: float
    mean_sequence_coverage: float


@dataclass(frozen=True)
class PreparedCandidate:
    manifest: FamilyManifest
    staged_raw_fasta: Path
    staged_aligned_fasta: Path


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "phylogfn-data/0.1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while chunk := response.read(chunk_size):
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = 100.0 * downloaded / total
                print(f"\rDownloading {destination.name}: {percent:5.1f}%", end="", file=sys.stderr)
    if total:
        print(file=sys.stderr)
    temporary.replace(destination)


def extract_tar_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name}")
            member_path = (destination / member.name).resolve()
            if destination_resolved not in member_path.parents and member_path != destination_resolved:
                raise ValueError(f"Unsafe archive member: {member.name}")
        # Explicit checks above keep this compatible with Python 3.10/3.11,
        # where tarfile's newer extraction filters are unavailable.
        bundle.extractall(destination)


def stable_family_id(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root)
    stem = relative.name
    for suffix in (".fasta", ".faa", ".fas", ".fa"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    clean = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    if clean:
        return clean
    return hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:16]


def deduplicate_records(records: list[FastaRecord]) -> list[FastaRecord]:
    seen_ids: set[str] = set()
    kept: list[FastaRecord] = []
    for record in records:
        sequence = ungap(record.sequence)
        if record.identifier in seen_ids:
            continue
        seen_ids.add(record.identifier)
        kept.append(FastaRecord(record.identifier, record.description, sequence))
    return kept


def select_records(
    records: list[FastaRecord],
    *,
    min_taxa: int,
    max_taxa: int,
    min_median_length: int,
    max_median_length: int,
    max_length_ratio: float,
    max_sequence_length: int,
    seed: int,
) -> list[FastaRecord] | None:
    valid = [record for record in records if validate_protein(record.sequence)]
    valid = deduplicate_records(valid)
    if len(valid) < min_taxa:
        return None

    lengths = [len(record.sequence) for record in valid]
    median_length = statistics.median(lengths)
    if not min_median_length <= median_length <= max_median_length:
        return None
    if max(lengths) > max_sequence_length:
        return None
    if max(lengths) / max(1, min(lengths)) > max_length_ratio:
        return None

    if len(valid) > max_taxa:
        family_seed = seed ^ int(
            hashlib.sha1("|".join(record.identifier for record in valid).encode()).hexdigest()[:8], 16
        )
        generator = random.Random(family_seed)
        valid = generator.sample(valid, max_taxa)
        valid.sort(key=lambda record: record.identifier)
    return valid


def run_mafft(input_fasta: Path, output_fasta: Path, threads: int, executable: str) -> None:
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"Cannot find {executable!r} on PATH. Install MAFFT or pass --mafft /absolute/path/to/mafft."
        )
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_fasta.with_suffix(output_fasta.suffix + ".part")
    command = [executable, "--auto", "--thread", str(threads), str(input_fasta)]
    with temporary.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"MAFFT failed for {input_fasta}:\n{result.stderr}")
    temporary.replace(output_fasta)


def alignment_statistics(records: list[FastaRecord]) -> tuple[int, float, float, float]:
    """Return alignment length, overall gap fraction, min coverage, and mean coverage."""
    if not records:
        raise ValueError("Alignment contains no sequences")
    alignment_lengths = {len(record.sequence) for record in records}
    if len(alignment_lengths) != 1:
        raise ValueError("Alignment contains inconsistent sequence lengths")
    alignment_length = alignment_lengths.pop()
    if alignment_length == 0:
        raise ValueError("Alignment has zero columns")

    coverages = [len(ungap(record.sequence)) / alignment_length for record in records]
    total_cells = len(records) * alignment_length
    total_residues = sum(len(ungap(record.sequence)) for record in records)
    gap_fraction = 1.0 - (total_residues / total_cells)
    return alignment_length, gap_fraction, min(coverages), statistics.mean(coverages)


def passes_alignment_filters(
    *,
    alignment_length: int,
    gap_fraction: float,
    min_sequence_coverage: float,
    max_alignment_length: int | None,
    min_gap_fraction: float,
    max_gap_fraction: float,
    min_sequence_coverage_required: float,
) -> bool:
    return (
        (max_alignment_length is None or alignment_length <= max_alignment_length)
        and min_gap_fraction <= gap_fraction <= max_gap_fraction
        and min_sequence_coverage >= min_sequence_coverage_required
    )


def validate_prepare_arguments(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    if args.min_taxa < 1 or args.max_taxa < args.min_taxa:
        raise ValueError("Require 1 <= --min-sequences <= --max-sequences")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.max_families is not None and args.max_families < 1:
        raise ValueError("--max-families must be positive")

    max_alignment_length = getattr(args, "max_alignment_length", None)
    if max_alignment_length is not None and max_alignment_length < 1:
        raise ValueError("--max-alignment-length must be positive")
    candidate_pool_size = getattr(args, "candidate_pool_size", None)
    if candidate_pool_size is not None and candidate_pool_size < 1:
        raise ValueError("--candidate-pool-size must be positive")

    min_gap = getattr(args, "min_gap_fraction", 0.0)
    max_gap = getattr(args, "max_gap_fraction", 1.0)
    coverage = getattr(args, "min_sequence_coverage", 0.0)
    if not 0.0 <= min_gap <= max_gap <= 1.0:
        raise ValueError("Require 0 <= --min-gap-fraction <= --max-gap-fraction <= 1")
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("--min-sequence-coverage must be between 0 and 1")


def prepare_families(args: argparse.Namespace) -> int:
    validate_prepare_arguments(args)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    aligned_dir = output_dir / "aligned"
    manifest_path = output_dir / "manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_alignment_length = getattr(args, "max_alignment_length", None)
    min_gap_fraction = getattr(args, "min_gap_fraction", 0.0)
    max_gap_fraction = getattr(args, "max_gap_fraction", 1.0)
    minimum_coverage = getattr(args, "min_sequence_coverage", 0.0)
    rank_by_gap = getattr(args, "rank_by_gap", False)
    candidate_pool_size = getattr(args, "candidate_pool_size", None)
    if rank_by_gap and candidate_pool_size is None and args.max_families is not None:
        candidate_pool_size = 4 * args.max_families

    candidates: list[PreparedCandidate] = []
    skipped = 0
    with tempfile.TemporaryDirectory(prefix=".phylo-candidates-", dir=output_dir) as staging_name:
        staging_root = Path(staging_name)
        for source_path in discover_fastas(source_dir):
            if candidate_pool_size is not None and len(candidates) >= candidate_pool_size:
                break
            if not rank_by_gap and args.max_families is not None and len(candidates) >= args.max_families:
                break
            try:
                records = read_fasta(source_path)
                records = select_records(
                    records,
                    min_taxa=args.min_taxa,
                    max_taxa=args.max_taxa,
                    min_median_length=args.min_median_length,
                    max_median_length=args.max_median_length,
                    max_length_ratio=args.max_length_ratio,
                    max_sequence_length=args.max_sequence_length,
                    seed=args.seed,
                )
                if records is None:
                    skipped += 1
                    continue

                family_id = stable_family_id(source_path, source_dir)
                staged_raw_path = staging_root / "raw" / f"{family_id}.fasta"
                staged_aligned_path = staging_root / "aligned" / f"{family_id}.fasta"
                write_fasta(records, staged_raw_path)
                run_mafft(staged_raw_path, staged_aligned_path, args.threads, args.mafft)

                aligned_records = read_fasta(staged_aligned_path)
                if {record.identifier for record in aligned_records} != {
                    record.identifier for record in records
                }:
                    raise RuntimeError(f"MAFFT changed sequence identifiers for {family_id}")
                alignment_length, gap_fraction, min_coverage, mean_coverage = alignment_statistics(
                    aligned_records
                )
                if not passes_alignment_filters(
                    alignment_length=alignment_length,
                    gap_fraction=gap_fraction,
                    min_sequence_coverage=min_coverage,
                    max_alignment_length=max_alignment_length,
                    min_gap_fraction=min_gap_fraction,
                    max_gap_fraction=max_gap_fraction,
                    min_sequence_coverage_required=minimum_coverage,
                ):
                    skipped += 1
                    continue

                raw_path = raw_dir / f"{family_id}.fasta"
                aligned_path = aligned_dir / f"{family_id}.fasta"
                lengths = [len(record.sequence) for record in records]
                entry = FamilyManifest(
                    family_id=family_id,
                    source_path=str(source_path),
                    raw_fasta=str(raw_path),
                    aligned_fasta=str(aligned_path),
                    num_sequences=len(records),
                    alignment_length=alignment_length,
                    min_ungapped_length=min(lengths),
                    median_ungapped_length=statistics.median(lengths),
                    max_ungapped_length=max(lengths),
                    gap_fraction=gap_fraction,
                    min_sequence_coverage=min_coverage,
                    mean_sequence_coverage=mean_coverage,
                )
                candidates.append(PreparedCandidate(entry, staged_raw_path, staged_aligned_path))
                print(
                    f"Candidate {family_id}: {len(records)} sequences, "
                    f"{alignment_length} columns, {gap_fraction:.3f} gaps",
                    file=sys.stderr,
                )
            except (OSError, ValueError, RuntimeError) as error:
                if args.strict:
                    raise
                skipped += 1
                print(f"Skipping {source_path}: {error}", file=sys.stderr)

        if rank_by_gap:
            candidates.sort(
                key=lambda candidate: (
                    candidate.manifest.gap_fraction,
                    candidate.manifest.family_id,
                )
            )
        eligible_candidates = len(candidates)
        if args.max_families is not None:
            candidates = candidates[: args.max_families]

        raw_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_suffix(".jsonl.part")
        with temporary_manifest.open("w", encoding="utf-8") as manifest:
            for selection_rank, candidate in enumerate(candidates, start=1):
                shutil.copy2(candidate.staged_raw_fasta, candidate.manifest.raw_fasta)
                shutil.copy2(candidate.staged_aligned_fasta, candidate.manifest.aligned_fasta)
                manifest_entry = asdict(candidate.manifest)
                manifest_entry.update(
                    {
                        "selection_rank": selection_rank,
                        "selection_policy": "lowest_gap" if rank_by_gap else "source_order",
                        "eligible_candidates_considered": eligible_candidates,
                    }
                )
                manifest.write(json.dumps(manifest_entry, sort_keys=True) + "\n")
        temporary_manifest.replace(manifest_path)

    print(f"Prepared {len(candidates)} families; skipped {skipped} files", file=sys.stderr)
    return 0 if candidates else 1


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dir", required=True, help="Directory containing one FASTA per family")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument(
        "--min-sequences", "--min-taxa", dest="min_taxa", type=int, default=8,
        help="Minimum gene-tree leaves (FASTA records) per family",
    )
    parser.add_argument(
        "--max-sequences", "--max-taxa", dest="max_taxa", type=int, default=64,
        help="Maximum leaves after deterministic subsampling",
    )
    parser.add_argument("--min-median-length", type=int, default=80)
    parser.add_argument("--max-median-length", type=int, default=900)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--max-sequence-length", type=int, default=1022)
    parser.add_argument("--max-alignment-length", type=int)
    parser.add_argument("--min-gap-fraction", type=float, default=0.0)
    parser.add_argument("--max-gap-fraction", type=float, default=1.0)
    parser.add_argument(
        "--min-sequence-coverage", type=float, default=0.0,
        help="Minimum fraction of MSA columns occupied by every retained sequence",
    )
    parser.add_argument(
        "--rank-by-gap", action="store_true",
        help="Select the lowest-gap alignments from the candidate pool",
    )
    parser.add_argument(
        "--candidate-pool-size", type=int,
        help="Stop after this many post-alignment candidates before ranking",
    )
    parser.add_argument("--max-families", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mafft", default="mafft")
    parser.add_argument("--strict", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-panther", help="Download and extract PANTHER FASTA data")
    download.add_argument("--data-root", default="data")
    download.add_argument("--url", default=PANTHER_FASTA_URL)
    download.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Filter family FASTAs and align them with MAFFT")
    add_prepare_arguments(prepare)

    all_parser = subparsers.add_parser("all", help="Download PANTHER and prepare selected families")
    all_parser.add_argument("--data-root", default="data")
    all_parser.add_argument("--url", default=PANTHER_FASTA_URL)
    all_parser.add_argument("--force", action="store_true")
    all_parser.add_argument("--output-dir")
    all_parser.add_argument(
        "--min-sequences", "--min-taxa", dest="min_taxa", type=int, default=8,
        help="Minimum gene-tree leaves (FASTA records) per family",
    )
    all_parser.add_argument(
        "--max-sequences", "--max-taxa", dest="max_taxa", type=int, default=64,
        help="Maximum leaves after deterministic subsampling",
    )
    all_parser.add_argument("--min-median-length", type=int, default=80)
    all_parser.add_argument("--max-median-length", type=int, default=900)
    all_parser.add_argument("--max-length-ratio", type=float, default=2.5)
    all_parser.add_argument("--max-sequence-length", type=int, default=1022)
    all_parser.add_argument("--max-alignment-length", type=int)
    all_parser.add_argument("--min-gap-fraction", type=float, default=0.0)
    all_parser.add_argument("--max-gap-fraction", type=float, default=1.0)
    all_parser.add_argument("--min-sequence-coverage", type=float, default=0.0)
    all_parser.add_argument("--rank-by-gap", action="store_true")
    all_parser.add_argument("--candidate-pool-size", type=int)
    all_parser.add_argument("--max-families", type=int)
    all_parser.add_argument("--seed", type=int, default=17)
    all_parser.add_argument("--threads", type=int, default=1)
    all_parser.add_argument("--mafft", default="mafft")
    all_parser.add_argument("--strict", action="store_true")
    return parser


def download_panther(data_root: Path, url: str, force: bool) -> Path:
    archive = data_root / "downloads" / Path(url).name
    extracted = data_root / "extracted" / f"panther-{PANTHER_VERSION}-fasta"
    if force or not archive.exists():
        download_file(url, archive)
    if force and extracted.exists():
        shutil.rmtree(extracted)
    if not extracted.exists() or not any(extracted.iterdir()):
        extract_tar_safely(archive, extracted)
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download-panther":
        extracted = download_panther(Path(args.data_root), args.url, args.force)
        print(extracted)
        return 0
    if args.command == "prepare":
        return prepare_families(args)
    if args.command == "all":
        source = download_panther(Path(args.data_root), args.url, args.force)
        args.source_dir = str(source)
        if args.output_dir is None:
            args.output_dir = str(Path(args.data_root) / "processed")
        return prepare_families(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
