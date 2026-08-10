# Protein-family data and ESM-2 preprocessing

This repository currently implements the first stage of the proposed conditional
PhyloGFN pipeline:

1. download the official PANTHER 19.0 protein-family FASTA archive;
2. select manageable homologous families;
3. produce a fresh multiple sequence alignment (MSA) with MAFFT;
4. remove alignment gaps before passing proteins through frozen ESM-2;
5. save residue embeddings together with exact MSA-column mappings.

PANTHER is used because its library is organized into protein families and can
later be paired with PANTHER gene-tree data. The full FASTA download is about
461 MB compressed. The tiny fixture under `tests/fixtures` is synthetic and is
only for checking the pipeline; it is not research data.

Here, a FASTA record is a **gene-tree leaf**, not automatically a unique taxon.
PANTHER families can contain paralogs from the same species. A later benchmark
can either retain those paralogs for gene-tree inference or select one ortholog
per species for a species-level experiment.

## Installation

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[esm,dev]'
```

Install MAFFT separately and make sure `mafft` is on `PATH`. On macOS:

```bash
brew install mafft
```

## Prepare PANTHER families

Download, extract, filter, and align families:

```bash
prepare-phylo-data all \
  --data-root data \
  --min-sequences 8 \
  --max-sequences 64 \
  --min-median-length 80 \
  --max-median-length 900 \
  --max-length-ratio 2.5 \
  --max-families 1000 \
  --threads 8
```

For an already downloaded/extracted collection of family FASTA files:

```bash
prepare-phylo-data prepare \
  --source-dir /path/to/family_fastas \
  --output-dir data/processed \
  --min-sequences 8 \
  --max-sequences 64 \
  --threads 8
```

The preparation command recursively discovers `.fa`, `.faa`, `.fas`, and
`.fasta` files. It removes exact duplicate sequences, rejects non-protein
records, filters extreme family sizes/lengths, optionally subsamples large
families deterministically, and calls MAFFT. It writes:

```text
data/processed/
  manifest.jsonl
  raw/<family>.fasta
  aligned/<family>.fasta
```

## Encode aligned families with ESM-2

```bash
encode-esm2 \
  --input-dir data/processed/aligned \
  --output-dir data/embeddings/esm2_t12_35M \
  --model facebook/esm2_t12_35M_UR50D \
  --device auto \
  --batch-size 4 \
  --storage-dtype float16
```

ESM-2 always receives ungapped proteins. For every family the encoder writes:

```text
data/embeddings/esm2_t12_35M/<family>/
  embeddings.npy
  metadata.json
```

`embeddings.npy` concatenates all residue embeddings. `metadata.json` records
the slice belonging to each protein as well as:

- `aligned_to_ungapped`: MSA column to residue index (`-1` for a gap);
- `ungapped_to_aligned`: residue index to MSA column.

The initial model is deliberately small. Once the pipeline is validated,
`facebook/esm2_t30_150M_UR50D` can replace it without changing the format.

## Smoke test without the PANTHER download

The synthetic fixture can exercise preparation if MAFFT is installed:

```bash
prepare-phylo-data prepare \
  --source-dir tests/fixtures/families \
  --output-dir /tmp/phylo-smoke \
  --min-sequences 4 \
  --max-sequences 16
```

Run unit tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
