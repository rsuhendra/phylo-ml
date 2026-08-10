# PhyloGFN: residue-aware protein language models for gene-tree inference

This repository implements an experimental conditional GFlowNet that learns to
sample **unrooted protein gene-tree topologies** for previously unseen homologous
families. It combines frozen ESM-2 residue embeddings, MSA-aware adaptation, and
a bottom-up tree policy trained from phylogenetic rewards.

The central design choice is important: **ESM embeddings are not mean-pooled at
the model input.** Ungapped proteins are encoded by ESM-2, each residue is
returned to its MSA column, homologs interact within aligned columns, and the
adapted residue representations are processed along the sequence. Learned site
weights pool them only after those operations. Plain per-protein ESM mean
pooling remains available solely as an ablation baseline.

## Implemented pipeline

1. Download and filter PANTHER 19 protein families.
2. Align each family with MAFFT while preserving distinct gene-tree leaves.
3. Encode ungapped proteins with frozen ESM-2 and save residue-level embeddings.
4. Scatter residues back onto the shared MSA coordinate system.
5. Adapt features across homologs at each site and along each aligned protein.
6. Construct an unrooted binary topology through reversible subtree merges.
7. Train a family-conditioned forward policy and conditional partition function
   with the Trajectory Balance objective.
8. Sample multiple topologies to represent uncertainty rather than returning
   only one greedy tree.

The current rewards are normalized unordered amino-acid parsimony and a small
Poisson-20 likelihood oracle. The latter uses one shared branch length and is a
controlled development baseline, not a replacement for IQ-TREE with LG+Gamma.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[esm,gfn,dev]'
```

Install MAFFT separately and ensure `mafft` is on `PATH` (for example,
`brew install mafft` on macOS).

## 1. Prepare PANTHER families

Download, extract, filter, and freshly align up to 1,000 families:

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

For the processing choices closest to the published PANTHER PLM-phylogeny
benchmark while retaining a larger training collection:

```bash
prepare-phylo-data all \
  --data-root data \
  --output-dir data/processed/panther_paper_like \
  --min-sequences 20 \
  --max-sequences 64 \
  --min-median-length 80 \
  --max-median-length 900 \
  --max-length-ratio 2.5 \
  --max-alignment-length 512 \
  --min-gap-fraction 0.10 \
  --rank-by-gap \
  --candidate-pool-size 4000 \
  --max-families 1000 \
  --threads 8
```

This is a paper-like training set, not an exact reproduction of the fixed
500-family benchmark. The exact benchmark should remain held out. A FASTA
record is a gene-tree leaf, not necessarily a unique species: paralogs from the
same species are intentionally retained. Only duplicate identifiers are
removed; biologically distinct records with identical sequences are preserved.

Prepared files have this layout:

```text
data/processed/panther_paper_like/
  manifest.jsonl
  raw/<family>.fasta
  aligned/<family>.fasta
```

## 2. Encode ungapped proteins with ESM-2

```bash
encode-esm2 \
  --input-dir data/processed/panther_paper_like/aligned \
  --output-dir data/embeddings/esm2_t12_35M \
  --model facebook/esm2_t12_35M_UR50D \
  --device auto \
  --batch-size 4 \
  --storage-dtype float16
```

ESM-2 receives ungapped proteins. For each family, `embeddings.npy` stores all
residue embeddings and `metadata.json` stores each protein's slice plus exact
`aligned_to_ungapped` and `ungapped_to_aligned` maps. The training loader uses
these maps to reconstruct a tensor with shape `[sequences, MSA columns,
embedding dimension]`; gap positions are masked rather than encoded by ESM.

## 3. Train the conditional GFlowNet

```bash
train-phylogfn \
  --embeddings-dir data/embeddings/esm2_t12_35M \
  --output-dir runs/parsimony \
  --epochs 20 \
  --batch-size 1 \
  --trajectories-per-family 4 \
  --reward parsimony \
  --beta 10 \
  --device auto
```

Families are deterministically assigned to train, validation, and test splits
by family identifier. The same model processes different numbers of leaves, so
training is conditionally amortized across families and taxon counts. Every
unrooted topology has a canonical representation rooted on the pendant edge of
the lexicographically first leaf; that artificial root is only a representation
device and does not turn the task into rooted-tree inference.

The run directory contains `checkpoint.pt`, `metrics.jsonl`, and `splits.json`.

## 4. Sample a posterior-like set of trees

```bash
sample-phylogfn \
  --checkpoint runs/parsimony/checkpoint.pt \
  --family-dir data/embeddings/esm2_t12_35M/FAMILY \
  --output runs/parsimony/FAMILY.samples.json \
  --num-samples 1000 \
  --device auto
```

The output reports unique Newick topologies, sample frequencies, rewards, the
learned conditional `log Z`, and MSA-site weights. These frequencies are model
samples; calibration against known or high-quality reference trees must be
measured rather than assumed.

## Controlled experiments and baselines

Generate alignments with known trees:

```bash
simulate-phylo-data \
  --output-dir data/simulated \
  --num-families 100 \
  --min-leaves 8 \
  --max-leaves 32
```

The simulator writes `aligned/<family>.fasta`, `raw/<family>.fasta`, and
`trees/<family>.nwk`. It currently uses Poisson-20 substitutions without indels,
duplication, or loss, so it is a topology recovery smoke test rather than a
realistic evolutionary benchmark.

Build a Neighbor-Joining baseline and compare a predicted topology with a
reference using root-invariant Robinson-Foulds distance:

```bash
build-nj-tree --alignment FAMILY.fasta --output nj.nwk
compare-phylo-trees --tree predicted.nwk --reference true_tree.nwk --json
```

Reward-oracle commands are also available independently:

```bash
score-parsimony --alignment FAMILY.fasta --tree candidate.nwk --beta 10 --json
score-likelihood --alignment FAMILY.fasta --tree candidate.nwk --json
```

Parsimony is normalized by alignment length and the maximum number of changes
per site before reward scaling, which makes the reward magnitude more comparable
across family sizes. Gaps, `?`, and `X` are missing states; `B`, `Z`, and `J`
retain their standard amino-acid ambiguity.

## Current research boundary

This code is a complete prototype for residue-aware conditioning, topology
construction, Trajectory Balance training, sampling, and controlled evaluation.
It does **not** yet establish that the method improves over standard phylogenetic
software. A defensible study still needs held-out PANTHER reference trees,
IQ-TREE/RAxML comparisons under strong substitution models, taxon-count and
distribution-shift experiments, posterior calibration tests, and ablations for
mean pooling, the residue adapter, ESM model size, and reward choice. Species-tree
inference and gene-tree/species-tree reconciliation are outside the current
task.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The test suite covers preprocessing, residue-to-MSA reconstruction, the
residue-aware adapter, reversible tree construction, policy sampling,
Trajectory Balance gradients, a one-epoch train-and-sample run, reward oracles,
simulation, Neighbor Joining, and RF distance.
