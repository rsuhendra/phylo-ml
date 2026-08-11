# PhyloGFN: residue-aware protein language models for gene-tree inference

This repository implements an experimental conditional GFlowNet that learns to
sample **unrooted protein gene-tree topologies** for previously unseen homologous
families. It combines frozen ESM-2 residue embeddings, MSA-aware adaptation, and
a bottom-up tree policy trained from phylogenetic rewards.

The central design choice is important: **ESM embeddings are not pooled into one
vector per protein.** Ungapped proteins are encoded by ESM-2, each residue is
returned to its MSA column, and homologs interact within aligned columns. The
model compares pairs at residue-matched sites and only then pools those
comparisons into pairwise sequence evidence. Exact recursively updated Fitch
features carry the parsimony-relevant state of every partial tree.

## Source layout

The import package is `phylogfn`; `src/` is the standard Python source-layout
directory rather than part of the import name.

```text
src/phylogfn/
  data/       FASTA/MSA preparation, ESM-2 encoding, features, simulation
  model/      residue-pair adapter, forward policy, GFlowNet objective
  phylo/      tree environment, rewards, baselines, topology metrics
  train.py    multi-family training entry point
  sample.py   checkpoint inference and topology sampling
```

The command-line programs retain their existing names, so scripts using
`train-phylogfn`, `encode-esm2`, or the other commands do not need to change.

## Implemented pipeline

1. Download and filter PANTHER 19 protein families.
2. Align each family with MAFFT while preserving distinct gene-tree leaves.
3. Encode ungapped proteins with frozen ESM-2 and save residue-level embeddings.
4. Scatter residues back onto the shared MSA coordinate system.
5. Adapt features across homologs and build masked leaf-pair evidence.
6. Construct an unrooted binary topology through reversible merges while
   updating exact protein Fitch features for every partial subtree.
7. Train a family-conditioned forward policy and conditional partition function
   with the Trajectory Balance objective.
8. Sample multiple topologies to represent uncertainty rather than returning
   only one greedy tree.

Training currently targets normalized unordered amino-acid parsimony. A small
Poisson-20 likelihood oracle remains available as an independent diagnostic,
but the Fitch-state policy is intentionally restricted to parsimony training;
likelihood training requires recursively updated Felsenstein features.

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

The trainable adapter projects ESM features, adds explicit amino-acid identity,
and attends across taxa independently within each MSA column. It deliberately
uses neither absolute MSA-column embeddings nor a second Transformer along each
sequence. For every leaf pair, symmetric residue comparisons are pooled only
over columns at which both leaves contain residues, yielding a tensor shaped
`[sequences, sequences, pair dimension]`.

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

After every epoch, the trainer evaluates the validation families and prints one
JSON record containing training loss plus held-out metrics. Validation defaults
to four sampled trajectories per family and can be controlled independently:

```bash
train-phylogfn \
  --embeddings-dir data/embeddings/esm2_t12_35M \
  --output-dir runs/parsimony \
  --epochs 20 \
  --validation-every 1 \
  --validation-trajectories-per-family 4 \
  --selection-metric normalized-tb \
  --device auto
```

Reported validation metrics include raw and per-action-normalized TB loss,
mean/best/modal normalized parsimony, Neighbor-Joining parsimony, unique-topology
fraction, normalized sample entropy, and taxon-count strata. Supplying
`--reference-trees-dir trees/validation` additionally reports expected and
modal normalized RF distance for matching `<family>.nwk` files.

Families are deterministically assigned to train, validation, and test splits
by family identifier. The same model processes different numbers of leaves, so
training is conditionally amortized across families and taxon counts. Every
unrooted topology has a canonical representation rooted on the pendant edge of
the lexicographically first leaf; that artificial root is only a representation
device and does not turn the task into rooted-tree inference.

The run directory contains the latest `checkpoint.pt`, validation-selected
`best_checkpoint.pt`, complete `metrics.jsonl`, and `splits.json`.
Checkpoints created by the earlier absolute-position/sequence-Transformer/
pooled-leaf architecture are not compatible with this model and must be
retrained.

## 4. Sample a posterior-like set of trees

```bash
sample-phylogfn \
  --checkpoint runs/parsimony/checkpoint.pt \
  --family-dir data/embeddings/esm2_t12_35M/FAMILY \
  --output runs/parsimony/FAMILY.samples.json \
  --num-samples 1000 \
  --device auto
```

The output reports unique Newick topologies, sample frequencies, rewards, and
the learned conditional `log Z`. These frequencies are model samples;
calibration against known or high-quality reference trees must be measured
rather than assumed.

At each construction step, an action logit combines two branches. The ESM branch
aggregates cross-subtree pair evidence plus each candidate's relationship to the
fixed anchor and remaining forest. The Fitch branch compares the two partial
trees' possible root amino-acid sets at every valid site and supplies the exact
immediate normalized mutation cost. Once a merge is selected, its new Fitch
state is computed deterministically by intersection when the child state sets
overlap and union plus one mutation otherwise.

Candidate pairs are scored as one vectorized tensor batch rather than through a
Python loop. During a trajectory, the policy also carries tensorized Fitch
states, subtree sizes, anchor-pair sums, and cross-subtree ESM sums forward after
each merge. Only statistics involving the newly merged subtree are formed; the
policy does not reconstruct unchanged partial subtrees from their leaves.
Candidate logits whose two subtrees survive a merge are retained as part of the
trajectory cache, so only logits involving the newly created subtree pass
through the Fitch encoder and action network. During training, the Python tree
environment carries topology only; accumulated Fitch scores are updated once in
the tensor cache and provide the terminal parsimony reward. The independent CPU
Fitch implementation remains the correctness oracle used by tests and scoring
commands.

## Controlled experiments and baselines

After model and checkpoint selection are fixed, evaluate the locked test split
with the same suite:

```bash
evaluate-phylogfn \
  --checkpoint runs/parsimony/best_checkpoint.pt \
  --embeddings-dir data/embeddings/esm2_t12_35M \
  --splits runs/parsimony/splits.json \
  --split test \
  --trajectories-per-family 100 \
  --output runs/parsimony/test_metrics.json
```

Do not use repeated test runs to choose hyperparameters; that would turn the
test set into another validation set.

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
distribution-shift experiments, posterior calibration tests, and Fitch-only,
ESM-only, pair-dimension, ESM-model-size, and reward ablations. Species-tree
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
