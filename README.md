# PLEXUS

**Reachability, not topology.** A diagnostic study of how graph structure should enter a masked
diffusion language model, and of a published result that turns out to measure something other than
what it is read as measuring.

[TAG-DLM](https://arxiv.org/abs/2606.31166) injects graph topology into LLaDA-8B through a binary
attention mask, and reports that swapping its target-centred *star* abstraction for the graph's
*real* ego-graph edges costs nearly thirty points of Cora link-prediction accuracy (91.62 → 61.76).
That is read as evidence that real topology is harmful. We show the comparison is confounded, that
the damage comes from the target losing sight of its own neighbourhood, and that restoring only
that reach recovers the entire collapse while keeping every real edge.

---

## What is here

| | |
| --- | --- |
| **Paper** | `submission/main.tex` → 7 pages plus a 15-page appendix, ICML format |
| **Proposal** | `submission/proposal.tex` → 2 pages, shares `references.bib` with the paper |
| **Experiment log** | `docs/EXPERIMENT_LOG.md` → 1,295 lines, one dated entry per finding |
| **Library** | `plexus/` → 2,145 lines |
| **Tests** | `tests/` → 1,436 lines, 82 tests |
| **Runs** | 44 LoRA fine-tuning runs, 31 frozen evaluations, ≈497 GPU-hours |

Every number in the paper and the proposal is a macro emitted by
`scripts/collect_results.py` directly from run artefacts. Neither document can state a figure the
experiments do not support, and a cell with no completed run behind it typesets as a visible
marker rather than being quietly omitted.

## The three findings

**The collapse is real, and it is not about topology.** Replacing the star mask with observed
ego-graph edges costs 15.77 points on Cora link prediction (88.31 → 72.54). `ego_star`, which
differs from `ego` in exactly one respect — the target keeps direct reach into its neighbourhood,
every neighbour–neighbour edge still present — recovers 18.21 points and finishes *above* the star
abstraction proposed to replace real topology. Both effects hold at p < 0.0001 at all three seeds
under paired decision-level tests, and they are the only comparisons in the paper that survive
correction for the 54 tests it reports (threshold 0.0009).

**Node classification cannot see the question.** The whole trained ladder spans 1.91 points while
per-arm standard deviations across seeds reach ±1.8, so the ladder is narrower than the
uncertainty on the cells inside it. Not one node-classification comparison separates at more than
one of its seeds. We report that as a task on which the question cannot be asked, not as a
negative answer.

**The graded bias expresses more and gains nothing.** `HopBias`, a learnable additive bias indexed
by shortest-path distance whose zero recovers the unmodified backbone exactly, is provably strictly
more expressive than the mask family. Given the freedom to place any distance-indexed bias, the
optimiser found nothing that beat the attention the model already had. That is reported as the
finding rather than as a shortfall.

## Why the instrument is the point

The project spends more effort on knowing what it cannot measure than on any single result.

- **A measured noise floor.** Two configurations were run to completion on two different machines
  with identical seed and code. They disagree by 1.11 and 1.29 points, so 1.29 is the smallest
  difference this harness can interpret, and no gap below it is called a difference whatever its
  sign.
- **Decision-level tests, not means.** Where per-example predictions exist, arms are compared with
  McNemar's test over the same examples. A family that changes nothing predicts symmetric
  discordant pairs, which is what the ladder is actually arguing about.
- **Correction for the number of tests.** Reporting every seed as its own replicate took the table
  to 54 tests, where an uncorrected 0.05 is met by chance by construction. The corrected threshold
  is emitted as a macro so the paper cannot drift from the number of tests actually run.
- **A reproduction gate that is not passed.** Our star mask reaches 3.14 points below the reported
  node-classification number and 3.31 below the link-prediction one. TAG-DLM released no code, so
  this is stated as a result with its assumptions rather than hidden.
- **Results retired by the instrument.** An earlier positive finding of our own did not survive the
  paired tests and was withdrawn. `docs/EXPERIMENT_LOG.md` records that rather than deleting it.

## Setup

```bash
conda activate plexus
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Requires Python ≥ 3.10. Fourteen tests gate on a cached LLaDA-8B and skip without it; point
`HF_HOME` and `PLEXUS_DATA_ROOT` at persistent paths, since anything under `/tmp` will not survive
a reboot and the backbone is a 16 GB download.

Regenerate every figure the documents quote:

```bash
python scripts/collect_results.py     # writes submission/results_generated.tex
cd submission && ./build.sh           # reports pages, unresolved refs, overfull boxes
```

### Hardware

Training ran on **titan3**, 8× RTX A6000 48 GB, one arm per card. Four runs on **ai-lab2**
(2× RTX 4090) exist only as the cross-machine replicates behind the noise floor. LoRA r=64 over a
frozen 8B backbone needs about 18.4 GB, activations checkpointed, micro-batch 1 to an effective
batch of 128, sequences truncated at 2048 and padded to a multiple of 128.

## Layout

```
plexus/
├── masks.py        mask families, graded biases, density-matched controls
├── data/           TAG loading, ego-graph sampling, linearisation
├── prompts/        prompt templates and span bookkeeping
├── models/         LLaDA adapter
├── metrics/        accuracy, calibration, attention-shift diagnostics
└── train/          LoRA fine-tuning harness
scripts/            run queue, result collection, paired tests, figure generation
docs/               research plan and the running experiment log
experiments/        one directory per run: config, history, per-example predictions
submission/         paper, proposal, shared bibliography, figures
baselines/          upstream repos, unmodified (git-ignored)
```

## Key references

Verified against the papers themselves rather than summaries.

- **TAG-DLM** — [arXiv:2606.31166](https://arxiv.org/abs/2606.31166). The baseline and the subject
  of the diagnosis. No public code release as of 2026-09-03, so everything compared against is
  reimplemented here.
- **LLaDA** — Nie et al., *Large Language Diffusion Models*, NeurIPS 2025 (oral). The backbone;
  `GSAI-ML/LLaDA-8B-Instruct`.
- **MDLM** — Sahoo et al., NeurIPS 2024, [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). The
  masked-diffusion formulation TAG-DLM inherits.
- **LLaGA** — Chen et al., ICML 2024. The strongest LLM+graph baseline and the source of the
  dataset preprocessing TAG-DLM adopts.
