# PLEXUS

Structure injection into masked diffusion language models for text-attributed graphs.

Target venue: **ICML 2027** (abstract Jan 16, full paper Jan 22, 2027 — projected from the 2026
cycle, not yet officially announced; re-check the official CFP before relying on it). Internal
milestone: **experiments complete by 2026-12-03**, leaving six weeks to write. Scope is a
**Cora-only diagnostic** (C1, C2) with PubMed confirmatory; see `docs/RESEARCH_PLAN.md` §7.

---

## Status

Pre-experimental. Nothing has been run. `docs/RESEARCH_PLAN.md` holds the phase plan and the
falsifiers; `docs/EXPERIMENT_LOG.md` holds results as they land; `docs/RELATED_WORK.md` records
what is already taken, with verification notes.

## The starting point

[TAG-DLM](https://arxiv.org/abs/2606.31166) (Chen, Bei, Xu, Zhao, Chen & Tong) casts
text-attributed graph learning as masked infilling. It linearises a sampled *k*-hop ego graph into
one token sequence, LoRA-tunes LLaDA-8B on it, and injects graph structure through a **binary
topology attention mask** — no GNN, no task head. It reports the best accuracy on Cora, PubMed and
ogbn-arxiv for both node classification and link prediction.

We read the paper closely before planning around it. Two things in it are load-bearing and are
not what the framing suggests.

### 1. The diffusion model never runs diffusion

The answer position is a **single `[M]` token**, and the scoring rule is, in the paper's own words,
"a single forward pass ... scored by the log-likelihood of that digit token at the masked answer
position." There is no iterative denoising anywhere in the reported results. Every number in the
paper is produced by one bidirectional forward pass of a masked language model.

This matters for what can honestly be built on top. Any contribution framed as *accelerating the
denoising loop* — speculative decoding, fewer sampling steps, block-wise denoising — is optimising
something that does not currently execute. Those directions only become meaningful after the task
itself is made generative, and that reframing, not the acceleration, would be the contribution.

It also raises a control the paper does not run: if the diffusion machinery is inert at inference,
what is LLaDA contributing over an autoregressive LLM of the same size, LoRA-tuned on the same
prompts? That question will be asked in review. We ask it first (E0.4).

### 2. Giving the model the real graph makes it dramatically worse

The paper's own ablation, Table 4:

| Task | Dataset | ego-graph mask | star mask |
| --- | --- | --- | --- |
| Node classification | Cora | 83.95 | **90.96** |
| Node classification | PubMed | 94.12 | **96.30** |
| Link prediction | Cora | 61.76 | **91.62** |
| Link prediction | PubMed | 73.92 | **95.31** |

The star mask is a target-centred abstraction: the target reads every sampled node, and each
neighbour reads only itself and the target. The ego-graph mask exposes the *actual* observed edges.
Replacing the abstraction with the real topology costs **30 points** on Cora link prediction.

The paper reads this as validating the star mask. We think it is the most interesting result in
the paper and that the stated explanation — neighbour-neighbour edges "amplify dataset-specific
local wiring" — is asserted rather than tested. Two alternatives are at least as plausible:

- **Target reach, not neighbour edges.** Under the ego mask a hop-2 node has no edge to the
  target, so the target *cannot read it at all*; under the star mask it can. The two masks differ
  in two ways simultaneously, and the paper attributes the whole effect to one of them. The union
  mask that adds real edges while keeping target reach is not reported.
- **Distribution shift in pretrained attention.** LLaDA was pretrained with dense bidirectional
  attention. A block-sparse binary mask is an input it has never seen, and rank-64 LoRA over 20
  epochs may simply be unable to repair it. On this account the problem is not *which* topology
  but *how hard* it is imposed.

These have different consequences, and both are cheap to test.

## Claims

**C1 (diagnostic).** A large share of TAG-DLM's attributed "topology" gain is not topology. The
missing controls are density-matched random rewiring, and neighbourhoods filled with random nodes
instead of graph neighbours. *Falsified if* both controls collapse accuracy, which would mean the
graph structure is genuinely carrying the signal and TAG-DLM is right for the stated reason.

**C2 (mechanism).** The ego-graph collapse is caused by hard binary gating and by lost target
reach, not by real topology being uninformative. A **graded structural bias** — a finite,
learnable additive term on attention scores as a function of graph distance and relation type —
lets real topology help where the binary mask hurts. *Falsified if* graded biases do no better
than the ego mask at matched structural content, or if the learned biases converge to zero (the
model choosing to ignore structure, which is itself a publishable negative result about structure
injection into pretrained LMs).

**C3 (method, to be earned).** Because a graded bias degrades gracefully to the unmodified model,
it supports structure the star mask cannot express — multi-hop distance, and typed edges on
heterogeneous graphs — and wins there. *Falsified if* the star mask remains at least as good once
relation types are available.

The ordering is deliberate. C1 and C2 are diagnostic and run on a reimplementation of somebody
else's method; they are publishable as a critical study even if C3 never lands.

### Why a graded bias rather than a better mask

LLaDA's `forward` already accepts `attention_bias` of shape `(B, 1, S, S)`, and **a float tensor
is added to the attention scores before the softmax** (verified in `modeling_llada.py`, lines
1196–1205). So a graded bias needs no architectural change at all — the same entry point TAG-DLM
uses for a binary mask accepts a real-valued one.

The property that matters is that **zero bias is exactly the unmodified model**. A learned graded
bias can always retreat to the baseline if topology does not help. A hard mask cannot: it forces a
commitment to a structure the model must then cope with. That asymmetry is the whole hypothesis,
and it is why `tests/test_masks.py` asserts the zero-bias special case explicitly.

## The reimplementation problem

**TAG-DLM has no public code release** (checked 2026-09-03: no GitHub repository, and the
OpenReview forum is not publicly readable). Everything we compare against, we build. Until E0.1
reproduces the paper's reported accuracies within tolerance, every TAG-DLM number in this repo is
labelled *our reimplementation* and no claim rests on it.

## Setup

Experiments run on **ai-lab2** (`ai-lab2`), in a venv rather than conda —
conda's libmamba solver is broken on that host and `/home` is full:

```bash
ssh ai-lab2
cd ~/projects/plexus
source .venv/bin/activate
export HF_HOME=/tmp/adelechinda/hf          # interim; see RESEARCH_PLAN.md §5
export CUDA_VISIBLE_DEVICES=<an idle GPU>   # check nvidia-smi first — see below
python -m pytest tests/ -q
```

On titan3 (dev only, no training) it is conda:

```bash
conda activate plexus
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Baselines are cloned under `baselines/` (git-ignored) and kept pristine:

```bash
bash scripts/setup_baselines.sh
```

### Hardware

Training and evaluation moved off titan3 on 2026-09-03; its 8× A6000 are fully committed to
another user. See E0.0d in the experiment log.

| | titan3 (dev only) | **ai-lab2 (training)** |
| --- | --- | --- |
| GPUs | 8× RTX A6000 48 GB | **2× RTX 4090 24 GB** |
| Measured bf16 | 114 TFLOPs each | **149.5 TFLOPs each** |
| Driver / stack | 530 / CUDA 12.1 | 550.144.03, held at cu121 for parity |
| Availability | none — another user holds all 8 | 1 GPU, contended with our own HERALD jobs |

TAG-DLM was trained on 8× A100 80 GB with ZeRO-2. **LoRA on a frozen 8B backbone needs ~18.4 GB of
a 24 GB card**, so a run requires a 4090 *to itself* and seq 2048 is the ceiling — ogbn-arxiv at
seq 4096 does not fit and has been dropped. Budget and scope: `docs/RESEARCH_PLAN.md` §6–§7.

## Layout

```
plexus/
├── masks.py        mask families, graded biases, and the density-matched controls
├── data/           TAG loading, ego-graph sampling, linearisation
├── prompts/        prompt templates and span bookkeeping
├── models/         LLaDA adapter; AR-LLM control
├── metrics/        accuracy, calibration, attention-shift diagnostics
└── train/          LoRA fine-tuning harness
docs/               plan, related work, running experiment log
baselines/          upstream repos, unmodified (git-ignored)
checkpoints/        LLaDA-8B and friends (git-ignored)
experiments/        one directory per run: config + results + notes
```

## Key references

Verified against the papers themselves, not summaries. Notes in `docs/RELATED_WORK.md`.

- TAG-DLM — [arXiv:2606.31166](https://arxiv.org/abs/2606.31166). The baseline and the target of
  C1/C2. Verified 2026-09-03: title, authors and all quoted numbers read off the arXiv HTML.
- LLaDA — Nie et al., *Large Language Diffusion Models*, NeurIPS 2025. The backbone;
  `GSAI-ML/LLaDA-8B-Instruct` on HuggingFace.
- MDLM — Sahoo et al., NeurIPS 2024, [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). The
  masked-diffusion formulation TAG-DLM inherits.
- LLaGA — Chen et al., ICML 2024. The strongest LLM+graph baseline, and the source of the dataset
  preprocessing TAG-DLM adopts.
