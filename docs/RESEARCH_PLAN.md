# PLEXUS — Research Plan

Living document. Amend it when evidence arrives; do not quietly drift from it.

Target: **ICML 2027**, full paper deadline **2027-01-22** (abstract 2027-01-16). Projected from
the ICML 2026 cycle; the official CFP was not published as of 2026-09-03. Confirm before relying
on it. That is roughly **20 weeks** from 2026-09-03.

---

## 0. Thesis

TAG-DLM injects graph topology into a pretrained masked diffusion LM through a binary attention
mask, and reports that the *real* ego-graph topology performs far worse than a target-centred
star abstraction (Cora link prediction: 91.62 → 61.76). We argue this is a fact about **how**
structure is imposed, not about whether topology is informative, and that the fix — a graded,
learnable additive attention bias that contains the unmodified model as its zero — makes real
topology usable and extends to structure the star mask cannot express.

## 1. What TAG-DLM actually does

Established by reading the arXiv HTML on 2026-09-03, not the abstract. These are the facts the
plan is built on; if any is wrong, the plan is wrong.

1. **Backbone** is LLaDA-8B-Instruct, frozen, with LoRA (r=64, α=64, all-linear). Only the
   adapters train.
2. **Inference is a single forward pass.** The answer position is one `[M]` token; each candidate
   answer is one digit token; the prediction is the argmax of that token's log-likelihood. The
   iterative denoising loop is never executed for NC or LP.
3. **The star mask** lets the target attend to all sampled nodes and each neighbour attend only to
   itself and the target. Default sampling is k=2 hops, K≤10 neighbours per hop.
4. **The theory (Prop. 1)** shows masked self-attention equals attention-based message passing
   over the neighbourhoods *induced by the mask*. Under the star mask that induced graph is a
   depth-1 star, so the proposition does not establish message passing over the actual TAG. The
   2-hop nodes are present in the sequence but structurally indistinguishable from 1-hop nodes.
5. **Training** is 20 epochs, AdamW, lr 5e-5, effective batch 128, seq 2048 (Cora/PubMed) and
   4096 (ogbn-arxiv), on 8× A100 80 GB with ZeRO-2.
6. **No public code.** We reimplement everything.

Consequence of (2) and (4) together: what the star mask implements is closer to *set pooling over
sampled nodes* than to graph message passing. Structure enters mainly through **which nodes get
sampled**, not through how they are wired. C1 is the test of that reading.

## 2. Where the mechanism lives

`plexus/masks.py` is the experimental surface. All families produce a float `(B, 1, S, S)` tensor
for LLaDA's `attention_bias` argument, which is added to attention scores pre-softmax (verified in
`modeling_llada.py` L1196–1205). The families are:

| Family | What it encodes | Role |
| --- | --- | --- |
| `none` | padding only | unmodified model; TAG-DLM's "w/o topo mask" |
| `star` | target hub, no neighbour-neighbour edges | TAG-DLM default; the number to beat |
| `ego` | observed edges only | TAG-DLM's failing variant |
| `ego_star` | observed edges **and** target reach | **not in the paper**; separates the two confounded changes |
| `random_rewire` | star + density-matched random non-star edges | control: is it *this* topology or *any* sparsity? |
| `HopBias` | graded additive bias by hop distance | the method; zero recovers `none` exactly |

`ego_star` is the single most important addition. TAG-DLM's `ego` mask changes two things at once
relative to `star`: it adds neighbour-neighbour edges *and* it removes the target's ability to read
hop-2 nodes. The paper credits the collapse entirely to the first. `ego_star` isolates it.

## 3. Phases

### Phase 0 — Reimplement, reproduce, and control (weeks 1–6)

Nothing is built until the baseline runs and the diagnostics report.

- **E0.1 Reproduce TAG-DLM.** Star mask, LLaDA-8B + LoRA, their recipe, on Cora and PubMed for
  NC and LP. *Gate: within 1.0 accuracy point of the reported 90.96 / 96.30 (NC) and 91.62 /
  95.31 (LP).* ogbn-arxiv reproduction is a stretch goal, not a gate — see §6 for why.
  *If the gate fails*, we do not proceed to claims about their method. We report what we could and
  could not reproduce, and say so in the paper. A failed reproduction of an uncoded paper is a
  legitimate finding but it changes the contribution, and pretending otherwise would be dishonest.

- **E0.2 The mask ladder.** `none` → `star` → `ego` → `ego_star` → `random_rewire`, everything else
  fixed, on Cora and PubMed, NC and LP. Report accuracy **with mask density beside it** in the same
  table; masks that differ in accuracy also differ in how much context each token sees, and that
  confound has to be visible.

  > **E0.2 is the fulcrum of the project.** If `ego_star` ≈ `star`, TAG-DLM's stated explanation
  > for the ego collapse is wrong and the real cause is lost target reach — a clean diagnostic
  > result. If `ego_star` ≈ `ego` (still collapsed), the hard-gating hypothesis survives and C2 is
  > the right method. If `random_rewire` ≈ `ego_star`, topology is contributing nothing beyond
  > sparsity and both readings are wrong. Run it before writing any method code.

- **E0.2b Path length to the answer, not density.** Measured in E0.0c: a mask constrains attention
  *within* a layer, but composing two layers of a star gives a complete graph over the sampled
  nodes. The star mask therefore does not restrict which nodes communicate — it restricts **how
  many layers it takes for a node to reach the answer position**. Compute, for each mask family,
  the mean layer-depth at which each node's information can first reach the `[M]` token, and test
  whether that predicts accuracy better than mask density does. Requires no training for the
  structural part; the correlation needs E0.2's accuracies.
  *Falsifier:* if accuracy tracks density and not path length, the mechanism story is wrong and C2
  should be re-derived rather than defended.

- **E0.3 Random-node control.** Keep the star mask and the prompt format; replace the sampled ego
  neighbours with uniformly random nodes from the graph. TAG-DLM's Table 5 shows accuracy rises
  with neighbour count K but never checks that the neighbours must be *graph* neighbours.
  *Falsifier for C1:* if random nodes collapse accuracy toward the K=0 row (Cora NC 80.44), the
  topology is real and C1 is dead — which we would report immediately, and pivot to C2/C3 alone.

- **E0.4 Is the diffusion model necessary?** Same prompts, same LoRA rank and budget, same masks,
  with an autoregressive 8B LLM in place of LLaDA. Because inference is one forward pass at a
  single `[M]`, the diffusion machinery may be contributing nothing that a bidirectional or even
  causal encoder would not. Reviewers will ask. *This is the ablation most likely to embarrass the
  whole framing, so it runs in Phase 0, not at rebuttal.*
  Open decision: an AR model is not bidirectional, so the honest comparison also wants an
  encoder-style control. Record the choice and its limitations in the log before running.

- **E0.5 Are the GNN baselines fair?** TAG-DLM's GNNs use "shallow text features"; the TAG
  literature has repeatedly shown GNNs on strong text embeddings are far stronger than usually
  reported. Run GCN/GAT/SAGE on modern sentence-embedding features. If a cheap GNN matches 90.96
  on Cora, the headline comparison is weaker than presented and we must say so.

### Phase 1 — Graded structural bias (weeks 7–12), conditional on E0.2

- **E1.1** Learned `HopBias`: per-hop-bucket scalars, shared across layers. Cheapest version of
  the method; establishes whether graded beats gated at matched structural content.
- **E1.2** Per-layer and per-head-group biases. Where in the network does structure belong? Prior
  work on structural biases suggests early layers; we do not assume it.
- **E1.3** Hard→soft interpolation. Sweep a temperature that carries `ego` continuously from a
  hard mask to a mild bias. If accuracy is monotone in softness, that is direct evidence for the
  distribution-shift account and against "neighbour edges are uninformative".
- **E1.4** Attention-shift diagnostic. Measure the divergence between the model's attention
  distribution with and without each mask, per layer, and correlate it with the accuracy drop.
  *This is the mechanism evidence.* Without it, C2 is a story that fits the numbers rather than an
  explanation. If the correlation is absent, C2's stated mechanism is wrong even if the method
  works, and the paper must say the method works for an unknown reason.

### Phase 2 — Structure the star cannot express (weeks 13–16), conditional on E1

- **E2.1** Heterogeneous TAGs with typed edges and relation-conditioned biases. This is the
  extension TAG-DLM's own limitations section names as future work, so it is expected and someone
  else may do it; our angle is that it is *only reachable* once graded biases fix the hard-mask
  failure, which is a claim they cannot make.
  **Open: dataset selection is not yet settled.** Needs a heterogeneous graph with genuine node
  text. Candidates to audit: ogbn-mag, DBLP, IMDB, Amazon. Do not commit until the text quality
  is inspected — several standard heterogeneous benchmarks have degenerate or missing text.
- **E2.2** Multi-hop depth: does the graded bias let k=3 help where the star saturates at k=2?
- **E2.3** Denser graphs, where the star's "target reads everything" costs sequence length fastest.

### Phase 3 — Generative TAGs (only if Phases 0–2 hold, and probably future work)

Making the task generative is what would make "diffusion language model" earn its name: co-diffuse
node text and structure, grow a graph from a prompt. It is also the point at which acceleration
work (speculative decoding, block diffusion) becomes meaningful, since only then is there an
iterative loop to accelerate. Scoped out of the ICML submission unless Phase 1 finishes early.

## 4. Metrics

Report together; each alone is gameable.

| Metric | Why |
| --- | --- |
| Accuracy (NC, LP) | Comparable to TAG-DLM's tables |
| **Mask density** | The confound in every mask comparison. Never report accuracy without it |
| Attention divergence vs. unmasked | The mechanism behind C2 |
| Learned bias magnitude per hop/layer | Whether the model chooses to use structure |
| Wall-clock and peak memory per config | Dense `(B,1,S,S)` biases are expensive; hiding that is dishonest |
| Variance over ≥3 seeds | Cora is small; several ablation gaps in the paper are within plausible seed noise |

**Standing rule.** Cora has 2708 nodes. A 1.5-point difference on it is not a result without
error bars. TAG-DLM reports single numbers; we report mean ± std over seeds, and where our error
bars overlap their claimed gaps we say so plainly.

## 5. Risks

| Risk | Mitigation |
| --- | --- |
| **Reproduction fails** (no public code) | Highest-probability failure. Budget 3 weeks. If E0.1 misses, report the gap honestly and reframe as a reproduction study plus the diagnostic. Contact authors early — do this in week 1, not week 10. |
| **E0.3 kills C1** | Fine. C2/C3 stand alone; the paper becomes a method paper rather than a method-plus-diagnostic paper. |
| **E0.4 shows diffusion is unnecessary** | Genuinely likely given the single-forward-pass inference. Would be a strong negative result about the TAG-DLM framing and should be reported as such, not buried. It does not invalidate the graded-bias method, which applies to any bidirectional LM. |
| **ogbn-arxiv is unaffordable** | Resolved by dropping it: ~61 GPU-days per run on one 4090 *and* it does not fit in 24 GB at seq 4096 (§6). The paper claims Cora with PubMed confirmatory at n=1, and says so plainly rather than implying a full sweep. |
| **Someone publishes graded structural bias for graph LLMs first** | The idea is a natural transfer of Graphormer-style spatial encoding to a pretrained LM. Sweep related work monthly and date the entries. |
| **Star mask simply wins everywhere** | Then the honest paper is "target-centred abstraction beats real topology, and here is why" — a diagnostic contribution with the method as a negative result. Decide on E0.2/E1.1, not on which story reads better. |
| **Attention-bias memory** | `(B,1,S,S)` fp32 is 67 MB/sample at S=4096, and an explicit mask disables the flash-attention SDPA backend, so the `32 × S² × 2 B` score tensor is materialised per layer. On a 24 GB card this is now a binding constraint, not an efficiency note: it is what puts S=4096 out of reach. Use bf16 biases; consider block-structured representations. |
| **GPU contention — now the top scheduling risk** | MEASURED 2026-09-03: LLaDA-8B LoRA needs ~18.4 GB of a 24 GB card, so it **cannot share a GPU with any other substantial job**. A first attempt to run `validate_attention_bias.py` OOMed because our own HERALD/v-atat jobs had taken 18 GB of the second card. PLEXUS needs one 4090 *exclusively* for the duration of a run — 18–37 h for a Cora run. This requires actual scheduling discipline between projects, not goodwill. |
| **Storage is volatile, and that is now a deliberate choice** | The 16 GB LLaDA checkpoint lives in `/tmp/adelechinda/hf` on ai-lab2's root filesystem (380 GB free), because `/home` is at 100% with 30 GB free. Checked 2026-09-03 and rejected: `/media/scratch` is root-owned *and* on the same device as `/tmp` (all three of `/`, `/tmp`, `/media/scratch` are device 66309), so it adds no capacity; `/media/drive1`, `/mnt` and `/opt` are root-owned and empty. `/media/drive2` (`/dev/sda1`, 5.4 TB free) is the only genuinely separate disk and needs an admin to create a directory. **Decision: stay on `/tmp`.** The host's `tmpfiles.d` rule (`D /tmp ... -`) sets no age threshold, so nothing is reaped mid-run; the only exposure is a reboot, which costs a 2-minute re-transfer from titan3. Mitigation: the training entrypoint asserts the snapshot exists before starting, so a wiped cache fails immediately rather than 30 hours in. Revisit only if outputs approach 380 GB, which the Cora-only scope will not. |

## 6. Compute budget

**Revised 2026-09-03 after migrating off titan3. Measurements in E0.0d and E0.0e.**

Training and evaluation run on `ai-lab2`, **one RTX 4090 24 GB**.
titan3's 8× A6000 are fully committed to another user and are not available on this project's
timeline. ai-lab2 has two 4090s, but both are contended with our own HERALD/v-atat jobs and
neither is reliably idle, so the plan budgets **one** and treats the second as upside (lever 7).

MEASURED: 4090 at **149.5 TFLOPs bf16** (8192³ matmul, torch 2.5.1+cu121) against **114 TFLOPs**
for an A6000. Per GPU the 4090 is 1.31× faster; in aggregate one 4090 is **6.1× less compute**
than titan3's eight A6000s (149.5 vs 912 TFLOPs). Driver 550.144.03 runs cu121–cu124; we hold at
cu121 for stack parity with the earlier titan3 measurements.

### Does it fit in 24 GB?

**MEASURED** on ai-lab2 GPU 1, `validate_attention_bias.py` step 3 — bf16 forward passes with a
dense `star` bias, **inference only** (no optimizer, no gradients). See E0.0e.

| Seq | Bias tensor | Peak GB | ms/fwd |
| --- | --- | --- | --- |
| 512 | 1.0 MB | 16.57 | 221 |
| 2048 | 16.8 MB | **18.59** | 1361 |
| 4096 | 67 MB | **OOM** | — |

**S=4096 out-of-memories at inference on a 24 GB card.** ogbn-arxiv is therefore not merely
expensive, it is impossible here, and its exclusion no longer rests on an estimate.

Training adds, DERIVED: LoRA r=64 all-linear is 168 M trainable, so AdamW fp32 moments + fp32
master + bf16 grads add **2.4 GB**, and gradient checkpointing stores 0.54 GB of layer inputs at
S=2048. Against the measured 18.59 GB inference peak, that projects to **~21–22 GB of 23.6 GiB at
S=2048** — it fits, but with 2 GB of headroom and no more. About 1.1 GB of the measured peak is
the script's own `output_hidden_states` retention, which training does not pay, so the true figure
is probably nearer 20.5 GB. Confirm on the first real E0.1 step rather than trusting this.

The materialised-attention term is what drives the S=4096 failure: a dense float bias disables the
flash SDPA backend (risk 7 below), so the `32 × S² × 2 B` score tensor is allocated per layer —
1.07 GB per layer at S=4096.

**Micro-batch is 1**, so the effective batch of 128 needs ~128 accumulation steps.

### The MFU assumption is wrong, and the wall-clock below is optimistic

A forward pass at S=2048 is 2 × 8.02e9 × 2048 ≈ 32.8 TFLOPs, which at the measured 149.5 TFLOPs/s
should take 220 ms. It took **1361 ms — 6.2× off peak, ~16% MFU**, against the 35% assumed below.
Part of that gap is instrumentation: the script downloads 33 hidden states as fp32 to CPU (1.1 GB
at S=2048), so 1361 ms is an upper bound on the true forward cost, not a clean measurement. But
none of the remaining gap is explained by the dense bias being cheap, and 35% is not defensible on
a memory-starved card at micro-batch 1.

*Owed: a clean forward/backward timing without `output_hidden_states`, on the first E0.1 step. If
MFU really is near 16%, lever 1 below stops being optional.*

### Wall-clock

Rescaled from the previous 8×A6000 estimates by 6.1× at ~4e10 FLOPs/token, under both the MFU
originally assumed and the one actually measured.

| Dataset | Train nodes | Seq | 20 epochs | 8×A6000 (was) | 1×4090 @35% MFU | 1×4090 @16% MFU |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | ~1.6k | 2048 | 6.6e7 tok | ~3–6 h | ~18–37 h | **~40–80 h** |
| PubMed | ~12k | 2048 | 4.9e8 tok | ~1–2 days | ~6–12 days | **~13–26 days** |
| ogbn-arxiv | ~91k | 4096 | 7.4e9 tok | ~10 days | — | **impossible — OOM** |

The right-hand column is the one to plan against until a clean timing says otherwise.

### The fixed 2048 was over half of that cost, and it was unnecessary

**MEASURED 2026-09-03**, `scripts/profile_sequence_lengths.py` over all 1624 Cora NC training
examples at k=2, K=10: token length is **p50 957, p95 1384, max 1583, mean 902**, with 2–21 nodes
per example. So *no* Cora example needs 2048; the budget was set by what fits in 24 GB, which is a
different question from what the data requires.

The micro-batch is 1, so there is no second example to align with and padding to a fixed 2048 buys
nothing whatsoever. Padding each example only to a multiple of 128 instead:

| pad to multiple of | mean padded | max | linear cost | quadratic cost |
| --- | --- | --- | --- | --- |
| 1 (exact) | 902 | 1583 | 0.441× | 0.217× |
| **128** | **966** | **1664** | **0.472×** | **0.246×** |
| 256 | 1031 | 1792 | 0.504× | 0.278× |

128 is the operating point: it keeps the number of distinct tensor shapes to ~13, which matters
for allocator churn, and costs 7% over an exact fit. Linear terms are the projections; the
quadratic ones are the bias and the score matrix, which a dense float bias forces to be
materialised because it defeats flash SDPA — so the quadratic saving is real, not notional.

**This is worth more than lever 7 and costs nothing.** Cora's 20 epochs fall from 6.6e7 to
**3.1e7 tokens**, so the pessimistic Cora column becomes **~19–38 h** rather than 40–80, and peak
memory is now set by the longest example at S=1664 rather than by 2048 — the bias drops from
16.8 MB to 11.1 MB per sample. Because the projections dominate at these lengths but the
materialised score matrix is not negligible, the true speedup should land between 2.1× and 4.1×.
The table above is **not** yet rewritten for this: it needs the clean timing from the smoke run,
and rescaling an estimate by another estimate is how the 35% MFU error happened in the first place.

### What that buys, and the seven levers

A Cora-only diagnostic at 3 seeds is ~54 LoRA runs (E0.1 6, E0.2 ladder 24, E0.3 6, E1.1 6,
E1.3 12). At 18–37 h that is 40–83 GPU-days; at the measured MFU it was 90–180 GPU-days against a
91-day calendar. **Dynamic padding takes at least 2.1× off that**, bringing the pessimistic case
to roughly 43–86 GPU-days — from "does not fit at all" to "fits only with the levers below". The
harness is now built, which returns the three weeks §7 allocated to it. Levers 1–4 are already
baked into §7; levers 5–7 are contingency, to be pulled in the order 7, 6, 5.

1. **Epoch reduction.** E0.1 runs the paper's 20 epochs, because a reproduction gate must match
   the recipe. Then measure a convergence curve on Cora *once* and run the ablation ladder at the
   smallest epoch count that preserves the star/ego ordering. If 8 epochs suffices that is 2.5×.
   Any reported ladder number states its epoch count.
2. **Seed tiering.** 3 seeds on the E0.2 fulcrum (`star` vs `ego` vs `ego_star`), 2 elsewhere.
   §4's standing rule is that a 1.5-pt Cora gap needs error bars; it is not satisfied by n=1.
3. **Task tiering.** NC and LP for the fulcrum; **LP only** for the E1.3 softness sweep, since the
   ego collapse there is 29.9 pts against 7.0 for NC and so has the most room to show an effect.
4. **PubMed becomes confirmatory with n=1.** At 6–12 days per run it is affordable 2–4 times in
   the whole project. Reserve it for `star` vs `ego_star` on LP at the end. The paper must say
   PubMed is n=1, not imply a sweep.

With levers 1–3 the core drops to roughly **19–40 GPU-days at 35% MFU, or 43–90 at 16%** before
dynamic padding, and **~20–43 at 16% with it**. That is the first version of this arithmetic that
leaves room for both the experiments and six weeks of writing. It assumes, however, that one 4090
is actually available — which on 2026-09-03 it was not for most of the day, and no amount of
efficiency fixes a card we do not have. If the clean timing disappoints, one of these gives too:

5. **Shrink the ladder.** `random_rewire` is the most droppable arm: it controls for "any sparsity"
   rather than testing the fulcrum. Dropping it saves 6 runs. `none` is nearly free to keep as a
   sanity anchor and should stay.
6. **Tighten lever 2 further:** two seeds on `ego` as well, keeping three only for `star` and
   `ego_star`. That one contrast is what E0.2 turns on, and it is the only place the standing rule
   is non-negotiable.
7. **Claim the second 4090.** It is ours, merely occupied by our own HERALD jobs. LoRA
   data-parallel across both cards all-reduces only the 168 M adapter params, so the `SYS`
   interconnect is tolerable — this buys ~2× without touching the science.

Pull 7 first: it costs only a scheduling decision between our own projects, whereas 5 and 6 cost
evidence. The cost of 7 is honest but non-technical — it means deciding that PLEXUS outranks
HERALD on this machine until December.

## 7. Timeline

**Revised 2026-09-03.** Scope is now a **Cora-only diagnostic**: C1 and C2, with PubMed
confirmatory. Phase 2 (C3) and ogbn-arxiv are out of scope for this submission. The internal
milestone is **experiments complete by 2026-12-03** (week 13), which leaves six weeks for writing
before the abstract on 2027-01-16 and the paper on 2027-01-22.

| Weeks | Dates | Work | Gate |
| --- | --- | --- | --- |
| 1–3 | Sep 3 – Sep 24 | Data pipeline, prompts, LoRA harness; `validate_attention_bias.py` step 3 on the 4090 | Measured peak memory at S=2048; harness trains Cora end-to-end |

**Week 1–3 status, 2026-09-03.** Built and tested: `plexus/data` (Cora with raw text, ego-graph
sampling), `plexus/prompts` (NC and LP linearisation with span bookkeeping), `plexus/models`
(LLaDA + LoRA + single-position scoring), `plexus/train` (the LoRA harness), `plexus/metrics`, and
`scripts/run_experiment.py`. 54 tests pass, including a stub-backbone test of the whole training
loop on CPU. Step 3 on the 4090 is done (E0.0e). **Outstanding: the end-to-end smoke run on a real
GPU**, which is blocked on a free card, not on code — see the contention note below. Until it
passes, the "harness trains Cora end-to-end" gate is unmet.
| 4–6 | Sep 24 – Oct 15 | E0.1 Cora at the paper's 20 epochs, NC+LP, 3 seeds; then the convergence curve that sets the ladder's epoch budget | **Reproduction within 1.0 pt** |
| 6–9 | Oct 15 – Nov 5 | E0.2 ladder (`none`/`ego`/`ego_star`/`random_rewire`), E0.2b path length, E0.3 random-node | **Go/no-go on framing** |
| 9–12 | Nov 5 – Nov 26 | E1.1 `HopBias`, E1.3 softness sweep (LP only), E1.4 attention-shift diagnostic | Graded > gated, with mechanism evidence |
| 13 | Nov 26 – Dec 3 | PubMed confirmatory (n=1), E0.5 GNN baselines | **Experiments complete** |
| 14–19 | Dec 3 – Jan 15 | Writing, final tables, seed top-ups on whatever the paper leans on | Abstract Jan 16 |
| 20 | Jan 15 – Jan 22 | Buffer | Paper Jan 22 |

**E0.4 runs as a cheap probe first.** The plan's position is that the AR-control ablation is the
one most likely to embarrass the framing and must not wait for rebuttal. But a second 8B backbone
is another ~16 GB of disk and ~6 runs (4.5–9 GPU-days) we do not have. So it runs at 1 seed,
reduced epochs, NC+LP in week 9 as a signal, and only earns full seeds if the signal is
interesting. The disk cost is no longer a constraint: E0.0f settled storage on `/tmp` with 380 GB
free, so a second backbone is affordable and E0.4 is gated only on GPU hours.

The schedule is serial: one GPU means nothing overlaps. Two risks eat it. The reproduction in
weeks 4–6 is the first, and it now has three weeks rather than the six an 8-GPU budget bought —
**if E0.1 has not passed by the end of week 7, stop trying to reproduce and reframe as a
reproduction study plus the E0.2/E0.2b diagnostic**, which needs no passing gate. The second is
GPU contention, and it has already cost us: on 2026-09-03 both 4090s were held by our own
HERALD/v-atat jobs for the whole working day, leaving 7–11 GB free against the 16 GB the backbone
needs, so the week-1 smoke run could not start. This is lever 7 in §6 arriving early. **The
schedule above assumes one 4090 is actually available to PLEXUS; it currently is not.**

## 8. Immediate next actions

**Revised 2026-09-03**, after the week 1–3 build. Items 2 and 3 of the original list are done
(`plexus/data`, `plexus/prompts`, `plexus/models`, `plexus/train`, `plexus/metrics`, and E0.0e's
memory measurement); ogbn-arxiv was dropped from item 2 as out of scope.

1. **Secure one 4090 for PLEXUS.** Everything below queues behind this. It is a scheduling
   decision between our own projects, not a technical problem — see lever 7 in §6.
2. **Smoke run**: `scripts/run_experiment.py --smoke`, which trains and evaluates on 24 Cora
   examples. This closes the week 1–3 gate and yields the owed measurement from E0.0e — a clean
   forward+backward timing without `output_hidden_states`, and the real training peak against the
   derived ~20.5–21 GB. If the peak exceeds ~22 GB, the sequence budget has to be cut before E0.1
   starts rather than after.
3. **Send the TAG-DLM email.** Drafted in `docs/outreach/tagdlm-code-request.md`; needs the
   corresponding author's address. The five questions in it are exactly the assumptions our
   reimplementation is carrying, the split most of all.
4. **E0.1 on Cora, NC first**, once 2 passes.
5. **Related-work sweep** on structural attention biases for graph LLMs, before writing E1 code.
   `docs/RELATED_WORK.md` is referenced by the README and does not exist yet.
