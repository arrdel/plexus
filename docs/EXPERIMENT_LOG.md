# PLEXUS — Experiment Log

Append-only. One entry per experiment, including the ones that went badly. A log that only records
successes is not evidence of anything.

**Template**

```
## <ID> — <one-line title>
Date: YYYY-MM-DD · Config: experiments/<id>/ · Seeds: <n>

**Hypothesis.** What we expected and why.
**Falsifier.** What outcome would have contradicted it. Written *before* running.
**Setup.** Model, data, mask family, budget. Enough to rerun.
**Result.** Numbers, with error bars and with mask density beside accuracy.
**Reading.** What it means. Label each statement measured / derived / assumed.
**Caveats.** What could still be wrong. What was not controlled.
**Consequence.** What changes in RESEARCH_PLAN.md as a result.
```

---

## E0.0 — Source audit of TAG-DLM
Date: 2026-09-03 · Config: n/a · Seeds: n/a

Not an experiment. Recorded because the plan depends on these facts and they should be
challengeable.

**Method.** Read the arXiv HTML of [2606.31166](https://arxiv.org/abs/2606.31166) in full,
including ablations and limitations. Fetched `modeling_llada.py` from
`GSAI-ML/LLaDA-8B-Instruct` and read the attention path.

**Findings (all MEASURED from the source text unless marked).**

1. Inference is a **single forward pass** at a **single `[M]` token**, scored by the
   log-likelihood of a digit option. The iterative denoising loop is not run for NC or LP.
   (§5.1, "Evaluation protocol".)
2. The star mask gives the target read access to all sampled nodes; neighbours read only
   themselves and the target. (§3.3.)
3. The ego-graph mask underperforms the star mask by **7.0 pts** (Cora NC), **29.9 pts** (Cora
   LP), **21.4 pts** (PubMed LP). (Table 4.)
4. The paper's explanation for (3) is that neighbour-neighbour edges "amplify dataset-specific
   local wiring". No experiment separates this from the alternative that the ego mask also
   removes the target's access to hop-2 nodes. (DERIVED: the union mask is absent from the paper;
   the confound follows from the mask definitions in §3.3.)
5. Prop. 1 establishes message passing over the **mask-induced** graph. Under the star mask that
   graph has depth 1. (§4.2, §4.3.)
6. Backbone LLaDA-8B-Instruct, frozen, LoRA r=64 α=64; 20 epochs, lr 5e-5, effective batch 128;
   seq 2048 / 4096; 8× A100 80 GB, ZeRO-2. (§5.1, "Training".)
7. Frozen LLaDA-8B scores near chance (Cora NC 52.65, LP ~48–52), so the entire effect is from
   LoRA tuning. (Table 1.)
8. **No public code release.** GitHub search returns nothing; the OpenReview forum
   (`VaZl85Y1AT`) is not readable through the public API (0 notes returned).
9. LLaDA's `forward` accepts `attention_bias` of shape `(B, 1, S, S)`; **float tensors are added
   to attention scores pre-softmax**, bool tensors are treated as hard gates. (`modeling_llada.py`
   L1196–1205, L1268–1276.) Graded biases therefore need no architectural change.

**Consequence.** The plan is built around (4) and (9): `ego_star` is added to the mask ladder to
resolve the confound, and `HopBias` exploits the float path. (1) demotes every
acceleration-flavoured direction, because there is no denoising loop to accelerate. (8) means
three weeks are budgeted for reproduction and every TAG-DLM number here is labelled *our
reimplementation* until E0.1 passes.

**Caveats.** All numbers above are transcribed from the arXiv HTML rendering; they have not been
checked against a PDF. Item (4) is our reading of an absence, and the authors may have run the
union mask and omitted it. Ask them.

---

## E0.0b — Environment and mask-family implementation
Date: 2026-09-03 · Config: `plexus/masks.py`, `tests/test_masks.py`

**What was built.** Six mask families behind one interface producing LLaDA-compatible float
biases: `none`, `star`, `ego`, `ego_star`, `random_rewire`, and graded `HopBias`.

**Verified by test.**
- `star` lets the target read hop-2 spans; `ego` does not. This is the confound in E0.0(4),
  now asserted in code rather than argued in prose.
- `ego_star` restores target reach while keeping real edges and excluding non-edges.
- `random_rewire` is **exactly** density-matched to `ego_star` across seeds. The first
  implementation was not: it matched total edge count, but target-incident edges are already
  implied by the star, so it added too many random edges and came out denser than the condition
  it controls. Caught by the density test.
- Zero `HopBias` reproduces the unmodified model bit-for-bit — the property that lets the method
  retreat to baseline.
- `MASK_BIAS + MASK_BIAS` stays finite and softmaxes to 0. LLaDA adds its padding mask to our
  bias internally, and `finfo.min` would overflow to `-inf` and produce NaNs in SDPA.

**Environment.** conda `plexus`, torch 2.5.1+cu121, 8× A6000 measured at 114.5 TFLOPs bf16 each.
transformers 5.16.1 loads LLaDA's remote config successfully.

**Caveats.** `HopBias.node_bias` currently approximates pairwise hop distance as `|h_i - h_j|`
composed through the target. That is exact for the star and an upper bound otherwise; exact APSP
is available via `pairwise_hops` but is not yet wired into the bias path. Fix before E1.1, and do
not report E1 results computed with the approximation without saying so.

---

## E0.0c — The star mask blocks neighbour-to-neighbour flow for exactly one layer
Date: 2026-09-03 · Config: `scripts/validate_attention_bias.py` · Seeds: n/a (deterministic)

**Hypothesis.** A custom float `attention_bias` reaches LLaDA-8B's attention and enforces the
specified topology.

**Falsifier.** If a neighbour's hidden states are unaffected by the mask, the bias is not being
applied and every downstream mask experiment is measuring nothing.

**Setup.** LLaDA-8B-Instruct, fp32 on CPU (all 8 GPUs were occupied by another job). A 4-node toy
ego graph — target, two hop-1 neighbours, one hop-2 node — linearised behind a prompt prefix.
We edit node 3's text and track the resulting perturbation at node 1, which is *not* adjacent to
node 3, across all 33 hidden-state outputs.

**Result.** Max |Δ| at node 1's span, by depth:

| Mask | layer 1 | layer 4 | final | Δ at target (final) |
| --- | --- | --- | --- | --- |
| `none` | 2.45e-01 | 5.67e-01 | 3.28e+01 | 1.70e+01 |
| `star` | **0.00e+00** | 5.10e-01 | 1.43e+01 | 5.41e+00 |
| `star-strict` (prompt also gated) | **0.00e+00** | 2.49e-01 | 5.60e+01 | 1.67e+01 |

Mask densities excluding padding: `none` 1.000, `star` 0.809, `ego` 0.809, `ego_star` 0.880.

**Reading.** MEASURED: the bias is applied correctly — node 1 is *exactly* unperturbed at layer 1
under both star variants, and perturbed under `none`. The mechanism works.

MEASURED, and more interesting: **the block lasts exactly one layer.** From layer 2 onward
information flows from node 3 to node 1 regardless of the mask, because both are adjacent to the
target and the target relays between them. By the final layer the perturbation is within a factor
of ~2 of the unmasked model.

DERIVED: TAG-DLM's §3.3 rationale — the star mask "suppress[es] direct communication among
neighbours" — is true per layer and false for the network. Prop. 1 is explicitly a single-layer
result, and §4.4 extends it to the composed model by assuming the induced neighbourhood is
preserved under composition. It is not: composing two layers of a star yields a *complete* graph
over the sampled nodes. Over 32 layers the star mask does not restrict which nodes communicate at
all; it only controls **how many layers it takes**, and hence the path length from each node to
the answer position.

DERIVED, and this is the new mechanistic hypothesis: what a topology mask actually controls in a
deep transformer is the **layer-depth required for information to reach the answer position**.
The star minimises it (everything reaches the target in one layer). The ego mask raises it (hop-2
nodes need two layers, through a hop-1 node's limited span). Since the prediction is read off a
single `[M]` token, path length to that token is exactly what should matter.

This predicts, testably: (i) `ego_star` recovers most of `star`'s accuracy, because it restores
one-layer target reach while keeping real edges; (ii) mask families should order by mean path
length to the answer position, not by density; (iii) the star/ego gap should shrink in shallower
models. Note that `star` and `ego` have *identical* density here (0.809), so density cannot be the
explanation for their very different reported accuracies.

**Caveats.** One toy graph, one seed, no fine-tuning — this is the untuned base model, whereas
TAG-DLM's effects appear after LoRA. Perturbation magnitude at a span is a crude proxy for
information flow and does not distinguish task-relevant from irrelevant signal. The fp32/CPU path
may differ slightly from bf16/GPU. None of this affects the layer-1 blocking conclusion, which is
exact, but the depth profile should be repeated post-LoRA before it goes in a paper.

**Consequence.** Adds **E0.2b** to the plan: measure mean path length to the answer position for
each mask family and test whether it predicts accuracy better than density does. This is a
sharper mechanism than the "distribution shift" account in C2, and it is cheaper to test — it
needs no training at all for the structural part.

Also: the earlier version of this script asserted node 1 would be bit-identical at the *final*
layer under `star`, and reported FAIL. The assertion was wrong, not the code. Recorded because the
mistake is the finding.

---

## E0.0d — Migration off titan3, and what a 24 GB card costs the plan
Date: 2026-09-03 · Config: n/a (infrastructure) · Seeds: n/a

Not an experiment. Recorded because §6 and §7 were rewritten on the basis of it, and the numbers
should be challengeable.

**Method.** Inventoried titan3 and `ai-lab2`, benchmarked bf16 matmul on an idle 4090,
built an environment, transferred the backbone, and attempted the step-3 cost measurement that
E0.0c had to skip.

**Findings (MEASURED unless marked).**

1. **titan3 is not available.** All 8 A6000s are held by user `hke3` — eight
   `train_motivation.py` processes, 41.9 GB of 49 GB each, 5+ hours elapsed. This is why E0.0c ran
   on CPU, and it is not a transient condition.
2. **ai-lab2**: 2× RTX 4090 24 GB (23.64 GiB usable per torch), driver 550.144.03, 32 cores,
   251 GB RAM.
3. **4090 at 149.5 TFLOPs bf16** (8192³ matmul, torch 2.5.1+cu121) against **114** for an A6000.
   Per GPU the 4090 is 1.31× faster; in aggregate one 4090 is **6.1× less compute** than titan3's
   eight (149.5 vs 912).
4. GPU0↔GPU1 is `SYS` — PCIe across the NUMA interconnect, no NVLink, no P2P. LoRA data-parallel
   would be fine (only the 168 M adapter params are all-reduced); FSDP/ZeRO-3 sharding of the 8 B
   weights would not.
5. **Storage is the tightest resource.** `/home` is a 3.6 TB volume at 100%, 30 GB free. There is
   no writable space elsewhere: no `/scratch`, `/mnt` and `/opt` are root-owned, and
   `/media/drive2` (7.3 TB, 5.4 TB free) is root-owned. `/tmp` is on the root ext filesystem with
   380 GB free and is writable, so the backbone is staged there.
6. **Environment is a venv, not conda.** conda 24.11.3 on ai-lab2 cannot load its libmamba solver
   (`libarchive.so.20` missing), and the one env with a suitable torch (`mdlm-atat`) is Python
   3.9.25, below this project's 3.10 floor. Built `~/projects/plexus/.venv` on python3.10.12 with
   torch 2.5.1+cu121, transformers 4.57.6, peft 0.20.0. Held at cu121 for parity with (3).
7. LLaDA-8B-Instruct copied titan3 → ai-lab2, **16.04 GB at 117 MB/s**, all snapshot symlinks
   resolving.
8. **The step-3 measurement OOMed, and that is the finding.** Loading the bf16 backbone died with
   19.69 MiB free, because our own jobs (`v-atat/scripts/run_e12b.py` and
   `E1.0_pretrain/train.py --arm latent`) had claimed 18.0 GB of GPU 1 while the environment was
   building. Three such jobs launched within eight minutes. Capacity on this host is not "two free
   GPUs"; it is whatever the HERALD side is not using, and that changes by the minute.

**DERIVED memory budget** (from the config: 32 layers, 32 heads, `d_model` 4096, `mlp_hidden`
12288, vocab 126464, no weight tying → 8.02 B params): bf16 weights 16.0 GB, plus LoRA r=64
all-linear at 168 M trainable with AdamW fp32 moments, fp32 master and bf16 grads at 2.4 GB, gives
**18.4 GB resident of 23.6 GiB**. With gradient checkpointing and micro-batch 1, peak is ~20–21 GB
at S=2048 (fits, ~3 GB headroom) and ~22.5 GB at S=4096 (marginal to OOM), the S=4096 term being
dominated by the `32 × S² × 2 B` score tensor that a dense float bias forces to be materialised.

**Reading.** MEASURED: the migration is forced, not chosen — (1) leaves no alternative on this
timeline.

DERIVED, and the substantive point: **the binding constraint changed character.** On 48 GB A6000s
memory was comfortable and time was the constraint. At 18.4 GB resident on a 24 GB card the
backbone occupies ~78% of the GPU before a single activation, so PLEXUS cannot timeshare — it
needs a 4090 *exclusively* for the 18–37 h of a Cora run. Combined with (8), that makes
inter-project scheduling a hard requirement rather than a courtesy.

DERIVED: at 6.1× less compute, a 3-seed Cora-only Phase 0 plus Phase 1 is 40–83 GPU-days against a
13-week calendar with no harness built. It does not fit without the four levers in §6.

**Caveats.** The memory table is derived from parameter counts, **not measured** — the measurement
is still owed and is blocked on an idle GPU. Do not put it in the paper. The 149.5 TFLOPs figure
is a dense-matmul peak and therefore an upper bound; real LoRA training at micro-batch 1 with a
flash-disabling dense bias will fall well short of it. The 6.1× *ratio* is fairer than either
absolute number, since both sides are peak matmul, but the wall-clock rows in §6 inherit the old
35%-MFU assumption and are optimistic. ogbn-arxiv's exclusion rests on the 61-day estimate as well
as the memory table, and the time argument alone suffices, so the memory uncertainty does not
change that call.

**Consequence.** §6 rewritten (1×4090 budget, derived memory table, wall-clock rescaled, four
levers). §7 rewritten: scope cut to a **Cora-only diagnostic** with PubMed confirmatory at n=1,
Phase 2 and ogbn-arxiv dropped, experiments complete 2026-12-03, and the E0.1 stop rule moved to
the end of week 7. §5 gained rows for GPU contention and volatile storage. README hardware table
and setup instructions updated; the `pyproject.toml` CUDA comment corrected, since it asserted a
driver constraint that is titan3-specific. Still owed: the step-3 measurement on an idle GPU, and
a lab-admin request for space on `/media/drive2`.

---

## E0.0e — Dense-bias cost on a 4090: S=2048 fits, S=4096 does not
Date: 2026-09-03 · Config: `scripts/validate_attention_bias.py` · Seeds: n/a (deterministic)

**Hypothesis.** A dense `(B, 1, S, S)` float bias is affordable at the sequence lengths the plan
needs — 2048 for Cora/PubMed, 4096 for ogbn-arxiv — on a 24 GB card.

**Falsifier.** If peak memory at S=2048 leaves no room for LoRA optimizer states and gradients,
then single-GPU training of the planned configuration is impossible, and the plan needs different
hardware or a quantised backbone — which would in turn threaten E0.1's 1.0-point gate.

**Setup.** LLaDA-8B-Instruct, bf16, ai-lab2 GPU 1 (RTX 4090, 23.64 GiB usable), torch
2.5.1+cu121, transformers 4.57.6, `expandable_segments:True`. Star mask on the same 4-node toy ego
graph as E0.0c. **Inference only** — no optimizer, no gradients. GPU 1 also carried a 590 MB job
of ours at ~26% utilisation throughout, so timings are contaminated upward.

**Result.** Steps 1–2 reproduce E0.0c on GPU in bf16 and **PASS**: the bias is accepted, densities
are unchanged (`none` 1.000, `star` and `ego` both 0.809, `ego_star` 0.880), layer-1 blocking is
exact under both star variants (node 1 first perturbed at layer 2), and the target does see the
edit. Step 3:

| Seq | Bias tensor | Peak GB | ms/fwd |
| --- | --- | --- | --- |
| 512 | 1.0 MB | 16.57 | 221 |
| 2048 | 16.8 MB | **18.59** | 1361 |
| 4096 | 67 MB | **OOM** | — |

**Reading.** MEASURED: **S=4096 does not fit for inference**, let alone training. ogbn-arxiv is
impossible on this hardware and its exclusion is now a measurement, not an estimate.

MEASURED: S=2048 inference peaks at 18.59 GB, leaving 5.05 GiB.

DERIVED: training adds ~2.4 GB of LoRA optimizer states and gradients plus 0.54 GB of checkpointed
layer inputs, and *removes* the ~1.1 GB this script spends retaining `output_hidden_states`, which
projects to **~20.5–21 GB — it fits, with 2–3 GB spare.** Single-GPU LoRA at S=2048 is viable, and
the entire revised plan depends on that one number, so it gets confirmed on the first E0.1 step.

MEASURED, and unwelcome: 1361 ms for a 32.8 TFLOP forward pass is **~16% of the measured
149.5 TFLOPs/s peak**, against the 35% §6 assumed. At S=512 the gap is similar (221 ms against a
55 ms ideal). Both are upper bounds — the fp32 hidden-state download and the co-tenant job inflate
them — but this is the only timing we have, and taken at face value it doubles every wall-clock
estimate in the plan.

Also resolves an open caveat from E0.0c: **the bf16/GPU path agrees with the fp32/CPU path** on the
structural conclusion. Magnitudes differ (`star` final perturbation 1.82e+01 on GPU against
1.43e+01 on CPU) but layer-1 blocking and the one-layer-delay result are identical, so E0.0c's
mechanism finding does not depend on having run in fp32 on CPU.

**Caveats.** Inference only; the training figure is derived and can be wrong in either direction.
Two implementation choices matter at the GB scale and neither is settled: activation-checkpointing
granularity, and whether the LM head is sliced to the single `[M]` position before projecting to
the 126464-vocab logits. The latter is worth ~1 GB at S=2048 and is not yet implemented. Timings
are upper bounds. One toy graph, untuned model, as in E0.0c.

**Consequence.** §6's memory table replaced with these measurements. ogbn-arxiv's exclusion
upgraded from derived to measured. A 16%-MFU column added to the wall-clock table and made the
planning default. Levers 5–7 added to §6: shrink the ladder, tier the seeds, or claim the second
4090 — the last being the only one that costs scheduling rather than evidence. Owed: a clean
forward+backward timing without `output_hidden_states`, on the first E0.1 step.

---

## E0.0f — Storage inventory closed: staying on /tmp
Date: 2026-09-03 · Config: n/a (infrastructure) · Seeds: n/a

Closes the "lab-admin request for `/media/drive2`" item left open by E0.0d.

**Findings (MEASURED).** `/media/scratch` exists but is empty, `root:root drwxr-xr-x`, and
`mkdir` is denied to uid 1011 with no passwordless sudo. More usefully, `stat -c %d` returns
device **66309** for `/`, `/tmp` *and* `/media/scratch` alike — it is a directory on the root
filesystem, not a separate scratch disk, so it offers nothing `/tmp` does not already provide.
`/media/drive1` is likewise empty and root-owned. `/media/drive2` (`/dev/sda1`, 7.3 TB, 5.4 TB
free) remains the only physically separate volume and still requires an admin.

**Reading.** DERIVED: the admin request is not worth making. The Cora-only scope stores one 16 GB
backbone plus LoRA adapters and logs — far inside the 380 GB free on the root filesystem — so the
only thing `/media/drive2` would buy is persistence across reboot, at the price of a 2-minute
re-transfer avoided.

**Cleanup policy, MEASURED.** The host's rule is `D /tmp 1777 root root -`. The trailing `-` is
the age field, so **`systemd-tmpfiles-clean.timer` will not remove our files by age** — the timer
is active but has no age threshold to act on for `/tmp`, and there is no `tmpreaper`/`tmpwatch` in
`cron.daily`. The `D` verb does empty the directory **on boot**. So the exposure is exactly one
event, reboot, and not a silent mid-run deletion. That is a better guarantee than assumed.

**Caveats.** A reboot during a multi-day run still loses the checkpoint and fails the run at
whatever point it next touches disk. Cheap mitigation: have the training entrypoint assert the
snapshot exists before it starts, so the failure is immediate and legible rather than a confusing
load error 30 hours in.

**Consequence.** §5's storage risk row rewritten to record the decision and the reasoning.
E0.0d's outstanding admin-request item is closed, not deferred.

---

## E0.0g — The harness exists, and Cora's provenance is the largest remaining assumption
Date: 2026-09-03 · Config: n/a (infrastructure) · Seeds: n/a

**Hypothesis.** The week 1–3 build is mostly mechanical: load Cora with text, sample ego graphs,
linearise with spans, wrap LoRA, train. The risk is concentrated in the experiment code, not in
the scaffolding.

**Falsifier.** Discovering that the scaffolding embeds a choice that materially changes what E0.1
measures — i.e. that a "mechanical" decision is actually a research decision in disguise.

**Setup.** `plexus/data` (TAG loader, ego sampling, batching), `plexus/prompts` (NC and LP
linearisation), `plexus/models/llada.py` (backbone + LoRA + single-position scoring),
`plexus/train/harness.py`, `plexus/metrics`, `scripts/run_experiment.py`. Cora obtained from
TAPE's Google Drive archive (783 MB, the McCallum corpus with 52 905 extraction files). 54 tests
pass on ai-lab2; ruff clean.

**Result.**

*Dataset, MEASURED.* 2708 nodes, 5278 undirected edges, 7 classes, mean degree 3.90, max degree
168, zero isolated nodes. The edge count matches PyG Planetoid Cora's 10556 directed edges
exactly, so the graph is the canonical one. Label distribution matches the standard counts
(818 Neural_Networks, 426 Probabilistic_Methods, …). Split 1624/542/542.

*Text coverage, MEASURED.* An extraction file is found for 100% of nodes, but only **98.45% have a
title and 88.29% an abstract; 86.74% have both.** So ~12% of Cora papers are represented by a
title alone.

*Throughput, MEASURED.* The stub-backbone training loop runs at ~1300 tok/s on CPU, which says
nothing about the real model but confirms the loop, collation, accumulation and LR schedule work.

**Reading.**

*The falsifier fired.* Three "mechanical" decisions turned out to be research decisions:

1. **The split is the largest single assumption in the project right now, ASSUMED.** TAG-DLM
   reports Cora NC at 90.96. The standard Planetoid public split gives 20 labelled nodes per
   class and strong text models reach the low 80s there, so 90.96 is not reachable on it. It is
   reachable on a dense split, and TAG-DLM says it follows LLaGA's preprocessing — whose released
   Cora files are named `sampled_2_10_*`, matching TAG-DLM's k=2/K=10 default, and which derive
   from the same TAPE source we used. TAPE splits 60/20/20. 2708 × 0.6 = 1624, which is the
   "~1.6k train nodes" §6's compute budget was already written against. So the pieces are
   consistent, but **no split file was ever published and this is inference, not verification.**
   Recorded as `SPLIT_PROVENANCE` in `plexus/data/tags.py` and copied into every run's
   `config.json`, so no result can be read without it. If E0.1 misses its gate, this is suspect #1.
2. **TAPE's own loader has a silent data bug we had to decide not to inherit, MEASURED.**
   `get_raw_text_cora` assumes every extraction file has both a `Title:` and an `Abstract:` line
   and, when one is missing, reuses the *previous paper's* value from the loop variable. Given
   ~12% of papers lack an abstract, that means a meaningful fraction of TAPE-derived Cora prompts
   carry another paper's abstract. We leave the field empty and report coverage instead. This is a
   deliberate divergence from the upstream pipeline and could itself account for a small gap
   against published numbers, in either direction.
3. **Link prediction leaks unless the queried edge is removed, DERIVED then enforced.** If the
   pair (u, v) stays in the sampled subgraph, then `ego` and `ego_star` — the two families the
   whole C2 argument rests on — encode the label directly in the attention pattern. The `ego` mask
   would then look *better* than the star on LP for a reason that has nothing to do with topology
   helping. `sample_ego_graph` takes `exclude_edges`, and `build_lp_example` raises rather than
   building an example where `(0, 1) in edges`.

*Two implementation traps, MEASURED.* `ff_out` names both the per-block down-projection and the
518 M-parameter output head (`weight_tying` is false), so `target_modules="all-linear"` or any
suffix match would have LoRA-tuned the output head and changed the very distribution the answer is
read from. `in_block_linear_names` enumerates exact paths and asserts 32 × 7 = 224 targets, which
also recovers the paper's 168 M trainable figure. Separately, activation checkpointing is LLaDA's
own `set_activation_checkpointing(ActivationCheckpointingStrategy)`, not HuggingFace's
`gradient_checkpointing_enable()`, which would have silently done nothing.

*`HopBias` is now exact, MEASURED.* E0.0b flagged that `node_bias` approximated pairwise distance
as `|h_i − h_j|`. That is worse than "an upper bound": two distinct hop-1 neighbours score
`|1−1| = 0` and land in the **self** bucket, so the bias asserted that two different neighbours of
the target were the same node. Since E1.1 reads the learned per-distance values directly, this
would have moved mass into the wrong bucket and misdescribed what the method learned. Now wired to
exact APSP via `pairwise_hops`, with the approximation retained behind `exact=False` for ablation
and a test asserting the two disagree.

**Caveats.** No number here involves the real backbone. The training peak (~20.5–21 GB) and the
MFU question from E0.0e are both still open, and both are blocked on a free GPU rather than on
code. The prompt template, the per-node token budget (256 target / 64 neighbour), and the
`unassigned_visible` treatment of instruction tokens are all ours and unverified against
TAG-DLM — they are in the outreach email for that reason. PubMed has no loader yet.

**Consequence.** §7 gains a week 1–3 status note recording that the code gate is met but the
end-to-end gate is not. §8 rewritten: securing one 4090 is now action 1, because the smoke run,
E0.1 and everything after it queue behind it. §7's stale "contingent on the storage grant" note
for E0.4 removed, since E0.0f settled that. The outreach email is drafted in
`docs/outreach/tagdlm-code-request.md` with the split question first.
