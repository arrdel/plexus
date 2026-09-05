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

---

## E0.0h — A silent-failure bug caught by a live GPU, and 2048 was twice the sequence we needed
Date: 2026-09-03 · Config: n/a (infrastructure) · Seeds: n/a

**Hypothesis.** With 63 tests passing, including a scaled-down real LLaDA on CPU, the first run on
a real GPU would either work or fail for a reason the tests could not have reached (memory,
throughput, dtype).

**Falsifier.** Failing for a reason the tests *could* have reached — i.e. the CPU test suite was
testing the wrong composition rather than a smaller version of the right one.

**Setup.** `scripts/await_gpu_and_run.sh` polling ai-lab2 for a card with ≥22 GB free over three
consecutive minutes, then launching `run_experiment.py --smoke` on it. Launched 16:07.

**Result.**

*The watcher worked, MEASURED.* GPU 1 freed at 17:11 and was claimed at 17:13, after two prior
false starts were correctly rejected by the three-poll stability rule. Cora loaded, the backbone
loaded, and LoRA attached at **167.8 M trainable on 8.02 B (2.09%)** — matching the paper's 168 M
figure independently of the arithmetic in E0.0g.

*Then it failed in 7 seconds, and the falsifier fired.* `enable_activation_checkpointing` raised
`AttributeError`. The setter lives on `LLaDAModel`, and the unwrap walked a fixed list of
attribute names, `("base_model", "model", "module")`. From a bare `LLaDAModelLM` that reaches it
in two steps, which is what the CPU test exercised. Under peft the chain is
`PeftModel → base_model → model → model` and needs `"model"` twice; the list had already spent it
and offered `"module"`, which does not exist. **The test covered the component; the runner used the
composition.** Fixed by searching the module tree for the outermost class implementing the setter,
which is indifferent to wrapper depth, and confirmed on the real 8 B model: it now reports
`base_model.model.model`.

Two related traps checked while fixing it. `hasattr(module, ...)` is unsafe here because peft's
`LoraModel.__getattr__` forwards unknown attributes to the wrapped model, so an instance check
reports True for wrappers that do not implement the method; the search uses `hasattr(type(m), ...)`.
And LLaDA passes **`use_reentrant=False`** to `torch.utils.checkpoint`, which matters: under
reentrant checkpointing a block whose inputs all have `requires_grad=False` — which is every early
block here, since the embeddings are frozen — produces no gradient at all, so the early adapters
would never train and the run would merely look like it converged badly. A test now asserts every
block receives gradient with checkpointing on.

*Sequence length, MEASURED.* Profiling all 1624 Cora NC training examples at k=2, K=10:

| statistic | tokens |
| --- | --- |
| p50 | 957 |
| p95 | 1384 |
| max | **1583** |
| mean | 902 |

Nodes per example: min 2, median 13, max 21. **No Cora example needs 2048.** The 2048 in §6 was
chosen as the largest length that fits in 24 GB, which is a different question from what the data
requires, and the two got conflated.

Since the micro-batch is 1 there is no second example to align with, so padding to a fixed 2048
buys nothing. Padding to a multiple of 128 instead costs **0.472× on the linear terms and 0.246×
on the quadratic ones**, with mean padded length 966 and max 1664.

**Reading.**

*The efficiency finding is worth more than lever 7 and costs nothing, DERIVED from the measured
distribution.* Cora's 20 epochs fall from 6.6e7 to 3.1e7 tokens. At the measured 16% MFU the
pessimistic Cora run goes from ~40–80 h to ~19–38 h, and the 54-run diagnostic from 90–180
GPU-days to ~43–86 against a 91-day calendar. Lever 7 — claiming the second 4090 — buys ~2× at the
cost of a scheduling fight with HERALD; this buys at least 2.1× at the cost of nothing, because
the padded positions were never carrying information. The quadratic saving is real rather than
notional precisely because the dense float bias defeats flash SDPA, so the S×S score matrix is
materialised per layer.

*Peak memory is now set by 1664, not 2048, DERIVED.* The bias falls from 16.8 MB to 11.1 MB per
sample and the score matrices by the same 0.66×. This has not been confirmed on hardware.

*The general lesson, and it is not about this bug.* Every CPU test here was written against a
component in isolation, and the one path not covered was the one where two correct components are
composed in the order the runner uses. The fix was five lines; finding it cost a free-GPU window
on a machine where windows are the scarce resource. Tests of composition are worth more than
tests of components when the components come from someone else's library.

**Caveats.** The smoke run still has not completed, so the clean forward+backward timing and the
real training peak remain owed, and §6's wall-clock table is deliberately **not** rescaled by the
2.1× — rescaling an estimate by another estimate is how the 35% MFU error happened. The sequence
profile is Cora NC only; LP puts two full-budget target blocks in each example and will be longer,
and PubMed is unmeasured. `pad_to_multiple_of=128` is a judgement call, trading 7% over an exact
fit for ~13 distinct shapes instead of hundreds.

**Consequence.** §6 gains a measured subsection on sequence length and revised lever arithmetic;
the wall-clock table keeps its old numbers with an explicit note that they are pending a clean
timing. The watcher is relaunched with the fix. `run_experiment.py` now builds all examples
*before* loading the backbone, so a CPU-side bug costs seconds rather than a GPU window.

---

## E0.0i — the smoke run completes; densities measured; the graded bias reaches the answer row

**Date:** 2026-09-03 (evening)

**Hypothesis.** The harness runs end-to-end on a real 4090 within the memory envelope derived in
E0.0d, and the graded bias (C3's method) is trainable as implemented.

**Falsifier.** OOM at the padded lengths measured in E0.0h; or no gradient reaching
`HopBias.values`.

**Setup.** GPU watcher fired at 20:49 on GPU 0 and ran `--smoke` to completion in 19 s. Separately,
`scripts/measure_densities.py` over 300 real Cora examples per task (CPU only, no GPU).

**Result.**

1. **MEASURED — the smoke run completed.** Peak allocated **18.96 GB** against 24 GB of device
   memory, comfortably inside the 20.5–21 GB derived in E0.0d. Cora loaded, 8.02 B backbone
   loaded, LoRA attached at 167.8 M (2.09%), activation checkpointing set on
   `base_model.model.model`, forward+backward+eval all succeeded. The week 1–3 gate is met.

2. **MEASURED — the smoke run's timing and accuracy are both unusable, for the same reason.**
   With `effective_batch=128`, `micro_batch=1` and `max_steps=8`, `accum_steps` is 128, so
   **zero optimiser steps completed**. The reported `nc star acc=62.50 ±9.88 (n=24)` is therefore
   a randomly-initialised LoRA on 24 examples, not a result, and `history["steps"]` is empty
   because the logging condition is keyed on optimiser steps. The 5.6 s wall clock is dominated by
   warmup. **Neither number should be quoted.** `scripts/benchmark_step.py` now exists to take the
   throughput measurement properly: it discards a configurable warmup and reports the median.

3. **MEASURED — mask density per family, over 300 real examples per task.** No GPU needed;
   density is a property of the mask and the graph.

   | family | NC ρ | LP ρ |
   |---|---|---|
   | `none` | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
   | `star` | 0.4869 ± 0.1604 | 0.3433 ± 0.1228 |
   | `ego` | 0.4242 ± 0.1389 | 0.2914 ± 0.0786 |
   | `ego_star` | 0.5820 ± 0.1630 | 0.4391 ± 0.1237 |
   | `random_rewire` | 0.5812 ± 0.1643 | 0.4321 ± 0.1270 |
   | `HopBias` | 1.0000 | 1.0000 |

**Reading.**

- **`random_rewire` is density-matched to `ego_star` empirically, to within 0.001 on NC.** That is
  what it was designed for — it starts from the star and adds the same count of *non-target* edges
  as `ego` has — and it is now verified rather than intended. Note for anyone reading the code:
  it is *not* a control for `ego`, and §3 of the paper originally described it as one. Fixed.

- **`star` is DENSER than `ego` (0.487 vs 0.424).** This is a *third* difference between the two
  masks, on top of the two the plan already names (neighbour–neighbour edges, target reach). It
  runs in the direction that favours `star`, so TAG-DLM's 30-point gap has a candidate explanation
  that involves no topology whatsoever. This strengthens C1 more than anything measured so far,
  and it cost no GPU time.

- **The standard deviations are large** (≈0.16 on a mean of 0.49) because density tracks
  neighbourhood size and Cora's degree distribution is heavy-tailed (mean 3.90, max 168). A single
  per-dataset density number would be misleading; report per family with spread.

4. **DERIVED — the graded bias never writes to the answer row.** Found by writing a test that
   gradients reach `HopBias.values`. The answer `[M]` position belongs to no node span (by the
   §2 design choice that keeps the instruction globally visible), so `node_to_token_bias` leaves
   its row unbiased and the graded bias influences the answer only *indirectly*, via the node
   representations. That is the same indirect route a hard mask has, so it is a fair comparison —
   but it discards the one thing only a graded bias can express: a *direct*, learnable "how much
   does a hop-k neighbour count when I answer". Added `HopBias.answer_as_target`, applying to the
   additive term only; a test asserts the hard families are bit-identical under the flag, since
   rewriting span ownership for the *gate* would silently redefine `star` and `ego`.

**Caveats.**

- The initial gradient test failed with an all-zero gradient, which was an artefact of the **test
  double**, not the method: `StubBackbone` read the bias through a `> -1e8` comparison (a step
  function, no gradient) and reduced over one score axis only, so each position read its own
  attention row — including the answer's, the one row a graded bias never writes. Fixed to reduce
  over both axes, which is the faithful stand-in for a 32-layer backbone. **A test double that is
  wrong in the same place the code is subtle will confirm whatever you already believe.**
- Densities are measured at k=2, K=10, pad-to-128, seed 0. They will move with K.
- Still owed: the throughput/MFU measurement (queued as `bench`), and every accuracy.

**Consequence.**

- Queue launched (`scripts/run_queue.sh`, resumable, marker files in `logs/queue/`) with 8 jobs:
  throughput benchmark ×2, then `star` NC 20 epochs (doubles as the convergence curve at
  `--eval-every-epochs 4`), `ego`, `ego_star`, `none`, both graded arms, `random_rewire`.
  As of 23:11 it is waiting; both cards remain held (21198 MiB / 12242 MiB).
- The `star`-denser-than-`ego` finding should be reported in the paper as a measured third
  confound, not as a footnote. It is now Appendix Table 3 and is cited from §3.
- 78 tests pass on ai-lab2 against the real LLaDA remote code (up from 71).

---

## E0.0j — Throughput measured; dynamic padding is worth 2.25× wall clock (2026-09-03, 23:47–23:56)

**Question.** What does a training step actually cost, and was the dynamic-padding decision of
E0.0h worth implementing?

**Method.** `scripts/benchmark_step.py`, steady-state fwd+bwd, warmup discarded, median over the
remainder, at the true padded length distribution. Two runs differing in exactly one setting:
pad-to-128 versus fixed `S = 2048`. One RTX 4090, other card idle. MFU against a separately
measured 149.5 TFLOP/s bf16 matmul peak, not a vendor figure.

| | pad to 128 | fixed S=2048 |
| --- | --- | --- |
| median step | 734 ms | 1825 ms |
| median padded length | 1050 | 2048 |
| throughput | 1294 tok/s | 1122 tok/s |
| effective | 62.3 TFLOP/s | 54.0 TFLOP/s |
| MFU | 41.6 % | 36.1 % |
| peak memory | 20.38 GB | 21.17 GB |
| Cora NC, 20 epochs | **7.32 h** | 16.47 h |

**Findings.**

1. **Dynamic padding is worth 2.25× in wall clock**, against only 1.15× in throughput. The gap is
   the point: shortening sequences removes tokens *and* makes each cheaper, and the removed tokens
   were the quadratically expensive ones. Cora's p95 is 1384 against a 2048 budget, so roughly a
   third of every fixed-length batch was padding being attended over.
2. **MFU *rises* as sequences shorten** (36.1 → 41.6 %), which is backwards from the usual
   direction. The explanation is the bias tensor: `(B,1,S,S)` bf16 is 8.4 MB/sample at S=2048 and
   2.1 MB at S=1024. It is pure memory traffic with no arithmetic attached, so it taxes
   utilisation directly, and halving S quarters it. Both TAG-DLM's formulation and ours pay this;
   neither avoids it while structure enters as a dense additive term.
3. **20.38 GB peak**, above the smoke run's 18.96 GB (longer batches). 20.38 is the number to
   schedule against, and is why the queue waits for 21 GB free.
4. The earlier ~16 % MFU figure was an artefact of `output_hidden_states` retaining all 33 layers
   with a competing job on the second card. Neither it nor the smoke run's wall clock is usable.
   Both are recorded in the paper's Appendix D.3 as failure modes rather than deleted.

**Consequence.**

- At 7.32 h/arm the 9-arm × 3-seed grid is ~200 GPU-h, ~4 days on two cards. Affordable against
  the 2026-12-03 experiment deadline, so the seed sweep is no longer the thing to cut.
- **The queue is now multi-worker.** One worker per card, claiming jobs by `mkdir` of a lock
  directory (atomic; a `-e` test then `touch` is not). Verified: two workers took distinct jobs on
  the same instant and both skipped a pre-claimed one. Queue extended to 22 jobs — seeds 1 and 2
  for every arm, ordered so seed 1 of a decisive arm precedes seed 2 of anything, so a cut leaves
  two seeds on the arms the claims rest on.
- **E0.1 — the reproduction gate — is running for real** as of 23:52 (253 optimiser steps, loss
  0.1051 at step 1, 1247 tok/s, 19.47 GB). This is the first run in the project's history to
  complete an optimiser step outside a unit test.
- Editing a bash script while bash is executing it will corrupt the running process, because bash
  reads the file lazily by byte offset. The in-flight E0.1 was preserved by pre-claiming its lock,
  killing only the parent shell, and letting the orphaned python child continue.

---

## E0.0k — titan3 freed; PubMed loader; the grid now runs on six GPUs (2026-09-04, 07:00–07:50)

**Context.** titan3's eight A6000s, fully booked by another user throughout E0.0d–E0.0j, came
free. The workspace is *on* titan3, so this is local, and the 15 GB LLaDA snapshot was already
in `checkpoints/hf`.

**What had to be true before using it.**

1. **Library parity, not just "it runs".** titan3's base conda python is 3.13 with
   torch 2.11+cu130 against a CUDA 12.1 driver — CUDA 13 is a major-version bump, so
   `torch.cuda.is_available()` was False. The existing `plexus` conda env already had
   torch 2.5.1+cu121, an exact match for ai-lab2, but `transformers` 4.49.0/`peft` 0.14.0 against
   ai-lab2's 4.57.6/0.20.0. Aligned them. **This is not pedantry: with arms split across two
   machines, a version difference between hosts is indistinguishable from an effect of the mask.**
   For a paper whose entire argument is that the baseline attributed a gain to the wrong cause,
   shipping that same error would be self-refuting.
2. Cora had to be copied (1.5 GB, mostly the McCallum extractions). A first pass excluded
   `mccallum/` and three tests failed — the exclusion dropped the titles and abstracts.
3. **78 tests pass on titan3**, identical to ai-lab2.

**Throughput on A6000.** 839 tok/s solo, ~790 under four-way concurrency; 20.4 GB peak.
**MFU was reported as 27.0% and that was wrong** — `benchmark_step.py` hardcoded the 4090's
149.5 TFLOP/s peak. The A6000's own measured bf16 peak is 108.7 TFLOP/s, giving **36.4%**, which
is the figure comparable to the 4090's 41.6%. The script now measures an unknown device's peak
rather than borrowing one. *Dividing by the wrong constant is the same class of error as the
throughput mistakes in E0.0j, and it is the third time in this project an unmeasured constant
produced a plausible wrong number.*

**Now running: six concurrent arms.**

| host | GPU | arm |
| --- | --- | --- |
| titan3 | 0–3 | `star`, `none`, `ego`, `ego_star` — NC seed 0, 20 epochs |
| ai-lab2 | 0–1 | `star`, `ego_star` — NC seed 0, 20 epochs |

The overlap is deliberate. `star` s0 and `ego_star` s0 run on both a 4090 and an A6000, which
**measures the hardware noise floor for free**: no difference between arms smaller than the
same-seed cross-machine gap is interpretable. `collect_results.py` now aggregates by *seed*, not
by run, so replicates do not masquerade as extra seeds and deflate the standard deviation;
`replicate_spread` reports the gap separately and the paper has a macro for it.

Step-1 losses, noisy but suggestive: `none` 0.3325, `ego` 0.1793, `ego_star` 0.1357,
`star` 0.1160. The ordering is the ladder's expected ordering *before a single optimiser step*,
which is consistent with the masks changing task difficulty rather than teaching topology.

**PubMed loader (E-plan item long outstanding).**

- Source: TAPE's `PubMed.zip` (LINQS Pubmed-Diabetes + `pubmed.json` titles/abstracts).
  19717 nodes, 44324 undirected edges after dropping self-loops and duplicates (README claims
  44338 links), 3 classes, text coverage **0.9999**.
- **Text is joined by PMID, not by position.** TAPE reads `pubmed.json` positionally, which is
  correct only while its order matches the NODE file's. A wrong join pairs every node with a
  neighbouring paper's abstract and is invisible in an accuracy number.
- **The label mapping is verified, not assumed.** A permuted mapping raises no exception and ruins
  every number. Probe vocabularies per class give:

  | class | animal | iddm | niddm |
  | --- | --- | --- | --- |
  | Experimental | **0.995** | 0.081 | 0.021 |
  | Type 1 | 0.776 | **0.413** | 0.029 |
  | Type 2 | 0.798 | 0.236 | **0.231** |

  My first version of this check read the matrix *row-wise* and appeared to fail: animal-model
  vocabulary is common to the whole diabetes corpus, so it is the most frequent probe in every
  class. Read down the columns — which class is most *enriched* — each probe lands on its own
  class. **A check that is uninformative is worse than no check, because it looks like evidence.**
- 3 epochs, not 20: PubMed's train split is 11830 against Cora's 1624, so equal epochs would give
  PubMed 7.3× the optimisation budget (90 h/arm) and confound dataset with budget. 3 × 11830 =
  35490 examples against Cora's 20 × 1624 = 32480, within 9%. Queue written, not launched.

**Caveat to check first, recorded before any real PubMed result exists.** A smoke run with **no
completed optimiser step** scored 91.67 (n=24) on PubMed NC, against TAG-DLM's reported 96.30, on
a 3-class problem. If the real `none` arm confirms it, **PubMed NC is close to saturated by text
alone and cannot discriminate between structure encodings** — which would itself be a finding, and
is why `none` is first in the PubMed queue rather than last. 82 tests pass.

---

## E0.0l — An rsync flag destroyed two finished runs (2026-09-04, 07:30–07:45)

Recorded in full because the loss was self-inflicted and the mechanism is easy to repeat.

**What happened.** Code was pushed to ai-lab2 with

    rsync -az --exclude 'data' ... ./ ai-lab2:projects/plexus/

`data` has no leading slash, so it matches **any** directory named `data` at any depth —
including `plexus/data/`, the package holding the dataset loaders. The remote had therefore been
running stale loader code for hours while every sync reported success. It surfaced only when the
new PubMed import failed on the remote test run.

The fix I reached for was `--delete-excluded`, on the belief that it would clean up stale
excluded content. **It does the opposite of what excluding a path means: it deletes the excluded
paths from the destination.** One command removed ai-lab2's `.venv`, `checkpoints/`, `data/`,
`logs/`, `baselines/` and `experiments/`.

**What was lost.** `star` seed 0 and `ego_star` seed 0 had *finished* — eight hours each, at
step 240/253 forty minutes earlier — and their `result.json` files were in `experiments/`. They
were never rsynced back, so they are gone. That is 16 GPU-hours and the first two real accuracy
numbers this project has ever produced.

**What survived, and why.** `HF_HOME=/tmp/adelechinda/hf` and
`PLEXUS_DATA_ROOT=/tmp/adelechinda/plexus_data` live outside the repo, so the 15 GB checkpoint and
the 2.2 GB corpus were untouched; the repo-local copies were redundant. titan3 was unaffected — it
was the rsync *source* — so all four of its arms kept running. Only the venv had to be rebuilt,
and only because it lived inside the repo.

**Why the science is intact.** titan3 is running `star` s0 and `ego_star` s0 itself, so the two
destroyed runs are being reproduced; what was actually lost is the *cross-hardware replicate*, not
a result. That is the one thing in the grid whose loss costs nothing but time.

**Changes made so this cannot recur.**

1. `scripts/sync_hosts.sh` replaces the ad-hoc rsync line. Every exclude is anchored with a
   leading slash, nothing is deleted, and it compares per-file checksums across hosts afterwards
   rather than trusting rsync's exit status. Excluding a path now means "this host owns its copy".
2. The queue exports `PYTHONPATH` instead of assuming `pip install -e .`.
3. **The queue stops after two failures that each take under two minutes.** The rebuilt venv
   lacked the editable install, so the first job failed on import in one second — and the queue
   then consumed all six remaining jobs at a second apiece and marked every one attempted. A fast
   failure is a property of the host, not of the job.
4. `pkill -f run_queue.sh` over ssh killed the ssh session itself, because the pattern matched the
   remote command line. Use `pgrep -f 'run_[q]ueue'`, or kill by PID.

**Rebalanced afterwards.** ai-lab2 was about to re-run `ego` and `none`, which titan3 already has.
It now runs the two graded-bias arms instead, pulling the paper's method result about thirteen
hours earlier. In flight: titan3 0–3 on `star`/`none`/`ego`/`ego_star`, ai-lab2 0–1 on
`HopBias` and `HopBias + answer grading`, all NC seed 0, 20 epochs.

**Two observations worth keeping.**

- Both graded arms open at loss **0.3311** against `none`'s **0.3325** on the other host. A
  zero-initialised graded bias is supposed to *be* the unmodified model, and at step 1 it measures
  as such. The residual 0.0014 is a 4090-versus-A6000 difference, i.e. a first read on the noise
  floor.
- Before it was killed, the ai-lab2 `star` run reached **loss 0.0000 by epoch 18**. Twenty epochs
  on 1624 examples with a 167.8 M-parameter adapter memorises the training set completely. The
  `--eval-every-epochs 4` curve exists precisely to show where test accuracy peaked, and the epoch
  budget should be re-set from it rather than from the plan's guess.

---

## E0.0m — The zero-shot ladder, and a hypothesis of mine that it refuted (2026-09-04, 08:55–10:00)

**Origin.** The step-1 training loss of a run is the loss of the **frozen** model under that mask,
because a LoRA at initialisation has no effect. Reading it off the in-flight logs cost nothing:

|  | none | star | ego | ego_star |
| --- | --- | --- | --- | --- |
| Cora | 0.3325 | **0.1160** | 0.1793 | 0.1357 |
| PubMed | 1.2277 | **0.3538** | 0.7905 | 0.4415 |

`star` cuts the pretrained model's loss by 65% on Cora and 71% on PubMed with no learning, the
ordering `none > ego > ego_star > star` holds on both datasets, and `ego_star` sits most of the way
from `ego` to `star`. I inferred that much of the gain the baseline attributes to topology needs no
training, and that target reach explains the ego collapse. Both would have been strong claims.

**The measurement refuted it.** `--no-lora` evaluates the frozen backbone with no optimiser step,
at ~3.5 min/arm. Cora NC, n=300:

| family | seed 0 | seed 1 | mean |
| --- | --- | --- | --- |
| none | 68.00 | 76.00 | **72.00 ± 5.66** |
| star | 66.33 | 71.00 | 68.67 ± 3.30 |
| ego | 65.67 | 74.33 | 70.00 ± 6.13 |
| ego_star | 65.67 | 73.00 | 69.33 ± 5.19 |
| random_rewire | 66.00 | — | 66.00 |

PubMed NC, n=300, seed 0: none **89.67**, star 89.67, ego 87.67, ego_star 90.33, rewire 89.67.

**The ordering is inverted.** `none` has the *highest* loss and the *highest* accuracy. A mask can
lower cross-entropy at the answer position — restricting context makes the model more confident —
without moving the `argmax` over the class names, and the loss sees only the confidence.
**Loss under a mask is not evidence about accuracy under that mask.** I had already written the
inference into the queue file's rationale before running it; the file now records the refutation
instead, because a plausible unrun inference is exactly the kind of thing that ends up in a paper.

**Three findings that do survive, all of them useful.**

1. **No mask beats full attention before training.** So whatever a binary topology mask contributes
   in the baseline is contributed *during optimisation*; it is not that the frozen model reads a
   masked context better. This closes off an alternative explanation for the trained ladder before
   the trained ladder has even landed.
2. **PubMed NC is close to saturated by text alone: 89.67 with no graph whatsoever**, against
   TAG-DLM's reported 96.30. At most ~6 points are available for *any* structural mechanism to
   explain, and the no-structure arm that would establish this is not reported there. Confirms the
   E0.0k caution at n=300 rather than n=24. **A benchmark with six points of headroom cannot
   discriminate between ways of encoding structure.**
3. **The seed effect exceeds the effect under study.** The no-structure arm alone moves **8.00
   points** across two seeds on Cora, against a largest between-family gap of ~3, under one
   protocol. Single-seed comparisons between families on Cora are uninterpretable. The plan's
   three-seed rule is arithmetic, not caution.

**Cost.** ~3.5 min/arm, run on GPU 7 *alongside* a training job — forward-only eval peaks ~17 GB,
and 20.9 + 17 fits in the A6000's 49 GB. The whole seed-0 block cost under 40 minutes of shared
time. **This should have been the first experiment in the project, not the fifteenth.** It needs no
training, it bounds what every other number can possibly mean, and its absence from the baseline is
the single clearest gap in it.

**In the paper.** New §6.1 with \cref{tab:zeroshot}, generated from artefacts like every other
table. Main body is 9 pages again after tightening; appendix 12.

**Also now running:** all 8 titan3 cards plus 2 on ai-lab2 — Cora ladder ×4 (GPU 0–3), PubMed
ladder ×4 (GPU 4–7), graded-bias ×2 (ai-lab2), zero-shot sweep sharing GPU 7.

## E0.0n — Node classification does not discriminate, so the grid moves to link prediction (2026-09-04, 10:00–11:00)

**The epoch-4 Cora NC ladder, full 542-example test set, seed 0.** All six arms:

| arm | accuracy | mask density |
| --- | --- | --- |
| `ego_star` (ours) | 86.72 ± 1.46 | 0.600 |
| `HopBias` + answer grading (ours) | 86.53 ± 1.47 | 1.000 |
| `none` — no structure at all | 86.35 ± 1.47 | 1.000 |
| `star` — the baseline's default | 86.35 ± 1.47 | 0.509 |
| `HopBias` (ours) | 85.98 ± 1.49 | 1.000 |
| `ego` — real topology | 85.61 ± 1.51 | 0.393 |

**Spread 1.11 points, against the 8.00-point seed spread measured in E0.0m.** Training loss on the
graded arms is already 0.0004, so this is convergence and not an early snapshot. **Nothing
discriminates on this task — including our own method.** Our `star` is 86.35 against the reported
90.96, so E0.1 fails the 1.0-point reproduction gate as it stands.

This is claim C1 confirmed about as emphatically as it can be: on Cora node classification, *how*
the graph is described to the model makes no measurable difference. It is a real diagnostic result.
It is **not** evidence that our method beats the baseline, because here nothing beats anything.

**The gap this exposed, which had been sitting in plain sight.** TAG-DLM's dramatic number — the one
this entire project is built around — is **link prediction**: 91.62 with the star mask collapsing to
61.76 with real ego topology, a 30-point effect. Their paper reports no comparable NC ablation.
**Our grid was entirely node classification.** We had been trying to explain a 30-point LP
phenomenon using a task whose total between-arm spread is 1.11 points.

The measured densities said so in advance. On LP the two families separate much further than on NC:

| task | `none` | `star` | `ego` | `ego_star` | `random_rewire` |
| --- | --- | --- | --- | --- | --- |
| NC | 1.000 | 0.487 | 0.424 | 0.582 | 0.581 |
| LP | 1.000 | 0.343 | 0.291 | 0.439 | 0.432 |

The LP prompt carries **two** endpoints rather than one target, so a target-centred star abstraction
has strictly more to abstract away, and each family's departure from full attention is larger.

**What was stopped, and what it cost.** The PubMed NC grid on cards 4–7 was killed mid-flight at
step 20/277 — still inside epoch 0, so **no evaluation had landed and no result was lost**. Its
partial output directories and queue locks were cleared, leaving it resumable. The justification is
E0.0m finding (2): PubMed NC is at 89.67 with no graph at all against a reported 96.30, so it is the
least informative grid we own. Cards 0–3 (Cora NC) were left alone and will run the **full 20
epochs** rather than stopping at the convergence elbow, so the paper can show the whole curve.

**Launched:** `scripts/queue_lp.txt`, 21 jobs = 7 arms × 3 seeds, `--task lp --dataset cora
--epochs 6 --eval-every-epochs 2`, four workers on cards 4–7. 3912 train pairs / 452 test, 183 steps
per epoch, padded tokens min/mean/max 152/1356/1845 so S=2048 holds with room. ~750 tok/s per card,
19.75 GB peak. 6 epochs = 23,472 examples, within 9% of the Cora NC 20-epoch budget of 32,480 — the
comparison is budget-matched, not epoch-matched.

**These are also the first runs to store per-example predictions** (commit `2b5e0d1`), so
`scripts/compare_decisions.py` and its McNemar test become usable for the first time. That matters
here more than anywhere: if the LP arms land close together the way the NC arms did, accuracy alone
cannot distinguish "the mask changed nothing" from "the mask traded errors", and on LP we would very
much like to know which.

**Step-1 losses, recorded with the E0.0m caveat attached.** Frozen-model loss under each LP mask:
`none` 1.0622, `star` 0.3855, `ego` 0.6433, `ego_star` 0.8747. The `none`-to-`star` ratio is 2.76×,
close to NC's 2.87×, but `ego_star` and `ego` have **swapped** relative to NC. **No inference is
drawn from this.** E0.0m established that loss ordering under a mask does not predict accuracy
ordering under it, and the whole point of recording that lesson was to not spend it.

## E0.0o — ai-lab2 released; the first two completed 20-epoch runs (2026-09-04, 16:45–17:00)

**Why.** The 4090s were needed elsewhere. Freeing them was a scheduling request, not a scientific
decision, but it forced the first proper harvest of that host.

**What was already finished.** Both graded arms had run to completion, 20 epochs, 5 evaluations
each, roughly 7 hours per arm:

| arm | accuracy (Cora NC, seed 0, n=542) |
| --- | --- |
| \textsc{HopBias} + answer grading (ours) | **89.48** |
| \textsc{HopBias} (ours) | 87.64 |

**These are the first completed training runs in the project.** Every other number to date has come
from a mid-flight evaluation. They are also the first cells in `tab:ladder-nc` that are not
`\pending`.

**The internal comparison is the interesting part, and it is clean.** Same seed, same 20-epoch
budget, same harness, one flag apart: grading the answer row is worth **+1.84 points**. That is the
gap §5.3 predicted from the design fact that `[M]` belongs to no node span and is therefore exempt
from the additive term under the standard lift. The prediction was made before the runs and the
measurement agrees with it. It is one seed, so it is a signal rather than a result, but it is the
first evidence that the answer-position treatment is load-bearing rather than a detail.

**What it does not show.** 89.48 is currently the highest completed number in the project, above
`star`'s 89.30 at epoch 12 on titan3. That comparison is **not** valid: `star` has not finished its
20 epochs, and E0.0n showed the arms still moving between epochs 8 and 12. No claim is made until
the mask families complete.

**Procedure, given E0.0l.** Artefacts were pulled *before* anything was killed, then verified by
md5 against the source (8 of 8 files identical) and only then were the processes stopped. Pulled to
`experiments/ailab2/` rather than merged into `experiments/`, because `collect_results.py` is
recursive and a separate subtree keeps the provenance of each host visible. Workers were killed
before their children so the queue could not launch a replacement job.

**Cost of stopping.** Two runs lost mid-flight, both at epoch 4 with one evaluation each:
`e02_rewire_nc_s0` and `e11_hopbias_nc_s1`. Their queue locks were cleared so both remain runnable.
The seed-1 graded arm is the more valuable of the two, since the +1.84 point answer-grading gap
currently rests on a single seed and E0.0m measured an 8.00-point seed spread on this benchmark.
There is no spare capacity to re-run it: all 8 titan3 cards are busy with the Cora NC ladder
(0--3) and the LP grid (4--7).

**Neither run stored per-example predictions**, since both predate commit `2b5e0d1`, so the
McNemar comparison cannot be applied to the +1.84 point gap. The LP grid remains the first place it
will be usable.

---

## E0.0p: the LP grid lands, and the +1.84 point answer-grading gap evaporates

**Date.** 2026-09-05, submission day.

### The measurement that matters

Cora link prediction, seed 0, n=452, 6 epochs, all four hard families complete:

| arm | acc. | vs `ego` |
|---|---|---|
| `ego_star` (ours) | **90.49** | +18.14 |
| `none` | 89.82 | +17.47 |
| `star` | 88.27 | +15.93 |
| `ego` | 72.35 | -- |

The ego collapse reproduces: `star` -> `ego` costs 15.93 points, against the 29.86 reported. But
`ego_star`, which keeps **every** neighbour-to-neighbour edge and restores only the target's
reach, recovers 18.14 of them and finishes 2.21 points **above** `star`. Real topology does not
hurt. Losing the target's reach hurts, and the ego mask loses it as a side effect.

This is claim C2, measured, and the prediction preceded the measurement: `ego_star` was designed
and its three possible outcomes written down in the paper before any LP run existed.

`none` also beats `star` by 1.55, so on this harness the baseline's mask does not pay for itself.
`ego_star` over `none` is only +0.66, below the noise floor, so no claim is made from it.

### The correction

E0.0o reported that grading the answer row was worth **+1.84 points**, from `HopBias` 87.64 and
`HopBias`+answer 89.48, both on ai-lab2. Both configurations had *also* been queued on titan3, and
those runs finished overnight: `HopBias` **88.75**, `HopBias`+answer **88.19**. Same seed, same
config, same code, different card.

| arm | ai-lab2 | titan3 | mean | spread |
|---|---|---|---|---|
| `HopBias` | 87.64 | 88.75 | 88.20 | 1.11 |
| `HopBias` + answer | 89.48 | 88.19 | 88.84 | 1.29 |

Averaged over both hosts the answer-grading gap is **+0.65**, and the replicate spread within a
single configuration is 1.11 to 1.29. **The sign of the effect flips depending on which card you
read.** The +1.84 was within-host noise, reported because only one host had finished.

The accidental by-product is the most useful number in the project: a **measured noise floor of
1.29 points**, from configurations that differ in nothing but floating-point accumulation order.
It is now the paper's unit of resolution, and it retires the entire trained Cora NC ladder, which
spans 1.48 points top to bottom with the density-matched *random* control on top.

**The lesson, which is E0.0m's lesson again.** A single-host, single-seed difference is not a
result, even when it confirms a prediction the paper made in advance. The answer-grading mechanism
may well be real; NC cannot see it, and the LP graded arms are still running.

### Status at submission

Main paper converted to NeurIPS 2026 workshop style (DiffuLM), single column, eight pages of body
plus references and a thirteen-page appendix. Four LP cells remain `\pending`: `random_rewire`,
both graded arms, and seed 1 of `none`. At epoch 3 of 6 the graded arm reads 90.49 and
`random_rewire` 87.83, but E0.0n is the standing reason not to quote a mid-run evaluation.
