"""Validate that PLEXUS topology biases actually control LLaDA-8B's attention.

Three things get checked, in increasing strength:

1. **It runs.** A custom ``(B, 1, S, S)`` float bias is accepted and changes the logits.
2. **It blocks.** This is the one that matters. We plant distinctive text in one neighbour span
   and then *edit that span*. Under the star mask, a different neighbour must not be able to see
   the edit, so its hidden states must be bit-identical; the target, which can see it, must
   change. If this fails, every mask-family experiment downstream is measuring nothing, and no
   amount of accuracy improvement would mean what we think it means.
3. **It is affordable.** Peak memory and latency at S=2048 and S=4096, because an explicit
   attention mask disables the flash SDPA backend and the bias itself is quadratic.

Run: python scripts/validate_attention_bias.py
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModel, AutoTokenizer

from plexus.masks import LinearisedGraph, build_attention_bias, describe

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


def build_toy_sequence(tokenizer, seq_len: int):
    """A prompt prefix plus four node spans, with real token boundaries recorded."""
    prefix = "Classify the target paper. Options: 0) theory 1) systems. Answer: "
    texts = [
        "Target paper: a study of spectral methods for graph partitioning.",
        "Neighbour A: convergence bounds for iterative eigensolvers.",
        "Neighbour B: distributed scheduling in datacenter networks.",
        "Neighbour C: cache-oblivious algorithms for sparse matrices.",
    ]

    ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    spans = []
    for t in texts:
        tid = tokenizer(t, add_special_tokens=False)["input_ids"]
        spans.append((len(ids), len(ids) + len(tid)))
        ids.extend(tid)

    pad_from = len(ids)
    if len(ids) > seq_len:
        raise ValueError(f"toy sequence ({len(ids)}) longer than seq_len ({seq_len})")
    ids = ids + [tokenizer.pad_token_id or 0] * (seq_len - len(ids))

    g = LinearisedGraph(
        spans=spans,
        seq_len=seq_len,
        edges={(0, 1), (0, 2), (1, 3)},
        pad_from=pad_from,
    )
    return torch.tensor(ids)[None, :], g, texts


def edit_span(tokenizer, ids: torch.Tensor, g: LinearisedGraph, node: int, new_text: str):
    """Overwrite one node's span with different text, padded/truncated to the same length."""
    s, e = g.spans[node]
    new = tokenizer(new_text, add_special_tokens=False)["input_ids"]
    new = (new + [tokenizer.pad_token_id or 0] * (e - s))[: e - s]
    out = ids.clone()
    out[0, s:e] = torch.tensor(new)
    return out


@torch.no_grad()
def hidden_states(model, ids, bias):
    """Return (all per-layer hidden states, logits). Index 0 is the embedding output."""
    out = model(input_ids=ids.to(model.device),
                attention_bias=bias.to(model.device) if bias is not None else None,
                output_hidden_states=True)
    hs = [h.float().cpu() for h in out.hidden_states]
    return hs, out.logits.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 4096])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    on_cuda = args.device.startswith("cuda")
    # fp32 on CPU: torch's bf16 CPU kernels are far slower, and the blocking test in step 2
    # needs exact comparisons anyway, which bf16 rounding would make harder to interpret.
    dtype = torch.bfloat16 if on_cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"loading {MODEL_ID} on {args.device} ({dtype}) ...")
    # transformers 4.x spells this `torch_dtype`; 5.x renamed it to `dtype` and broke LLaDA's
    # remote code in other ways, so this repo is pinned to 4.49 (see pyproject).
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=dtype
    ).to(args.device).eval()
    print(f"loaded. params={sum(p.numel() for p in model.parameters())/1e9:.2f}B\n")

    # ---------------------------------------------------------------- 1. it runs
    ids, g, texts = build_toy_sequence(tok, seq_len=512)
    families = ["none", "star", "ego", "ego_star"]
    ref = None
    print("=== 1. bias is accepted and changes the output ===")
    for fam in families:
        bias = build_attention_bias([g], family=fam, dtype=torch.float32)
        _, logits = hidden_states(model, ids, bias)
        stats = describe(bias[0, 0])
        tag = ""
        if ref is None:
            ref = logits
        else:
            tag = f"  max|Δlogit| vs none = {(logits - ref).abs().max():.4f}"
        print(f"  {fam:14s} density={stats['density']:.3f} "
              f"mean_row_vis={stats['mean_row_visibility']:.0f}{tag}")

    # ------------------------------------------- 2. depth-resolved information flow
    #
    # A mask constrains attention *within a layer*. TAG-DLM's Prop. 1 is explicitly a
    # single-layer statement. But LLaDA has 32 layers, and a star mask does not prevent
    # neighbour-to-neighbour communication across depth -- it routes it through the target and
    # delays it by one layer. Asserting bit-identity at the final layer would therefore be
    # asserting something false.
    #
    # So we measure *when* information arrives instead: edit node 3, and track the perturbation
    # at node 1 as a function of depth. Layer 1 is the well-defined blocking claim; everything
    # after is the routing behaviour, which is the thing TAG-DLM's analysis does not cover.
    print("\n=== 2. depth-resolved information flow ===")
    print("    edit node 3's text, then watch the perturbation reach node 1 (not adjacent to it)")
    ids_b = edit_span(tok, ids, g, 3, "Neighbour C: an unrelated treatise on medieval poetry.")
    s1, e1 = g.spans[1]
    st, et = g.spans[0]

    configs = [
        ("none", "none", True),
        ("star", "star", True),
        ("star-strict", "star", False),  # also gate the prompt prefix
    ]
    profiles = {}
    for label, fam, unassigned in configs:
        bias = build_attention_bias([g], family=fam, dtype=torch.float32,
                                    unassigned_visible=unassigned)
        ha, _ = hidden_states(model, ids, bias)
        hb, _ = hidden_states(model, ids_b, bias)
        d_n1 = [(a[0, s1:e1] - b[0, s1:e1]).abs().max().item() for a, b in zip(ha, hb, strict=True)]
        d_tgt = [(a[0, st:et] - b[0, st:et]).abs().max().item() for a, b in zip(ha, hb, strict=True)]
        profiles[label] = (d_n1, d_tgt)
        first = next((i for i, v in enumerate(d_n1) if v > 1e-6), None)
        print(f"  {label:12s} node1 first perturbed at layer {first}"
              f"   L1={d_n1[1]:.2e}  L4={d_n1[min(4,len(d_n1)-1)]:.2e}  "
              f"final={d_n1[-1]:.2e}  target_final={d_tgt[-1]:.2e}")

    ok = True
    d_n1_star, d_tgt_star = profiles["star"]
    d_n1_strict, _ = profiles["star-strict"]

    if d_n1_star[1] <= 1e-6:
        print("  layer-1 blocking holds for `star` with a visible prompt.")
    else:
        print("  NOTE: node 1 is perturbed already at layer 1 with a visible prompt.")
        print("        The globally-visible prompt prefix is acting as a one-hop relay, so the")
        print("        realised communication graph is denser than the star that was specified.")
    if d_n1_strict[1] > 1e-6:
        print("  FAIL: even with the prompt gated, node 1 moves at layer 1 -- mask not applied.")
        ok = False
    else:
        print("  layer-1 blocking holds exactly for `star-strict`: the bias is being applied.")
    if d_tgt_star[-1] <= 1e-6:
        print("  FAIL: the target never sees the edit; it is not reading node 3 at all.")
        ok = False
    print(f"  => {'PASS' if ok else 'FAIL'}: the bias reaches LLaDA's attention as specified.")

    # ------------------------------------------------------------- 3. it is affordable
    if not on_cuda:
        print("\n=== 3. cost of a dense bias: SKIPPED (needs a free GPU) ===")
        print("\ndone.")
        return

    print("\n=== 3. cost of a dense bias ===")
    print(f"  {'seq':>6} {'bias MB':>9} {'peak GB':>9} {'ms/fwd':>8}")
    for S in args.seq_lens:
        try:
            ids_s, g_s, _ = build_toy_sequence(tok, seq_len=S)
            bias = build_attention_bias([g_s], family="star", dtype=torch.float32)
            torch.cuda.reset_peak_memory_stats(args.device)
            for _ in range(2):
                hidden_states(model, ids_s, bias)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                hidden_states(model, ids_s, bias)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 5 * 1000
            peak = torch.cuda.max_memory_allocated(args.device) / 1e9
            print(f"  {S:>6} {bias.numel()*4/1e6:>9.1f} {peak:>9.2f} {ms:>8.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {S:>6} {'OOM':>9} {'-':>9} {'-':>8}")
            torch.cuda.empty_cache()

    print("\ndone.")


if __name__ == "__main__":
    main()
