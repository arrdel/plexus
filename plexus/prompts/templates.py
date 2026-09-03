"""Linearising a sampled ego graph into one token sequence, with exact span bookkeeping.

The output contract is `LinearisedExample`: token ids, a `LinearisedGraph` whose `spans` index
those ids, and the single answer position. Everything the mask families do depends on the spans
being exactly right, so `tests/test_prompts.py` decodes each span back to text and checks it.

Design choices that TAG-DLM does not specify, recorded because they are ours and could matter:

- **The whole node block is inside the span**, including its `Node i:` header. A header left
  outside would be globally visible under every mask, so a gated-out node would still announce
  its existence to the target. Gating should hide the node, not just its content.
- **The instruction prefix and the answer suffix are outside all spans**, hence globally visible.
  They carry the task, not graph content; gating them would confound "structure hurts" with "the
  prompt became unreadable". This matches `unassigned_visible=True` in `plexus.masks`.
- **Answers are single digit tokens** at one `[M]` position, which is what makes TAG-DLM's
  single-forward-pass scoring work. `candidate_token_ids` asserts the digits really are one token
  each in LLaDA's vocabulary rather than assuming it.
- **Truncation is per-node and neighbours are dropped highest-hop-first.** Dropping is recorded
  in `meta` so a run can report how often it happened; a silently truncated neighbourhood would
  quietly weaken every mask family that depends on hop-2 nodes existing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from plexus.data.sampling import EgoSample
from plexus.data.tags import TAG
from plexus.masks import LinearisedGraph

NC_INSTRUCTION = (
    "You are given a paper from a citation network, together with some of its neighbours.\n"
    "Classify the target paper into one of the following categories.\n"
)
LP_INSTRUCTION = (
    "You are given two papers from a citation network, together with some of their neighbours.\n"
    "Decide whether the two target papers cite each other.\n"
)
LP_OPTIONS = ["no", "yes"]


@dataclass
class LinearisedExample:
    """One training or evaluation example."""

    input_ids: torch.Tensor          # (S,) long, padded to seq_len
    graph: LinearisedGraph
    answer_pos: int
    candidate_ids: list[int]         # vocabulary ids of the answer options, in label order
    label: int
    meta: dict[str, Any] = field(default_factory=dict)


def candidate_token_ids(tokenizer, n_options: int) -> list[int]:
    """Vocabulary ids for the digits `0..n_options-1`, verified to be one token each.

    TAG-DLM scores a single masked position by the log-likelihood of a digit token. If a digit
    tokenises to more than one piece that scoring rule is not well defined, so this fails loudly
    rather than silently comparing first-pieces.
    """
    ids = []
    for i in range(n_options):
        pieces = tokenizer(str(i), add_special_tokens=False)["input_ids"]
        if len(pieces) != 1:
            raise ValueError(
                f"option digit {i!r} tokenises to {len(pieces)} tokens ({pieces}); "
                "single-token answers are required for single-position scoring"
            )
        ids.append(pieces[0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"option digits collide in vocabulary: {ids}")
    return ids


def _encode(tokenizer, text: str, max_tokens: int | None = None) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids if max_tokens is None else ids[:max_tokens]


def _assemble(
    tokenizer,
    prefix: str,
    blocks: list[str],
    suffix: str,
    *,
    seq_len: int,
    mask_token_id: int,
    pad_token_id: int,
    block_budgets: list[int],
    n_required: int,
    pad_to_multiple_of: int = 1,
) -> tuple[torch.Tensor, list[tuple[int, int]], int, int, int]:
    """Lay out prefix + node blocks + suffix + [M], returning ids, spans and the answer position.

    Returns `(ids, spans, answer_pos, pad_from, n_dropped)`.

    Two invariants, both of which cost correctness if broken:

    - **The kept blocks are a prefix of `blocks`.** As soon as one does not fit we stop, rather
      than skipping it and trying the next. Skipping would leave `spans[i]` pointing at node
      *j > i* while `hops` and `edges` still described node *i*, so every mask would gate the
      wrong span -- silently, and with plausible-looking accuracy. Since callers order blocks by
      hop, stopping early is exactly "drop the most distant context first".
    - **The first `n_required` blocks are never dropped.** For node classification that is the
      target; for link prediction, both endpoints. An example without its target is not a harder
      example, it is a corrupt one, so this raises instead.
    """
    prefix_ids = _encode(tokenizer, prefix)
    suffix_ids = _encode(tokenizer, suffix)
    # +1 for the [M] token itself.
    reserved = len(prefix_ids) + len(suffix_ids) + 1
    if reserved >= seq_len:
        raise ValueError(f"prefix+suffix ({reserved}) does not fit in seq_len {seq_len}")

    ids = list(prefix_ids)
    spans: list[tuple[int, int]] = []
    n_dropped = 0
    for i, (block, budget) in enumerate(zip(blocks, block_budgets, strict=True)):
        block_ids = _encode(tokenizer, block, budget)
        if len(ids) + len(block_ids) + len(suffix_ids) + 1 > seq_len:
            if i < n_required:
                raise ValueError(
                    f"required block {i} ({len(block_ids)} tokens) does not fit in seq_len "
                    f"{seq_len} with {reserved} reserved for prompt scaffolding; raise seq_len "
                    f"or lower max_target_tokens"
                )
            n_dropped = len(blocks) - i
            break
        spans.append((len(ids), len(ids) + len(block_ids)))
        ids.extend(block_ids)

    ids.extend(suffix_ids)
    answer_pos = len(ids)
    ids.append(mask_token_id)
    pad_from = len(ids)

    # `seq_len` is the truncation *budget*, not the tensor length. Padding every example out to
    # the budget would be pure waste: the micro-batch is 1 (24 GB card), so there is no other
    # example to align with, and the dense float bias defeats flash attention, meaning the
    # S x S score matrix is materialised per layer. Cora's mean example is 902 tokens against a
    # 2048 budget, so padding to the budget costs ~2.3x on the linear terms and ~5x on the
    # quadratic ones. Rounding to a multiple instead keeps the number of distinct shapes small,
    # which matters for allocator churn, at a cost of at most `pad_to_multiple_of - 1` tokens.
    m = max(1, pad_to_multiple_of)
    total = min(seq_len, ((pad_from + m - 1) // m) * m)
    ids.extend([pad_token_id] * (total - len(ids)))
    return torch.tensor(ids, dtype=torch.long), spans, answer_pos, pad_from, n_dropped


def _node_block(local: int, text: str, role: str) -> str:
    body = text.strip() or "(no text available)"
    return f"\n{role} [node {local}]:\n{body}\n"


def _drop_to_kept(sample: EgoSample, n_kept: int) -> tuple[list[int], set[tuple[int, int]]]:
    """Restrict a sample's hops and edges to the first `n_kept` local nodes."""
    hops = sample.hops[:n_kept]
    edges = {(u, v) for u, v in sample.edges if u < n_kept and v < n_kept}
    return hops, edges


def build_nc_example(
    tag: TAG,
    sample: EgoSample,
    tokenizer,
    *,
    seq_len: int = 2048,
    mask_token_id: int = 126336,
    max_target_tokens: int = 256,
    max_neighbour_tokens: int = 64,
    pad_to_multiple_of: int = 128,
) -> LinearisedExample:
    """Node classification: predict the target's class at a single `[M]`.

    `seq_len` caps the example; the returned tensor is only as long as the content needs, rounded
    up to `pad_to_multiple_of`. See `_assemble` for why that is not merely a micro-optimisation.
    """
    options = "\n".join(
        f"{i}) {name}" for i, name in enumerate(tag.class_prompt_names)
    )
    prefix = f"{NC_INSTRUCTION}Categories:\n{options}\n"
    suffix = "\nThe target paper's category is: "

    # Neighbours are already in BFS order, so appending in order puts the highest hop last and
    # makes the drop policy "drop the most distant context first".
    blocks = [
        _node_block(i, tag.text[g], "Target paper" if i == 0 else "Neighbour paper")
        for i, g in enumerate(sample.node_ids)
    ]
    budgets = [max_target_tokens] + [max_neighbour_tokens] * (len(blocks) - 1)

    ids, spans, answer_pos, pad_from, n_dropped = _assemble(
        tokenizer, prefix, blocks, suffix,
        seq_len=seq_len, mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id or 0, block_budgets=budgets,
        n_required=1, pad_to_multiple_of=pad_to_multiple_of,
    )
    hops, edges = _drop_to_kept(sample, len(spans))
    graph = LinearisedGraph(
        spans=spans, seq_len=int(ids.numel()), edges=edges, hops=hops,
        node_ids=sample.node_ids[: len(spans)], pad_from=pad_from, answer_pos=answer_pos,
    )
    return LinearisedExample(
        input_ids=ids,
        graph=graph,
        answer_pos=answer_pos,
        candidate_ids=candidate_token_ids(tokenizer, tag.num_classes),
        label=int(tag.labels[sample.node_ids[0]]),
        meta={"task": "nc", "center": sample.node_ids[0], "n_nodes": len(spans),
              "n_dropped": n_dropped, "n_tokens": pad_from,
              "padded_len": int(ids.numel()),
              "pad_token_id": tokenizer.pad_token_id or 0},
    )


def build_lp_example(
    tag: TAG,
    sample: EgoSample,
    tokenizer,
    label: int,
    *,
    seq_len: int = 2048,
    mask_token_id: int = 126336,
    max_target_tokens: int = 256,
    max_neighbour_tokens: int = 64,
    pad_to_multiple_of: int = 128,
) -> LinearisedExample:
    """Link prediction: local nodes 0 and 1 are the queried pair.

    The caller must have sampled with `exclude_edges={(u, v)}` so the queried edge is absent from
    `sample.edges`; otherwise the topology masks encode the answer. `sample_lp_examples` and
    `sample_ego_graph` are written to make that the natural usage, and the assert here refuses to
    build an example that would leak.
    """
    if sample.n_nodes < 2:
        raise ValueError("link prediction needs both endpoints in the sample")
    if (0, 1) in sample.edges:
        raise ValueError(
            "the queried pair is present in sample.edges -- re-sample with "
            "exclude_edges={(u, v)}, or the ego/ego_star masks reveal the label"
        )

    options = "\n".join(f"{i}) {name}" for i, name in enumerate(LP_OPTIONS))
    prefix = f"{LP_INSTRUCTION}Options:\n{options}\n"
    suffix = "\nDo the two target papers cite each other? "

    roles = ["First target paper", "Second target paper"]
    blocks = [
        _node_block(i, tag.text[g], roles[i] if i < 2 else "Neighbour paper")
        for i, g in enumerate(sample.node_ids)
    ]
    budgets = [max_target_tokens] * 2 + [max_neighbour_tokens] * (len(blocks) - 2)

    ids, spans, answer_pos, pad_from, n_dropped = _assemble(
        tokenizer, prefix, blocks, suffix,
        seq_len=seq_len, mask_token_id=mask_token_id,
        pad_token_id=tokenizer.pad_token_id or 0, block_budgets=budgets,
        n_required=2, pad_to_multiple_of=pad_to_multiple_of,
    )
    hops, edges = _drop_to_kept(sample, len(spans))
    graph = LinearisedGraph(
        spans=spans, seq_len=int(ids.numel()), edges=edges, hops=hops,
        node_ids=sample.node_ids[: len(spans)], pad_from=pad_from, answer_pos=answer_pos,
    )
    return LinearisedExample(
        input_ids=ids,
        graph=graph,
        answer_pos=answer_pos,
        candidate_ids=candidate_token_ids(tokenizer, len(LP_OPTIONS)),
        label=int(label),
        meta={"task": "lp", "pair": tuple(sample.node_ids[:2]), "n_nodes": len(spans),
              "n_dropped": n_dropped, "n_tokens": pad_from,
              "padded_len": int(ids.numel()),
              "pad_token_id": tokenizer.pad_token_id or 0},
    )
