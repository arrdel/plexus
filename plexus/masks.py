"""Topology attention masks and structural attention biases.

This is the experimental surface of the project. TAG-DLM injects graph structure into LLaDA
through a single binary mask (the *star* mask) and reports that exposing the real ego-graph
topology instead makes things dramatically worse (Cora link prediction 91.62 -> 61.76). Every
family below exists to localise *why*, so they share one interface and differ only in which
entries of the attention matrix they touch.

Representation. LLaDA's ``forward`` accepts ``attention_bias`` of shape ``(B, 1, S, S)``; a float
tensor is added to the attention scores before the softmax, and a bool tensor is interpreted as
"True = may attend". We always build float biases so that hard masks and graded biases are the
same object and can be compared, composed, and interpolated without a change of type.

Disallowed entries use ``MASK_BIAS`` rather than ``torch.finfo.min``. LLaDA adds the padding mask
to the topology bias internally, and ``finfo.min + finfo.min`` overflows to ``-inf``, which
``F.scaled_dot_product_attention`` turns into NaNs rather than zero-weight. A large-but-finite
constant sums safely and is numerically indistinguishable after the softmax.

Row/column convention throughout: ``bias[i, j]`` governs query position *i* reading key position
*j*, so the *rows* are the readers. Asymmetric masks (the star is asymmetric) depend on this.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch

# Large but finite: two of these may be summed before reaching the softmax.
MASK_BIAS = -1e9

TARGET = 0  # the target node always occupies local index 0 / the first span


@dataclass
class LinearisedGraph:
    """A sampled ego graph flattened into one token sequence.

    ``spans[i]`` is the half-open token range ``[start, end)`` occupied by local node *i*, and
    local index ``TARGET`` is the prediction target. ``edges`` holds undirected edges in *local*
    index space -- the edges actually observed among the sampled nodes, which is a subgraph of the
    real ego graph because neighbour sampling is capped at ``K`` per hop.

    ``hops[i]`` is the shortest-path distance from the target to local node *i* in the sampled
    subgraph. It is what the graded biases are a function of, so it is stored rather than
    recomputed per mask family.
    """

    spans: list[tuple[int, int]]
    seq_len: int
    edges: set[tuple[int, int]] = field(default_factory=set)
    hops: list[int] = field(default_factory=list)
    edge_types: dict[tuple[int, int], int] = field(default_factory=dict)
    node_ids: list[int] = field(default_factory=list)
    pad_from: int | None = None
    answer_pos: int | None = None

    def __post_init__(self):
        if self.pad_from is None:
            self.pad_from = self.seq_len
        if not self.hops:
            self.hops = self._bfs_hops()
        # Store edges canonically so membership tests do not depend on ordering.
        self.edges = {(min(u, v), max(u, v)) for u, v in self.edges}

    @property
    def n_nodes(self) -> int:
        return len(self.spans)

    def _bfs_hops(self) -> list[int]:
        adj: dict[int, set[int]] = {i: set() for i in range(len(self.spans))}
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)
        hops = [-1] * len(self.spans)
        hops[TARGET] = 0
        frontier = [TARGET]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if hops[w] == -1:
                        hops[w] = d
                        nxt.append(w)
            frontier = nxt
        return hops

    def span_of_token(self) -> torch.Tensor:
        """``(seq_len,)`` long tensor mapping each token to its local node index, or -1."""
        owner = torch.full((self.seq_len,), -1, dtype=torch.long)
        for i, (s, e) in enumerate(self.spans):
            owner[s:e] = i
        return owner


# --------------------------------------------------------------------------------------------
# Node-level adjacency: each family decides which *nodes* may exchange information. Lifting to
# token level is shared, so a family is only ever a statement about graph structure.
# --------------------------------------------------------------------------------------------

def _node_allow_full(g: LinearisedGraph) -> torch.Tensor:
    return torch.ones((g.n_nodes, g.n_nodes), dtype=torch.bool)


def _node_allow_star(g: LinearisedGraph) -> torch.Tensor:
    """TAG-DLM's default. Target reads everything; each neighbour reads only itself and target."""
    n = g.n_nodes
    allow = torch.zeros((n, n), dtype=torch.bool)
    allow[TARGET, :] = True
    allow[:, TARGET] = True
    allow[torch.arange(n), torch.arange(n)] = True
    return allow


def _node_allow_ego(g: LinearisedGraph) -> torch.Tensor:
    """Attention only along observed ego-graph edges (plus self).

    This is the variant TAG-DLM reports as catastrophic. Note what it does *not* contain: a node
    at hop 2 has no edge to the target, so under this mask the target cannot read it at all. The
    star mask lets the target read every sampled node. The two therefore differ in two ways at
    once -- neighbour-neighbour edges are added *and* target reach is removed -- which is why
    ``ego_star`` below exists to separate them.
    """
    n = g.n_nodes
    allow = torch.zeros((n, n), dtype=torch.bool)
    allow[torch.arange(n), torch.arange(n)] = True
    for u, v in g.edges:
        allow[u, v] = True
        allow[v, u] = True
    return allow


def _node_allow_ego_star(g: LinearisedGraph) -> torch.Tensor:
    """Union of the ego-graph edges and the star. Real topology *without* losing target reach.

    If TAG-DLM's ego-graph collapse is caused by the target losing sight of its 2-hop context
    rather than by neighbour-neighbour edges being harmful, this variant recovers the star's
    accuracy and their stated conclusion does not follow.
    """
    return _node_allow_ego(g) | _node_allow_star(g)


def _node_allow_random_rewire(g: LinearisedGraph, generator: torch.Generator | None = None) -> torch.Tensor:
    """Control for ``ego_star``: the same number of *non-star* edges, but random endpoints.

    Isolates "the model benefits from *this* topology" from "the model benefits from *some*
    sparse attention pattern of this density". If ``ego_star`` and ``random_rewire`` score the
    same, the topology is doing nothing beyond setting sparsity.

    Only edges that do not touch the target are randomised. Target-incident edges are already
    implied by the star, so counting them would inflate the number of random edges added and
    leave the control denser than the condition it is supposed to match.
    """
    n = g.n_nodes
    allow = _node_allow_star(g)
    n_extra = sum(1 for u, v in g.edges if TARGET not in (u, v))
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if TARGET not in (u, v)]
    if n_extra and pairs:
        idx = torch.randperm(len(pairs), generator=generator)[:n_extra]
        for t in idx.tolist():
            u, v = pairs[t]
            allow[u, v] = True
            allow[v, u] = True
    return allow


NODE_MASK_FAMILIES: dict[str, Callable[[LinearisedGraph], torch.Tensor]] = {
    "full": _node_allow_full,
    "star": _node_allow_star,
    "ego": _node_allow_ego,
    "ego_star": _node_allow_ego_star,
    "random_rewire": _node_allow_random_rewire,
}


# --------------------------------------------------------------------------------------------
# Graded biases. Instead of hard-gating attention, add a finite bias that is a function of graph
# distance or relation type. The pretrained model keeps a dense attention pattern and structure
# only reweights it, which is the hypothesis that hard masks fail because they are an
# out-of-distribution input to a model that never saw block-sparse attention in pretraining.
# --------------------------------------------------------------------------------------------

@dataclass
class HopBias:
    """Bias by shortest-path distance, in the spirit of Graphormer's spatial encoding.

    ``values[d]`` is added to attention scores between nodes at hop distance *d*; index
    ``len(values) - 1`` is the catch-all for "further than that, or unreachable". When these are
    ``nn.Parameter``s the model *learns* how much topology should matter, and the degenerate
    solution (all zeros) is exactly the no-structure baseline -- so the method can always fall
    back to the thing that already works, which a hard mask cannot do.

    ``answer_as_target`` extends the bias over the answer ``[M]`` position by treating it as part
    of the target node's span. By default the answer position belongs to no node span -- the
    instruction prefix and answer suffix are deliberately left globally visible -- so the graded
    bias never writes to the one attention row that produces the prediction, and its effect on the
    answer is purely indirect through the node representations. That is the same indirect route a
    hard mask has, which is fine for a like-for-like comparison but gives up the thing a graded
    bias can uniquely express: a *direct*, learnable "how much does a hop-k neighbour count when I
    answer". Off by default so the headline arm stays comparable to the mask ladder; the ablation
    is what tells us whether the direct route matters.
    """

    values: torch.Tensor  # (n_buckets,)
    scale: float = 1.0
    exact: bool = True
    answer_as_target: bool = False

    def node_bias(self, g: LinearisedGraph) -> torch.Tensor:
        n_buckets = self.values.shape[0]
        if self.exact:
            d = pairwise_hops(g, cutoff=n_buckets)
            # `pairwise_hops` marks unreachable as -1; that is the catch-all bucket, not distance 0.
            d = torch.where(d < 0, torch.full_like(d, n_buckets - 1), d)
        else:
            # Kept only for ablation. `|h_i - h_j|` composed through the target is exact for the
            # star but badly wrong in general: two distinct hop-1 neighbours score |1-1| = 0, so
            # they land in the *self* bucket and the bias claims they are the same node. Their
            # true distance is 2 (via the target) unless they happen to be adjacent. Anything
            # reported from this path must say so.
            hops = torch.tensor(g.hops, dtype=torch.long)
            d = (hops[:, None] - hops[None, :]).abs()
            d = torch.where(hops[:, None] < 0, n_buckets - 1, d)
            d = torch.where(hops[None, :] < 0, n_buckets - 1, d)
        d = d.clamp(min=0, max=n_buckets - 1)
        return self.values.to(d.device)[d] * self.scale


def pairwise_hops(g: LinearisedGraph, cutoff: int = 8) -> torch.Tensor:
    """Exact all-pairs shortest-path distances over the sampled subgraph, ``-1`` if unreachable."""
    n = g.n_nodes
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in g.edges:
        adj[u].add(v)
        adj[v].add(u)
    dist = torch.full((n, n), -1, dtype=torch.long)
    for src in range(n):
        dist[src, src] = 0
        frontier, d = [src], 0
        while frontier and d < cutoff:
            d += 1
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if dist[src, w] == -1:
                        dist[src, w] = d
                        nxt.append(w)
            frontier = nxt
    return dist


# --------------------------------------------------------------------------------------------
# Lifting node-level structure to a token-level attention bias.
# --------------------------------------------------------------------------------------------

def node_to_token_bias(
    g: LinearisedGraph,
    node_allow: torch.Tensor | None = None,
    node_bias: torch.Tensor | None = None,
    *,
    unassigned_visible: bool = True,
    answer_as_target: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Expand ``(n_nodes, n_nodes)`` node-level structure to ``(seq_len, seq_len)`` token bias.

    ``node_allow`` is a boolean gate and ``node_bias`` an additive term; supplying both applies
    the gate and then adds the bias on the surviving entries.

    ``unassigned_visible`` controls tokens that belong to no node span -- the shared instruction
    prefix, the answer position, separators. They are left globally visible by default: they carry
    the task definition rather than graph content, and gating them would confound "structure
    hurts" with "the prompt became unreadable".

    Padding is masked in both directions, matching TAG-DLM.
    """
    S = g.seq_len
    owner = g.span_of_token()
    bias = torch.zeros((S, S), dtype=dtype)

    assigned = owner >= 0
    idx = owner.clamp(min=0)

    if node_allow is not None:
        allow_tok = node_allow.to(torch.bool)[idx[:, None], idx[None, :]]
        pair_assigned = assigned[:, None] & assigned[None, :]
        if unassigned_visible:
            allow_tok = allow_tok | ~pair_assigned
        else:
            allow_tok = allow_tok & pair_assigned
        bias = bias.masked_fill(~allow_tok, MASK_BIAS)

    if node_bias is not None:
        b_assigned, b_idx = assigned, idx
        if answer_as_target and g.answer_pos is not None:
            # Only the additive term sees this, never the gate. Rewriting the gate's ownership
            # would silently change what `star` and `ego` mean and make the graded arm
            # incomparable to the mask ladder it is supposed to be measured against.
            b_assigned = assigned.clone()
            b_assigned[g.answer_pos] = True
            b_idx = idx.clone()
            b_idx[g.answer_pos] = TARGET
        add = node_bias.to(dtype)[b_idx[:, None], b_idx[None, :]]
        pair_assigned = b_assigned[:, None] & b_assigned[None, :]
        bias = bias + torch.where(pair_assigned, add, torch.zeros((), dtype=dtype))

    if g.pad_from is not None and g.pad_from < S:
        bias[g.pad_from :, :] = MASK_BIAS
        bias[:, g.pad_from :] = MASK_BIAS

    return bias


def build_attention_bias(
    graphs: Sequence[LinearisedGraph],
    family: str = "star",
    hop_bias: HopBias | None = None,
    *,
    seq_len: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    unassigned_visible: bool = True,
) -> torch.Tensor:
    """Build the batched ``(B, 1, S, S)`` tensor to hand to LLaDA's ``attention_bias``.

    ``family="none"`` yields padding-only masking, i.e. the unmodified bidirectional model.

    Memory warning: this is dense and quadratic. At ``S=4096`` in fp32 each sample costs 67 MB,
    so the bias -- not the activations -- can become the binding constraint on batch size for
    ogbn-arxiv. ``dtype=torch.bfloat16`` halves it, at the cost of coarsening graded biases.
    """
    if family not in NODE_MASK_FAMILIES and family != "none":
        raise KeyError(f"unknown mask family {family!r}; have {sorted(NODE_MASK_FAMILIES) + ['none']}")

    S = seq_len or max(g.seq_len for g in graphs)
    out = torch.zeros((len(graphs), 1, S, S), dtype=dtype)

    for b, g in enumerate(graphs):
        if g.seq_len != S:
            g = LinearisedGraph(
                spans=g.spans, seq_len=S, edges=set(g.edges), hops=list(g.hops),
                edge_types=dict(g.edge_types), node_ids=list(g.node_ids),
                pad_from=g.pad_from, answer_pos=g.answer_pos,
            )
        if family == "none":
            allow = None
        elif family == "random_rewire":
            allow = _node_allow_random_rewire(g, generator=generator)
        else:
            allow = NODE_MASK_FAMILIES[family](g)

        nb = hop_bias.node_bias(g) if hop_bias is not None else None
        out[b, 0] = node_to_token_bias(
            g, node_allow=allow, node_bias=nb, dtype=dtype,
            unassigned_visible=unassigned_visible,
            answer_as_target=bool(hop_bias is not None and hop_bias.answer_as_target),
        )

    return out.to(device)


def _live_rows(bias: torch.Tensor) -> torch.Tensor:
    """Rows that can read at least one position, i.e. the non-padding rows."""
    return (bias > MASK_BIAS / 2).any(-1)


def mask_density(bias: torch.Tensor, exclude_padding: bool = True) -> float:
    """Fraction of attention entries left readable. The confound every mask comparison shares.

    Two mask families that differ in accuracy also differ in how much context each token sees.
    Reporting density alongside accuracy is what separates "this topology is better" from "this
    topology happens to let more tokens talk".

    Padding is excluded by default. Sequences are padded to a fixed length and padding is masked
    in both directions, so including it makes density a measurement of the padding fraction rather
    than of the mask: a short sequence in a 4096-token buffer reads as ~1% dense under *every*
    family, and the families become indistinguishable.
    """
    readable = bias > MASK_BIAS / 2
    if not exclude_padding:
        return readable.float().mean().item()
    live = _live_rows(bias)
    if live.sum() == 0:
        return 0.0
    sub = readable[..., live.squeeze(), :][..., live.squeeze()] if live.dim() == 1 else None
    if sub is None:
        # Batched: reduce per sample so that different samples may have different lengths.
        vals = [mask_density(bias[i], exclude_padding=True) for i in range(bias.shape[0])]
        return float(sum(vals) / len(vals))
    return sub.float().mean().item()


def describe(bias: torch.Tensor) -> dict[str, float]:
    """Summary statistics for logging a mask family alongside its accuracy."""
    if bias.dim() > 2:
        per = [describe(bias[i]) for i in range(bias.shape[0])]
        return {k: float(sum(d[k] for d in per) / len(per)) for k in per[0]}
    readable = bias > MASK_BIAS / 2
    live = _live_rows(bias)
    per_row = readable[live].float().sum(-1)
    return {
        "density": mask_density(bias),
        "n_live_tokens": float(live.sum().item()),
        "min_row_visibility": per_row.min().item() if per_row.numel() else 0.0,
        "mean_row_visibility": per_row.mean().item() if per_row.numel() else 0.0,
        "graded_frac": ((bias != 0) & readable)[live].float().mean().item() if live.any() else 0.0,
    }


def sample_random_node_control(
    n_nodes: int,
    all_node_ids: Iterable[int],
    generator: torch.Generator | None = None,
) -> list[int]:
    """Control for the *content* of the neighbourhood rather than its wiring.

    Replaces the sampled ego-graph neighbours with uniformly random nodes from the graph while
    keeping the star mask and the prompt format fixed. TAG-DLM's neighbour-count ablation shows
    accuracy rises with K, but never checks whether the neighbours have to be *graph* neighbours.
    If random nodes do as well, the reported gain is prompt context, not topology.
    """
    pool = torch.tensor(list(all_node_ids))
    idx = torch.randperm(pool.numel(), generator=generator)[:n_nodes]
    return pool[idx].tolist()
