"""Ego-graph sampling.

TAG-DLM samples a *k*-hop ego graph around the target, capped at *K* neighbours per hop, and
linearises it into one sequence. Its default is k=2, K=10 -- the same `2_10` that names LLaGA's
released files, which is the strongest available evidence that the two use the same procedure.

Two properties are load-bearing for the experiments and are asserted in the tests:

1. **Local index 0 is always the target.** `plexus.masks` builds every family around `TARGET = 0`.
2. **`edges` is the induced subgraph on the sampled nodes**, not the edges BFS happened to walk.
   The `ego` and `ego_star` families are statements about observed topology, so an edge between
   two sampled hop-1 nodes must be present even though BFS never traversed it -- that neighbour-
   neighbour edge is precisely what TAG-DLM blames for the ego mask's collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from plexus.data.tags import TAG


@dataclass
class EgoSample:
    """A sampled neighbourhood in local index space, ready for linearisation.

    `node_ids[i]` is the global node id of local node *i*, with local 0 the target. `hops[i]` is
    the hop at which local node *i* was discovered. `edges` is the induced undirected subgraph in
    local indices, canonicalised as `(min, max)`.
    """

    node_ids: list[int]
    hops: list[int]
    edges: set[tuple[int, int]]

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)


def _derive_generator(seed: int, *tags: int) -> torch.Generator:
    """A generator keyed on (seed, example identity).

    Sampling must not depend on iteration order, or two runs that differ only in shuffling would
    see different neighbourhoods and the seed-to-seed variance in §4 would be measuring the wrong
    thing.
    """
    h = seed
    for t in tags:
        h = (h * 1_000_003 + int(t)) % (2**63 - 1)
    return torch.Generator().manual_seed(h)


def _choose(candidates: list[int], k: int, generator: torch.Generator) -> list[int]:
    """Uniformly sample at most `k` of `candidates`, without replacement."""
    if len(candidates) <= k:
        return list(candidates)
    idx = torch.randperm(len(candidates), generator=generator)[:k]
    return [candidates[i] for i in idx.tolist()]


def sample_ego_graph(
    adj: list[list[int]],
    centers: int | list[int],
    *,
    k: int = 2,
    K: int = 10,
    seed: int = 0,
    exclude_edges: set[tuple[int, int]] | None = None,
) -> EgoSample:
    """Sample the *k*-hop neighbourhood around `centers`, at most `K` new nodes per hop.

    `centers` may be a list, in which case all centres occupy the first local indices in the order
    given and BFS expands from all of them at once -- that is how a link-prediction pair is
    sampled, so that both endpoints get context.

    `exclude_edges` is removed from consideration *both* when walking and when computing the
    induced subgraph. Link prediction needs this: if the queried edge survives into `edges`, then
    the `ego` and `ego_star` masks encode the answer directly in the attention pattern and the
    task becomes trivial for exactly the families under test. That would not look like a bug, it
    would look like a result.
    """
    if isinstance(centers, int):
        centers = [centers]
    excluded = set()
    if exclude_edges:
        excluded = {(min(u, v), max(u, v)) for u, v in exclude_edges}

    def neighbours(u: int) -> list[int]:
        return [v for v in adj[u] if (min(u, v), max(u, v)) not in excluded]

    node_ids: list[int] = list(dict.fromkeys(centers))  # dedupe, keep order
    hops: list[int] = [0] * len(node_ids)
    seen = set(node_ids)

    frontier = list(node_ids)
    for hop in range(1, k + 1):
        # Pool candidates across the whole frontier, then cap, so that K bounds the growth per
        # hop rather than per frontier node -- otherwise sequence length scales with degree.
        pool = [v for u in frontier for v in neighbours(u) if v not in seen]
        pool = sorted(dict.fromkeys(pool))
        gen = _derive_generator(seed, *centers, hop)
        chosen = _choose(pool, K, gen)
        for v in chosen:
            seen.add(v)
            node_ids.append(v)
            hops.append(hop)
        frontier = chosen
        if not frontier:
            break

    local_of = {g: i for i, g in enumerate(node_ids)}
    edges: set[tuple[int, int]] = set()
    for gu, lu in local_of.items():
        for gv in adj[gu]:
            lv = local_of.get(gv)
            if lv is None or lv == lu:
                continue
            if (min(gu, gv), max(gu, gv)) in excluded:
                continue
            edges.add((min(lu, lv), max(lu, lv)))

    return EgoSample(node_ids=node_ids, hops=hops, edges=edges)


def sample_lp_examples(
    tag: TAG,
    node_idx: torch.Tensor,
    *,
    n_examples: int | None = None,
    seed: int = 0,
) -> list[tuple[int, int, int]]:
    """Balanced `(u, v, label)` link-prediction examples with both endpoints in `node_idx`.

    Positives are existing edges; negatives are uniformly sampled non-adjacent pairs. Returned
    balanced 50/50, because the metric TAG-DLM reports is accuracy and an unbalanced set makes
    accuracy uninterpretable -- a majority-class predictor would score whatever the imbalance is.
    """
    allowed = set(node_idx.tolist())
    src, dst = tag.edge_index.tolist()
    positives = sorted({(min(u, v), max(u, v)) for u, v in zip(src, dst, strict=True)
                        if u in allowed and v in allowed and u != v})
    adjacent = set(positives)

    gen = _derive_generator(seed, len(positives), 0xA11)
    n_pos = len(positives) if n_examples is None else min(len(positives), n_examples // 2)
    order = torch.randperm(len(positives), generator=gen)[:n_pos]
    chosen_pos = [positives[i] for i in order.tolist()]

    pool = sorted(allowed)
    negatives: list[tuple[int, int]] = []
    seen_neg: set[tuple[int, int]] = set()
    # Rejection sampling: Cora's density is ~0.1%, so a random pair is almost never an edge and
    # this terminates quickly. The bound stops it spinning if a caller passes a tiny node set.
    for _ in range(64 * max(n_pos, 1)):
        if len(negatives) >= n_pos:
            break
        pair_idx = torch.randint(len(pool), (2,), generator=gen).tolist()
        u, v = pool[pair_idx[0]], pool[pair_idx[1]]
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in adjacent or pair in seen_neg:
            continue
        seen_neg.add(pair)
        negatives.append(pair)

    examples = [(u, v, 1) for u, v in chosen_pos] + [(u, v, 0) for u, v in negatives]
    shuffle = torch.randperm(len(examples), generator=gen).tolist()
    return [examples[i] for i in shuffle]
