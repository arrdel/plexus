"""Text-attributed graph loading.

One canonical in-memory representation, `TAG`, so that everything downstream -- ego-graph
sampling, prompt construction, the GNN baselines of E0.5 -- reads the same object.

**Provenance matters more than convenience here.** TAG-DLM reports Cora node classification at
90.96, which is far above what the standard Planetoid public split supports (20 labelled nodes per
class; strong text models land in the low 80s). It is only reachable with a *dense* label split,
and TAG-DLM states it follows LLaGA's preprocessing. LLaGA's released Cora is a Box download we
cannot script, but it derives from the same McCallum-Cora-with-abstracts that TAPE distributes,
and TAPE splits 60/20/20 at random. 2708 * 0.6 = 1625, which is the "~1.6k train nodes" the
compute budget in RESEARCH_PLAN.md §6 was written against.

So we use TAPE's source and TAPE's split ratio, and record the choice as an assumption rather
than a fact: **if E0.1 misses its gate, the split is the first thing to suspect.** See
`SPLIT_PROVENANCE` below, which is carried into every run's config so no result is ever separated
from the assumption it rests on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Recorded in every experiment config. If E0.1 fails its gate, this is suspect #1.
SPLIT_PROVENANCE = (
    "TAPE/McCallum Cora (title+abstract), random 60/20/20 split by seed. "
    "ASSUMED to match TAG-DLM, which cites LLaGA's preprocessing but publishes no code and no "
    "split file. Not verified against the paper."
)

PUBMED_SPLIT_PROVENANCE = (
    "TAPE/LINQS Pubmed-Diabetes (title+abstract), random 60/20/20 split by seed. "
    "ASSUMED to match TAG-DLM on the same grounds as Cora: it cites LLaGA's preprocessing but "
    "publishes no code and no split file. Not verified against the paper."
)

# The three LINQS labels, in the order the Pubmed-Diabetes README lists them, which is the order
# `label=1,2,3` in the NODE file indexes. The mapping is asserted, not assumed: see
# `pubmed_label_keyword_agreement`, which checks it against the abstracts.
PUBMED_CLASSES = [
    "Diabetes_Mellitus_Experimental",
    "Diabetes_Mellitus_Type_1",
    "Diabetes_Mellitus_Type_2",
]

PUBMED_CLASS_PROMPT_NAMES = [
    "experimentally induced diabetes",
    "type 1 diabetes",
    "type 2 diabetes",
]

# TAPE's class ordering, kept verbatim so our label indices match theirs.
CORA_CLASSES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]

# Human-readable class names for prompts. The raw labels are snake_case identifiers; a language
# model should be shown English. Ordering is CORA_CLASSES, so index i means the same thing in both.
CORA_CLASS_PROMPT_NAMES = [
    "case-based reasoning",
    "genetic algorithms",
    "neural networks",
    "probabilistic methods",
    "reinforcement learning",
    "rule learning",
    "theory",
]


@dataclass
class TAG:
    """A text-attributed graph.

    `edge_index` is `(2, E)` and **undirected**: every edge appears in both directions, which is
    what the neighbour lookup in `sampling.py` assumes. `text[i]` is node *i*'s raw attribute;
    it may be an empty string when the source corpus has no extraction for that node, and
    `text_coverage` records how often that happens rather than letting it pass silently.
    """

    name: str
    text: list[str]
    labels: torch.Tensor          # (N,) long
    edge_index: torch.Tensor      # (2, E) long, both directions present
    class_names: list[str]
    class_prompt_names: list[str]
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    split_seed: int
    split_provenance: str = SPLIT_PROVENANCE
    text_coverage: dict[str, float] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.text)

    @property
    def num_edges(self) -> int:
        """Undirected edge count, i.e. half the columns of `edge_index`."""
        return self.edge_index.shape[1] // 2

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def build_adjacency(self) -> list[list[int]]:
        """Sorted neighbour lists. Sorted so that sampling is reproducible across runs."""
        adj: list[list[int]] = [[] for _ in range(self.num_nodes)]
        src, dst = self.edge_index.tolist()
        for u, v in zip(src, dst, strict=True):
            adj[u].append(v)
        return [sorted(set(a)) for a in adj]

    def summary(self) -> str:
        deg = self.edge_index.shape[1] / max(self.num_nodes, 1)
        return (
            f"{self.name}: {self.num_nodes} nodes, {self.num_edges} undirected edges, "
            f"{self.num_classes} classes, mean degree {deg:.2f}, "
            f"split {len(self.train_idx)}/{len(self.val_idx)}/{len(self.test_idx)} "
            f"(seed {self.split_seed}), text coverage "
            f"{self.text_coverage.get('both', float('nan')):.3f}"
        )


def _parse_cora_graph(cora_dir: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Parse `cora.content` and `cora.cites` into (paper_ids, labels, undirected edge_index).

    Mirrors TAPE's `parse_cora`, including dropping citation entries whose endpoints are not in
    `cora.content` -- the citation file references papers outside the labelled set.
    """
    content = cora_dir / "cora.content"
    cites = cora_dir / "cora.cites"
    if not content.exists() or not cites.exists():
        raise FileNotFoundError(
            f"expected {content} and {cites}. Fetch the TAPE Cora archive first:\n"
            f"  gdown 1hxE0OPR7VLEHesr48WisynuoNMhXJbpl -O cora_orig.zip && unzip cora_orig.zip"
        )

    rows = np.genfromtxt(content, dtype=str)
    paper_ids = rows[:, 0]
    label_names = rows[:, -1]

    class_map = {c: i for i, c in enumerate(CORA_CLASSES)}
    unknown = set(label_names) - set(class_map)
    if unknown:
        raise ValueError(f"unexpected Cora labels {sorted(unknown)}")
    labels = np.array([class_map[c] for c in label_names], dtype=np.int64)

    id_to_local = {pid: i for i, pid in enumerate(paper_ids)}
    raw = np.genfromtxt(cites, dtype=str)
    pairs = [
        (id_to_local[a], id_to_local[b])
        for a, b in raw
        if a in id_to_local and b in id_to_local
    ]
    # Symmetrise and drop self-loops; duplicates are removed by the unique below.
    both = {(u, v) for u, v in pairs if u != v} | {(v, u) for u, v in pairs if u != v}
    edge_index = torch.tensor(sorted(both), dtype=torch.long).t().contiguous()
    return paper_ids, labels, edge_index


_TITLE = re.compile(r"^Title:\s*(.*)$")
_ABSTRACT = re.compile(r"^Abstract:\s*(.*)$")


def _parse_cora_text(cora_dir: Path, paper_ids: np.ndarray) -> tuple[list[str], dict[str, float]]:
    """Read title and abstract per node from the McCallum extractions.

    TAPE's loader assumes every extraction file contains both a `Title:` and an `Abstract:` line
    and reuses the previous paper's values when one is absent, which silently mislabels those
    nodes. We record misses instead and leave the field empty, then report coverage -- a prompt
    built from another paper's abstract is a data bug that would be invisible in the accuracy.
    """
    papers = cora_dir / "mccallum" / "cora" / "papers"
    extractions = cora_dir / "mccallum" / "cora" / "extractions"
    if not papers.exists() or not extractions.is_dir():
        raise FileNotFoundError(f"expected {papers} and {extractions}/")

    pid_to_file: dict[str, str] = {}
    with open(papers, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                pid_to_file[parts[0]] = parts[1].strip()

    texts: list[str] = []
    n_title = n_abstract = n_both = n_file = 0
    for pid in paper_ids:
        fn = pid_to_file.get(str(pid))
        title = abstract = ""
        if fn and (extractions / fn).exists():
            n_file += 1
            with open(extractions / fn, encoding="utf-8", errors="replace") as f:
                for line in f.read().splitlines():
                    if (m := _TITLE.match(line)) and not title:
                        title = m.group(1).strip()
                    elif (m := _ABSTRACT.match(line)) and not abstract:
                        abstract = m.group(1).strip()
        n_title += bool(title)
        n_abstract += bool(abstract)
        n_both += bool(title and abstract)
        # Keep the shape uniform: a node with no title still gets an (empty) abstract slot.
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        texts.append("\n".join(parts))

    n = len(paper_ids)
    coverage = {
        "file_found": n_file / n,
        "title": n_title / n,
        "abstract": n_abstract / n,
        "both": n_both / n,
        "empty": sum(1 for t in texts if not t) / n,
    }
    return texts, coverage


def _split_60_20_20(num_nodes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TAPE's split: shuffle all nodes, take 60/20/20. Sorted within each split, as TAPE does."""
    rng = np.random.RandomState(seed)
    order = np.arange(num_nodes)
    rng.shuffle(order)
    a, b = int(num_nodes * 0.6), int(num_nodes * 0.8)
    return (
        torch.tensor(np.sort(order[:a]), dtype=torch.long),
        torch.tensor(np.sort(order[a:b]), dtype=torch.long),
        torch.tensor(np.sort(order[b:]), dtype=torch.long),
    )


def load_cora(root: str | Path, seed: int = 0, cache: bool = True) -> TAG:
    """Load Cora with raw title/abstract text.

    `root` is the directory *containing* `cora_orig/`. Reading 2708 extraction files out of 52905
    takes a few seconds, so the parsed result is cached alongside it as `cora_tag_seed{seed}.pt`.
    """
    root = Path(root)
    cache_path = root / f"cora_tag_seed{seed}.pt"
    if cache and cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        return TAG(**blob)

    cora_dir = root / "cora_orig"
    paper_ids, labels, edge_index = _parse_cora_graph(cora_dir)
    texts, coverage = _parse_cora_text(cora_dir, paper_ids)
    train_idx, val_idx, test_idx = _split_60_20_20(len(paper_ids), seed)

    tag = TAG(
        name="cora",
        text=texts,
        labels=torch.tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        class_names=list(CORA_CLASSES),
        class_prompt_names=list(CORA_CLASS_PROMPT_NAMES),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_seed=seed,
        text_coverage=coverage,
    )
    if cache:
        torch.save(tag.__dict__, cache_path)
    return tag


def _parse_pubmed_graph(pubmed_dir: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Parse the LINQS NODE and DIRECTED files into (pmids, labels, undirected edge_index).

    The NODE file's first two lines are a banner and a schema; data rows begin at line 3 and start
    with the PMID followed by `label=k` for k in 1..3. The cites file references papers by PMID
    and, unlike Cora's, every endpoint we have seen is present in the node file -- but we drop
    unknown endpoints anyway rather than trust that, and report how many were dropped.
    """
    nodes = pubmed_dir / "data" / "Pubmed-Diabetes.NODE.paper.tab"
    cites = pubmed_dir / "data" / "Pubmed-Diabetes.DIRECTED.cites.tab"
    if not nodes.exists() or not cites.exists():
        raise FileNotFoundError(
            f"expected {nodes} and {cites}. Fetch the TAPE PubMed archive first:\n"
            f"  gdown 1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W -O PubMed.zip && unzip PubMed.zip"
        )

    pmids: list[str] = []
    labels: list[int] = []
    with open(nodes, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f):
            if lineno < 2 or not line.strip():
                continue
            fields = line.split("\t")
            label_field = next((x for x in fields if x.startswith("label=")), None)
            if label_field is None:
                raise ValueError(f"{nodes}:{lineno + 1} has no label= field")
            k = int(label_field.split("=", 1)[1])
            if not 1 <= k <= len(PUBMED_CLASSES):
                raise ValueError(f"{nodes}:{lineno + 1} has label {k}, expected 1..3")
            pmids.append(fields[0])
            labels.append(k - 1)

    id_to_local = {pid: i for i, pid in enumerate(pmids)}
    pairs, dropped = [], 0
    with open(cites, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f):
            if lineno < 2 or not line.strip():
                continue
            refs = [x.split("paper:", 1)[1] for x in line.split() if x.startswith("paper:")]
            if len(refs) != 2:
                continue
            a, b = refs
            if a in id_to_local and b in id_to_local:
                pairs.append((id_to_local[a], id_to_local[b]))
            else:
                dropped += 1
    if dropped:
        print(f"  pubmed: dropped {dropped} citations with an endpoint outside the node file")

    both = {(u, v) for u, v in pairs if u != v} | {(v, u) for u, v in pairs if u != v}
    edge_index = torch.tensor(sorted(both), dtype=torch.long).t().contiguous()
    return np.array(pmids), np.array(labels, dtype=np.int64), edge_index


def _parse_pubmed_text(pubmed_dir: Path, pmids: np.ndarray) -> tuple[list[str], dict[str, float]]:
    """Read title (`TI`) and abstract (`AB`) per node from `pubmed.json`, keyed by PMID.

    TAPE reads this file positionally, which is correct only while its order matches the NODE
    file's. We key by PMID instead and report coverage, for the same reason the Cora loader does:
    a prompt built from the wrong paper's abstract is invisible in an accuracy number.
    """
    blob = pubmed_dir / "pubmed.json"
    if not blob.exists():
        raise FileNotFoundError(f"expected {blob}")

    records = json.loads(blob.read_text(encoding="utf-8", errors="replace"))
    by_pmid = {str(r.get("PMID")): r for r in records}

    texts: list[str] = []
    n_title = n_abstract = n_both = n_found = 0
    for pid in pmids:
        rec = by_pmid.get(str(pid))
        title = (rec or {}).get("TI", "").strip()
        abstract = (rec or {}).get("AB", "").strip()
        n_found += rec is not None
        n_title += bool(title)
        n_abstract += bool(abstract)
        n_both += bool(title and abstract)
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        texts.append("\n".join(parts))

    n = len(pmids)
    coverage = {
        "file_found": n_found / n,
        "title": n_title / n,
        "abstract": n_abstract / n,
        "both": n_both / n,
        "empty": sum(1 for t in texts if not t) / n,
    }
    return texts, coverage


# Terms that should be over-represented in each class if the label index means what we think it
# does. Deliberately blunt: the point is to catch a permuted mapping, not to classify.
_PUBMED_LABEL_PROBES = (
    ("rat", "rats", "mice", "mouse", "streptozotocin", "animal"),
    ("iddm", "insulin-dependent", "juvenile"),
    ("niddm", "non-insulin-dependent", "noninsulin-dependent"),
)


def pubmed_label_keyword_agreement(texts: list[str], labels: torch.Tensor) -> list[list[float]]:
    """Rate at which each class's texts hit each probe set, as a 3x3 matrix, row `i` = class `i`.

    A permuted label mapping is the kind of error that costs a paper its numbers and raises no
    exception, so we check ours instead of asserting it. Read the result **down the columns**, not
    across the rows: animal-model vocabulary is common to the whole diabetes literature, so
    probe 0 is the most frequent probe in every class and every row peaks on it. What identifies
    the mapping is which class is most enriched for a given probe, i.e. `argmax` of each column,
    and a correct mapping puts that on the diagonal.
    """
    lowered = [t.lower() for t in texts]
    out = []
    for i in range(len(PUBMED_CLASSES)):
        idx = [j for j, y in enumerate(labels.tolist()) if y == i]
        row = [
            sum(any(p in lowered[j] for p in probes) for j in idx) / max(len(idx), 1)
            for probes in _PUBMED_LABEL_PROBES
        ]
        out.append(row)
    return out


def pubmed_label_mapping_is_consistent(texts: list[str], labels: torch.Tensor) -> bool:
    """True when each probe set is most enriched in the class it was written for."""
    m = pubmed_label_keyword_agreement(texts, labels)
    return all(max(range(len(m)), key=lambda i: m[i][c]) == c for c in range(len(m[0])))


def load_pubmed(root: str | Path, seed: int = 0, cache: bool = True) -> TAG:
    """Load PubMed (Pubmed-Diabetes) with raw title/abstract text.

    `root` is the directory *containing* `PubMed_orig/`. 19717 nodes against Cora's 2708, so this
    is the dataset that tells us whether a conclusion drawn on Cora survives a 7x larger graph.
    """
    root = Path(root)
    cache_path = root / f"pubmed_tag_seed{seed}.pt"
    if cache and cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        return TAG(**blob)

    pubmed_dir = root / "PubMed_orig"
    pmids, labels, edge_index = _parse_pubmed_graph(pubmed_dir)
    texts, coverage = _parse_pubmed_text(pubmed_dir, pmids)
    train_idx, val_idx, test_idx = _split_60_20_20(len(pmids), seed)

    tag = TAG(
        name="pubmed",
        text=texts,
        labels=torch.tensor(labels, dtype=torch.long),
        edge_index=edge_index,
        class_names=list(PUBMED_CLASSES),
        class_prompt_names=list(PUBMED_CLASS_PROMPT_NAMES),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_seed=seed,
        split_provenance=PUBMED_SPLIT_PROVENANCE,
        text_coverage=coverage,
    )
    if cache:
        torch.save(tag.__dict__, cache_path)
    return tag


_LOADERS = {"cora": load_cora, "pubmed": load_pubmed}


def load_tag(name: str, root: str | Path, seed: int = 0, **kw) -> TAG:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(_LOADERS)}")
    return _LOADERS[name](root, seed=seed, **kw)
