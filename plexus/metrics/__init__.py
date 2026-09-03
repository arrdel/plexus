"""Metrics. Accuracy is never reported alone.

RESEARCH_PLAN.md §4 makes two rules that this module enforces rather than trusts:

- **Mask density travels with accuracy.** Two families that differ in accuracy also differ in how
  much context each token sees. `Result.row()` refuses to format one without the other.
- **A Cora difference without error bars is not a result.** Cora has 2708 nodes and a 542-node
  test split, so a 1.5-point gap is inside plausible seed noise. `aggregate` reports mean +/- std
  over seeds and the paired count, and `overlaps` says plainly whether two results are separable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Result:
    """One evaluated configuration."""

    accuracy: float
    n: int
    mask_family: str
    mask_density: float
    task: str
    seed: int
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def stderr(self) -> float:
        """Binomial standard error, the floor on the uncertainty of a single run."""
        p = self.accuracy
        return math.sqrt(max(p * (1 - p), 0.0) / self.n) if self.n else float("nan")

    def row(self) -> str:
        return (
            f"{self.task:3s} {self.mask_family:14s} acc={100 * self.accuracy:6.2f} "
            f"+/-{100 * self.stderr:4.2f} (n={self.n}) density={self.mask_density:.4f} seed={self.seed}"
        )


@dataclass
class Aggregate:
    """A configuration evaluated over several seeds."""

    mask_family: str
    task: str
    mean: float
    std: float
    n_seeds: int
    mask_density: float
    n: int

    def row(self) -> str:
        return (
            f"{self.task:3s} {self.mask_family:14s} acc={100 * self.mean:6.2f} "
            f"+/-{100 * self.std:4.2f} over {self.n_seeds} seeds  density={self.mask_density:.4f}"
        )


def accuracy(predictions: list[int], labels: list[int]) -> float:
    if len(predictions) != len(labels):
        raise ValueError(f"{len(predictions)} predictions but {len(labels)} labels")
    if not labels:
        return float("nan")
    return sum(int(p == y) for p, y in zip(predictions, labels, strict=True)) / len(labels)


def aggregate(results: list[Result]) -> Aggregate:
    """Mean and *sample* std over seeds.

    Sample std (n-1) not population std: with 3 seeds the population form understates the spread
    by ~18%, and these error bars decide whether E0.2's differences are real.
    """
    if not results:
        raise ValueError("nothing to aggregate")
    families = {r.mask_family for r in results}
    tasks = {r.task for r in results}
    if len(families) != 1 or len(tasks) != 1:
        raise ValueError(f"cannot aggregate across {families} / {tasks}")

    accs = [r.accuracy for r in results]
    mean = sum(accs) / len(accs)
    if len(accs) > 1:
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
    else:
        std = float("nan")
    return Aggregate(
        mask_family=results[0].mask_family,
        task=results[0].task,
        mean=mean,
        std=std,
        n_seeds=len(accs),
        mask_density=sum(r.mask_density for r in results) / len(results),
        n=results[0].n,
    )


def overlaps(a: Aggregate, b: Aggregate, k: float = 1.0) -> bool:
    """Whether two aggregates' error bars overlap at `k` standard deviations.

    Used to keep the write-up honest: where our bars overlap a gap TAG-DLM reports as meaningful,
    §4 requires us to say so rather than repeat their framing.
    """
    if math.isnan(a.std) or math.isnan(b.std):
        return True  # one seed each: not separable, whatever the means say
    return abs(a.mean - b.mean) <= k * (a.std + b.std)


__all__ = ["Aggregate", "Result", "accuracy", "aggregate", "overlaps"]
