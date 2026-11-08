"""HA-MLT: Hedge-Algebra-based Multi-Level Thresholding.

Classical multi-level thresholding (Otsu, Kapur entropy, fuzzy entropy, ...)
searches directly for ``K - 1`` free threshold positions on the gray-level
axis. As the number of classes ``K`` grows this search space grows with it,
which is exactly why the literature turned to population-based metaheuristics
(PSO, GA, EHO, ...) for anything beyond 3-4 levels, and why every one of them
needs a repair/sorting step to keep candidate thresholds ordered.

**HA-MLT reparametrizes the problem.** Instead of searching over ``K - 1``
raw threshold values, it fixes the thresholds to be the Semantically
Quantifying Mapping (SQM) values ``nu(t)`` of a rank-selected subset of the
Hedge Algebra's linguistic term set (the same "sqm" prototype-initialisation
rule already used by :class:`haclustering.clustering.HAClustering`). Because
``nu`` is strictly increasing in term rank for *any* valid HA parameters, the
resulting thresholds are ordered **by construction** -- no repair operator is
ever needed. The only free parameters are the two structural constants of the
algebra, ``theta`` (generator split point) and ``alpha`` (negative-hedge
fuzziness share); the search space is therefore **2-dimensional regardless of
K**. Each induced class also inherits an ordered linguistic label (e.g. "Very
Dark" -> ... -> "Very Light") directly from the SQM decoding, which the
gray-level-only baselines have no equivalent for.

See ``paper/icera_2026`` for the accompanying manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .hedge_algebra import HedgeAlgebra, Term

_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- #
# Histogram utilities shared with the baselines
# --------------------------------------------------------------------------- #
def normalized_histogram(image: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Normalised gray-level histogram ``p(g)``, ``g = 0 .. n_bins - 1``."""
    image = np.asarray(image)
    hist, _ = np.histogram(image, bins=n_bins, range=(0, n_bins - 1))
    hist = hist.astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def otsu_between_class_variance(hist: np.ndarray, thresholds: Sequence[int]) -> float:
    """Otsu's total between-class variance for gray-level ``thresholds``."""
    levels = np.arange(len(hist))
    mu_t = float(np.sum(levels * hist))
    edges = [-1] + list(thresholds) + [len(hist) - 1]
    variance = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (levels > a) & (levels <= b)
        n_k = hist[mask].sum()
        if n_k <= _EPS:
            continue
        mu_k = float(np.sum(levels[mask] * hist[mask]) / n_k)
        variance += n_k * (mu_k - mu_t) ** 2
    return variance


def kapur_entropy(hist: np.ndarray, thresholds: Sequence[int]) -> float:
    """Kapur's total class entropy for gray-level ``thresholds`` (to maximise)."""
    edges = [-1] + list(thresholds) + [len(hist) - 1]
    total_entropy = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        p = hist[a + 1 : b + 1]
        n_k = p.sum()
        if n_k <= _EPS:
            continue
        p_norm = p[p > 0] / n_k
        total_entropy += -float(np.sum(p_norm * np.log(p_norm)))
    return total_entropy


CRITERIA = {"otsu": otsu_between_class_variance, "kapur": kapur_entropy}


def apply_thresholds(image: np.ndarray, thresholds: Sequence[int]) -> np.ndarray:
    """Quantise ``image`` to per-class mean gray level given ``thresholds``."""
    image = np.asarray(image, dtype=float)
    edges = [-1] + list(thresholds) + [255]
    out = np.zeros_like(image)
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (image > a) & (image <= b)
        if np.any(mask):
            out[mask] = image[mask].mean()
    # Pixels at the lowest level (== 0) satisfy `> -1` already; nothing left out.
    return out


# --------------------------------------------------------------------------- #
# HA-MLT model
# --------------------------------------------------------------------------- #
@dataclass
class HAThresholdResult:
    thresholds: np.ndarray
    theta: float
    alpha: float
    criterion: str
    score: float
    labels: List[str]
    n_evaluations: int


class HAThreshold:
    """Hedge-Algebra multi-level thresholding.

    Parameters
    ----------
    n_classes:
        Number of gray-level classes ``K`` (``K - 1`` thresholds are produced).
    criterion:
        ``"otsu"`` (between-class variance) or ``"kapur"`` (class entropy),
        both maximised.
    decode_depth:
        Hedge depth used when labelling classes with a linguistic phrase.
    n_bins:
        Number of gray levels (256 for 8-bit images).
    seed:
        Seed for the differential-evolution optimiser.
    """

    def __init__(
        self,
        n_classes: int = 4,
        criterion: str = "otsu",
        decode_depth: int = 2,
        n_bins: int = 256,
        seed: Optional[int] = None,
    ) -> None:
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {list(CRITERIA)}")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes
        self.criterion = criterion
        self.decode_depth = decode_depth
        self.n_bins = n_bins
        self.seed = seed

    # ------------------------------------------------------------------ #
    # SQM-anchored threshold generation (order-guaranteed by construction)
    # ------------------------------------------------------------------ #
    def _terms_for_rank_selection(self, ha: HedgeAlgebra) -> List[Term]:
        depth = 1
        terms = ha.generate_terms(depth)
        while len(terms) < self.n_classes - 1:
            depth += 1
            terms = ha.generate_terms(depth)
        return terms

    def thresholds_from_params(self, theta: float, alpha: float) -> Tuple[np.ndarray, List[str]]:
        """``K - 1`` gray-level thresholds and their SQM class labels for ``(theta, alpha)``."""
        ha = HedgeAlgebra(theta=theta, alpha=alpha)
        terms = self._terms_for_rank_selection(ha)
        nus = np.array(sorted(ha.nu(t) for t in terms))
        n_thr = self.n_classes - 1
        idx = np.linspace(0, len(nus) - 1, n_thr).round().astype(int)
        semantic_thr = nus[idx]
        gray_thr = np.round(semantic_thr * (self.n_bins - 1)).astype(int)
        gray_thr = np.sort(np.clip(gray_thr, 0, self.n_bins - 2))
        # Integer rounding can collapse two SQM ranks that were distinct on the
        # continuous [0, 1] axis into the same gray level; nudge forward to
        # keep the thresholds strictly increasing (order-guarantee survives
        # finite quantisation as long as K - 1 << n_bins).
        for i in range(1, len(gray_thr)):
            if gray_thr[i] <= gray_thr[i - 1]:
                gray_thr[i] = gray_thr[i - 1] + 1
        gray_thr = np.minimum(gray_thr, self.n_bins - 2)
        # Class boundaries -> representative semantic value -> linguistic label.
        edges = [-1] + list(gray_thr) + [self.n_bins - 1]
        labels = []
        for a, b in zip(edges[:-1], edges[1:]):
            mid = (a + b + 1) / 2.0 / (self.n_bins - 1)
            labels.append(ha.label(mid, max_depth=self.decode_depth))
        return gray_thr, labels

    # ------------------------------------------------------------------ #
    # Fit: 2-D search over (theta, alpha) only, independent of K
    # ------------------------------------------------------------------ #
    def fit(self, image: np.ndarray, popsize: int = 12, maxiter: int = 40) -> HAThresholdResult:
        hist = normalized_histogram(image, self.n_bins)
        score_fn = CRITERIA[self.criterion]
        n_evals = 0

        def objective(params: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            theta, alpha = params
            thr, _ = self.thresholds_from_params(theta, alpha)
            if len(set(thr.tolist())) < len(thr):
                return 1e9  # degenerate collapse (rare at extreme theta/alpha)
            return -score_fn(hist, thr)

        bounds = [(0.05, 0.95), (0.05, 0.95)]
        result = differential_evolution(
            objective,
            bounds,
            popsize=popsize,
            maxiter=maxiter,
            seed=self.seed,
            polish=True,
            tol=1e-7,
        )
        theta, alpha = result.x
        thresholds, labels = self.thresholds_from_params(theta, alpha)
        self.result_ = HAThresholdResult(
            thresholds=thresholds,
            theta=float(theta),
            alpha=float(alpha),
            criterion=self.criterion,
            score=float(score_fn(hist, thresholds)),
            labels=labels,
            n_evaluations=n_evals,
        )
        return self.result_

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Quantise ``image`` to the fitted per-class mean gray levels."""
        return apply_thresholds(image, self.result_.thresholds)
