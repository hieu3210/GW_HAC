"""Baseline multi-level thresholding methods, evaluated on the same criteria
(Otsu between-class variance / Kapur entropy) as :mod:`haclustering.ha_threshold`
so that HA-MLT can be compared fairly.

* :func:`multi_otsu_exact` -- classical *exact* multi-Otsu (fast recursive
  search, `skimage.filters.threshold_multiotsu`). Deterministic global optimum
  of the Otsu criterion; used as the quality ceiling reference.
* :class:`ThresholdDE` -- direct differential-evolution search over the
  ``K - 1`` raw threshold positions (the classical formulation optimised by
  PSO/GA/EHO-based multi-level thresholding in the literature). Search-space
  dimensionality grows with ``K``, unlike HA-MLT's fixed 2-D search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import differential_evolution
from skimage.filters import threshold_multiotsu

from .ha_threshold import CRITERIA, normalized_histogram


@dataclass
class ThresholdResult:
    thresholds: np.ndarray
    criterion: str
    score: float
    n_evaluations: int
    dimensionality: int


def multi_otsu_exact(image: np.ndarray, n_classes: int, n_bins: int = 256) -> ThresholdResult:
    """Exact multi-Otsu thresholds (fast recursive search), classes = ``n_classes``."""
    thr = threshold_multiotsu(image, classes=n_classes, nbins=n_bins)
    thr = np.round(thr).astype(int)
    hist = normalized_histogram(image, n_bins)
    score = CRITERIA["otsu"](hist, thr)
    return ThresholdResult(
        thresholds=thr, criterion="otsu", score=float(score),
        n_evaluations=-1,  # not a black-box search; exact DP, not comparable in count
        dimensionality=n_classes - 1,
    )


class ThresholdDE:
    """Direct differential-evolution search over ``K - 1`` raw thresholds.

    Stands in for the population-based metaheuristics (PSO/GA/EHO/...) used
    throughout the multi-level thresholding literature to search this same
    ``K - 1``-dimensional space.
    """

    def __init__(self, n_classes: int = 4, criterion: str = "otsu", n_bins: int = 256, seed=None):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {list(CRITERIA)}")
        self.n_classes = n_classes
        self.criterion = criterion
        self.n_bins = n_bins
        self.seed = seed

    def fit(self, image: np.ndarray, popsize: int = 15, maxiter: int = 100) -> ThresholdResult:
        hist = normalized_histogram(image, self.n_bins)
        score_fn = CRITERIA[self.criterion]
        n_thr = self.n_classes - 1
        n_evals = 0

        def objective(params: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            thr = np.round(np.sort(params)).astype(int)
            thr = np.clip(thr, 0, self.n_bins - 2)
            if len(set(thr.tolist())) < len(thr):
                return 1e9
            return -score_fn(hist, thr)

        bounds = [(1, self.n_bins - 2)] * n_thr
        result = differential_evolution(
            objective, bounds, popsize=popsize, maxiter=maxiter,
            seed=self.seed, polish=True, tol=1e-7,
        )
        thresholds = np.sort(np.round(result.x).astype(int))
        self.result_ = ThresholdResult(
            thresholds=thresholds,
            criterion=self.criterion,
            score=float(score_fn(hist, thresholds)),
            n_evaluations=n_evals,
            dimensionality=n_thr,
        )
        return self.result_
