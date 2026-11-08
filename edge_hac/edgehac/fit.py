"""Host-side training of the float reference model (EWA-HAC) and its float classifier.

Only the *core* of the repository is used here: :class:`haclustering.clustering.HAClustering`.
Hyper-parameters follow the protocol of the published EWA-HAC study (Silhouette selection over the
36-point grid ``theta, alpha in {0.3, 0.5, 0.7}``, ``gamma in {0.1, 0.5, 1, 2}``, ``sqm_lambda = 0.01``),
except that Silhouette is evaluated on a deterministic sub-sample when ``N`` is large.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_score

from haclustering.clustering import HAClustering

GRID_THETA_ALPHA = (0.3, 0.5, 0.7)
GRID_GAMMA = (0.1, 0.5, 1.0, 2.0)
SQM_LAMBDA = 0.01


@dataclass
class FittedHA:
    """Everything the edge compiler needs from a trained EWA-HAC model."""

    centers: np.ndarray     # (K, F) in [0, 1]
    weights: np.ndarray     # (F,) simplex
    lo: np.ndarray          # (F,)
    hi: np.ndarray          # (F,)
    theta: float
    alpha: float
    gamma: float
    n_iter: int
    train_labels: np.ndarray

    @property
    def K(self) -> int:
        return self.centers.shape[0]

    @property
    def F(self) -> int:
        return self.centers.shape[1]


def semantize(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Min-max semantization with clipping to [0, 1] (the sensor front-end of the reference model)."""
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((np.asarray(X, dtype=float) - lo) / span, 0.0, 1.0)


def float_scores(S: np.ndarray, centers: np.ndarray, weights: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Weighted squared semantic distances ``D_k(s)``, shape (N, K), computed in float64."""
    N = len(S)
    out = np.empty((N, centers.shape[0]))
    for i in range(0, N, chunk):
        d = S[i:i + chunk, None, :] - centers[None, :, :]
        out[i:i + chunk] = np.einsum("nkf,f->nk", d * d, weights)
    return out


def float_predict(S: np.ndarray, centers: np.ndarray, weights: np.ndarray):
    """Reference classifier ``f``: argmin of ``D_k`` (lowest index on ties). Returns labels, margin."""
    D = float_scores(S, centers, weights)
    lab = np.argmin(D, axis=1)
    srt = np.sort(D, axis=1)
    return lab, srt[:, 1] - srt[:, 0], D


def fit_ewahac(X: np.ndarray, K: int, sil_sample: int = 2000, seed: int = 0,
               theta_alpha=GRID_THETA_ALPHA, gammas=GRID_GAMMA) -> FittedHA:
    """Train EWA-HAC with Silhouette-based selection (no labels used)."""
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    best = None
    for th, al, ga in product(theta_alpha, theta_alpha, gammas):
        m = HAClustering(n_clusters=K, theta=th, alpha=al, entropy_gamma=ga,
                         sqm_lambda=SQM_LAMBDA, init="sqm").fit(X)
        labels = m.labels_
        counts = np.bincount(labels, minlength=K)
        valid = bool(np.all(counts >= max(2, int(np.ceil(len(X) / (10 * K))))))
        sil = -1.0                                    # degenerate partitions rank last
        if len(np.unique(labels)) >= 2:
            try:
                sil = silhouette_score(X, labels, sample_size=min(sil_sample, len(X)), random_state=seed)
            except ValueError:                        # the Silhouette sub-sample contains a single cluster
                sil = -1.0
        key = (valid, sil)
        if best is None or key > best[0]:
            best = (key, m, th, al, ga)
    _, m, th, al, ga = best
    return FittedHA(centers=m.cluster_centers_.copy(), weights=m.feature_weights_.copy(), lo=lo, hi=hi,
                    theta=th, alpha=al, gamma=ga, n_iter=int(m.n_iter_), train_labels=m.labels_.copy())
