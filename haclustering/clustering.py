"""Entropy-Weighted Adaptive Hedge Algebra Clustering (EWA-HAC).

The model partitions data on the **semantic axis** of a Hedge Algebra:

1. **Semantization.** Each feature ``j`` is mapped from its observed range onto
   the HA semantic axis ``[0, 1]`` via :meth:`HedgeAlgebra.semantize`.
2. **Clustering.** Cluster prototypes ``SC`` live in ``[0, 1]^F``; an alternating
   optimization (AO) minimises a weighted semantic dispersion with (i) a Shannon
   entropy regulariser on the feature weights ``W`` and (ii) an optional
   SQM-regularisation term that gently attracts prototypes toward linguistically
   meaningful SQM term positions throughout optimisation.
3. **Desemantization.** Each prototype coordinate is decoded back into a
   linguistic term (e.g. *Very High*), giving human-readable cluster summaries.

Objective
---------
With crisp partition ``Pi``, prototypes ``SC``, weights ``W`` (``sum_j w_j = 1``),
and SQM term values ``{nu(t)}``:

    J = sum_k sum_{i in C_k} sum_j w_j (S_ij - SC_kj)^2
        + g * sum_j w_j ln(w_j)
        + sqm_lambda * sum_k sum_j min_t (SC_kj - nu(t))^2

where ``S`` is the semantized data, ``g`` is a scale-stable entropy coefficient,
and ``sqm_lambda >= 0`` controls the SQM attraction strength. With
``sqm_lambda = 0`` the third term vanishes and the objective reduces to the
standard entropy-weighted k-means on the semantic axis.

Scale-stable weighting (fixes the "over-weighting paradox").
The naive update ``w_j ~ exp(-D_j / gamma)`` collapses on high-dimensional data
because the raw dispersions ``D_j`` grow with ``N`` and the feature scale, so a
fixed ``gamma`` over-concentrates the weight on one feature. We compute a single
data-driven scale ``s`` (mean per-feature total variance of the semantized data)
**once** and use ``g = entropy_gamma * s`` throughout. Because ``g`` is held
fixed during iteration, ``J`` is a well-defined objective and AO decreases it
monotonically (see :meth:`HAClustering.fit` and the convergence test).

Convergence
-----------
``J`` is bounded below (the dispersion term is non-negative and the entropy term
is ``>= g * (-ln F)`` and the SQM term is non-negative) and each AO step is a
coordinate-wise global minimisation (assignment, closed-form prototype update,
convex weight subproblem), so ``J`` is non-increasing and converges.
``J_history_`` records this trajectory.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .hedge_algebra import HedgeAlgebra


class HAClustering:
    """Entropy-Weighted Adaptive Hedge Algebra Clustering.

    Parameters
    ----------
    n_clusters:
        Number of clusters ``K``.
    theta, alpha:
        Hedge Algebra parameters (used both to build the SQM for prototype
        initialisation/decoding and, with ``ha``, to define the semantic space).
        Ignored if an explicit ``ha`` is supplied.
    entropy_gamma:
        Entropy regularisation strength. ``None`` or ``0`` disables feature
        weighting (uniform weights -> plain HA clustering); ``> 0`` enables the
        adaptive entropy weighting (EWA-HAC). Larger values push the weights
        towards uniform.
    init:
        Prototype initialisation: ``"sqm"`` (HA-driven, deterministic; default),
        ``"quantile"``, or ``"kmeans++"``.
    max_iter, tol:
        Iteration budget and convergence tolerance on prototype shift.
    decode_depth:
        Maximum hedge depth of the term set used to decode prototypes into
        linguistic labels.
    feature_names:
        Optional names used in :attr:`cluster_descriptions_`.
    random_state:
        Seed for the stochastic ``kmeans++`` initialisation only.
    ha:
        An explicit :class:`HedgeAlgebra`. If ``None``, one is built from
        ``theta``/``alpha`` with the canonical ``{Little, Very}`` hedges.

    Attributes
    ----------
    cluster_centers_ : ndarray (K, F)
        Prototypes in the semantic space ``[0, 1]``.
    labels_ : ndarray (N,)
    feature_weights_ : ndarray (F,)
    objective_ : float
        Final value of ``J``.
    J_history_ : list of float
        ``J`` after each AO sweep (monotonically non-increasing).
    cluster_descriptions_ : list of dict
        Per-cluster ``{feature_name: linguistic_phrase}``.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        theta: float = 0.5,
        alpha: float = 0.5,
        entropy_gamma: Optional[float] = None,
        sqm_lambda: float = 0.0,
        weighting: str = "entropy",
        init: str = "sqm",
        max_iter: int = 100,
        tol: float = 1e-5,
        decode_depth: int = 2,
        feature_names: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        ha: Optional[HedgeAlgebra] = None,
        track_history: bool = False,
    ) -> None:
        if init not in ("sqm", "quantile", "kmeans++"):
            raise ValueError("init must be 'sqm', 'quantile' or 'kmeans++'")
        if sqm_lambda < 0:
            raise ValueError("sqm_lambda must be >= 0")
        if weighting not in ("entropy", "discriminative"):
            raise ValueError("weighting must be 'entropy' or 'discriminative'")
        self.n_clusters = n_clusters
        self.theta = theta
        self.alpha = alpha
        self.entropy_gamma = entropy_gamma
        self.sqm_lambda = sqm_lambda
        self.weighting = weighting
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.decode_depth = decode_depth
        self.feature_names = feature_names
        self.random_state = random_state
        self.ha = ha if ha is not None else HedgeAlgebra(theta=theta, alpha=alpha)
        self.track_history = track_history

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _semantize(self, X: np.ndarray) -> np.ndarray:
        """Map each column of ``X`` onto the semantic axis ``[0, 1]``."""
        S = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            S[:, j] = self.ha.semantize(X[:, j], self._lo[j], self._hi[j])
        return S

    def _init_centers(self, S: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        K, F = self.n_clusters, S.shape[1]
        if self.init == "sqm":
            # K linguistic terms spread evenly (by SQM rank) across [0, 1];
            # each feature is initialised at the same K semantic values.
            depth = 1
            terms = self.ha.generate_terms(depth)
            while len(terms) < K:
                depth += 1
                terms = self.ha.generate_terms(depth)
            nus = np.array([self.ha.nu(t) for t in terms])
            idx = np.linspace(0, len(nus) - 1, K).round().astype(int)
            return np.tile(nus[idx][:, None], (1, F))
        if self.init == "quantile":
            qs = np.linspace(1.0 / (K + 1), K / (K + 1), K)
            return np.quantile(S, qs, axis=0)
        # kmeans++ on the semantic data
        centers = [S[rng.integers(len(S))]]
        for _ in range(1, K):
            d2 = np.min(
                [np.sum((S - c) ** 2, axis=1) for c in centers], axis=0
            )
            probs = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(S[rng.choice(len(S), p=probs)])
        return np.array(centers)

    def _weighted_sqdist(self, S: np.ndarray, centers: np.ndarray, w: np.ndarray) -> np.ndarray:
        diff = S[:, None, :] - centers[None, :, :]      # (N, K, F)
        return np.sum(w * diff ** 2, axis=2)            # (N, K)

    def _dispersion(self, S: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Per-feature dispersion ``D_j = sum_k sum_{i in C_k} (S_ij - SC_kj)^2``."""
        D = np.zeros(S.shape[1])
        for k in range(self.n_clusters):
            idx = labels == k
            if np.any(idx):
                D += np.sum((S[idx] - centers[k]) ** 2, axis=0)
        return D

    def _between_dispersion(self, S: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Per-feature between-cluster dispersion ``B_j = sum_k n_k (SC_kj - Sbar_j)^2``."""
        Sbar = S.mean(axis=0)
        B = np.zeros(S.shape[1])
        for k in range(self.n_clusters):
            n_k = int(np.sum(labels == k))
            if n_k > 0:
                B += n_k * (centers[k] - Sbar) ** 2
        return B

    def _sqm_penalty(self, centers: np.ndarray) -> float:
        """SQM regularization: sqm_lambda · Σ_k Σ_j min_t (SC_kj - ν(t))^2."""
        if self.sqm_lambda == 0.0:
            return 0.0
        # diff shape: (K, F, T)
        diff = centers[:, :, None] - self._sqm_nus[None, None, :]
        return self.sqm_lambda * float(np.sum(np.min(diff ** 2, axis=2)))

    def _objective(self, S, labels, centers, w) -> float:
        dist = np.sum(self._weighted_sqdist(S, centers, w)[np.arange(len(S)), labels])
        nz = w > 0
        entropy = float(np.sum(w[nz] * np.log(w[nz])))
        return dist + self._gamma_eff * entropy + self._sqm_penalty(centers)

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "HAClustering":
        X = np.asarray(X, dtype=float)
        N, F = X.shape
        rng = np.random.default_rng(self.random_state)

        # Semantization (store bounds for predict()).
        self._lo = X.min(axis=0)
        self._hi = X.max(axis=0)
        S = self._semantize(X)
        self.semantic_data_ = S

        # Fixed, data-driven entropy scale -> keeps J well-defined and stable.
        scale = float(np.mean(np.sum((S - S.mean(axis=0)) ** 2, axis=0)))
        g = self.entropy_gamma if self.entropy_gamma else 0.0
        self._gamma_eff = g * scale if g > 0 else 0.0
        self._weighting = g > 0

        # Cache SQM values for regularization and convergence proof.
        self._sqm_nus = self.ha.sqm_values(self.decode_depth)

        centers = self._init_centers(S, rng)
        w = np.ones(F) / F

        self.J_history_: List[float] = []
        self.history_: List[dict] = []
        labels = None
        for _ in range(self.max_iter):
            # Step 1 - assignment; ties are resolved in favour of the incumbent
            # cluster (Proposition 2), otherwise by the lowest index.
            dist = self._weighted_sqdist(S, centers, w)
            new_labels = np.argmin(dist, axis=1)
            if labels is not None:
                rows = np.arange(N)
                keep = dist[rows, labels] <= dist[rows, new_labels]
                new_labels[keep] = labels[keep]
            n_changed = N if labels is None else int(np.sum(new_labels != labels))
            labels = new_labels

            # Step 2 - prototype update.
            # With sqm_lambda > 0: closed-form minimiser of
            #   sum_i w_j (S_ij - SC_kj)^2 + sqm_lambda (SC_kj - nu*_kj)^2
            # where nu*_kj is the nearest SQM value to the *previous* prototype
            # (lagged anchor keeps the update a strict descent step).
            # With sqm_lambda = 0: reduces to the semantic mean.
            new_centers = centers.copy()
            # Pre-compute nearest SQM anchor for each current prototype coordinate.
            if self.sqm_lambda > 0:
                nu_star = self._sqm_nus[
                    np.argmin(np.abs(centers[:, :, None] - self._sqm_nus[None, None, :]), axis=2)
                ]  # (K, F)
            for k in range(self.n_clusters):
                idx = labels == k
                n_k = int(idx.sum())
                if n_k > 0:
                    mean_k = S[idx].mean(axis=0)  # (F,)
                    if self.sqm_lambda > 0:
                        # Weighted average of data mean and SQM anchor.
                        denom = w * n_k + self.sqm_lambda          # (F,)
                        new_centers[k] = (w * n_k * mean_k + self.sqm_lambda * nu_star[k]) / denom
                    else:
                        new_centers[k] = mean_k
                else:
                    far = np.argmax(np.min(self._weighted_sqdist(S, centers, w), axis=1))
                    new_centers[k] = S[far]

            # Step 3 - weight update (closed-form, scale-stable).
            if self._weighting:
                W = self._dispersion(S, labels, new_centers)
                if self.weighting == "entropy":
                    # EWA-HAC: reward low within-cluster dispersion.
                    logits = -(W - W.min()) / self._gamma_eff
                else:
                    # Semantic Discriminative Weighting (SDW): reward high
                    # between- minus within-cluster dispersion, so channels that
                    # do not separate the clusters (e.g. nuisance/artefact maps
                    # amplified by min-max semantization) are down-weighted.
                    B = self._between_dispersion(S, labels, new_centers)
                    score = B - W
                    logits = (score - score.max()) / self._gamma_eff
                w_new = np.exp(logits)
                w = w_new / w_new.sum()

            self.J_history_.append(self._objective(S, labels, new_centers, w))

            shift = np.max(np.abs(new_centers - centers))
            if self.track_history:
                nz = w > 0
                entropy = float(-np.sum(w[nz] * np.log(w[nz])))
                self.history_.append({
                    "J": self.J_history_[-1],
                    "max_shift": float(shift),
                    "weight_entropy": entropy,
                    "effective_features": float(np.exp(entropy)),
                    "anchor_distance": self.anchor_distance(new_centers),
                    "n_changed": n_changed,
                    "centers": new_centers.copy(),
                })
            centers = new_centers
            if shift < self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.feature_weights_ = w
        self.objective_ = self.J_history_[-1]
        self.n_iter_ = len(self.J_history_)
        self.cluster_descriptions_ = self._describe(centers)
        return self

    def anchor_distance(self, centers: Optional[np.ndarray] = None) -> float:
        """Mean distance ``|SC_kj - nu*_kj|`` from prototypes to their nearest SQM anchor."""
        C = self.cluster_centers_ if centers is None else centers
        d = np.abs(C[:, :, None] - self._sqm_nus[None, None, :])
        return float(np.mean(np.min(d, axis=2)))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        S = self._semantize(X)
        return np.argmin(
            self._weighted_sqdist(S, self.cluster_centers_, self.feature_weights_), axis=1
        )

    # ------------------------------------------------------------------ #
    # Interpretability
    # ------------------------------------------------------------------ #
    def _describe(self, centers: np.ndarray) -> List[dict]:
        F = centers.shape[1]
        names = self.feature_names if self.feature_names is not None else [f"f{j}" for j in range(F)]
        out = []
        for k in range(self.n_clusters):
            out.append(
                {names[j]: self.ha.label(centers[k, j], max_depth=self.decode_depth) for j in range(F)}
            )
        return out
