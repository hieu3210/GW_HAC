"""Free-Energy Hedge-Algebra Clustering (FE-HAC).

A deterministic-annealing / rate-distortion generalisation of EWA-HAC.  The model
minimises a **free energy** over soft (fuzzy) assignments ``u``, prototypes ``SC``
and feature weights ``w`` on the Hedge-Algebra semantic axis::

    F = sum_i sum_k u_ik d_w(i, k)            # weighted semantic distortion
        + T  sum_i sum_k u_ik ln u_ik         # annealed partition entropy
        + g  sum_j w_j ln w_j                 # scale-stable weight entropy
        + lam sum_k sum_j min_t (SC_kj-nu_t)^2 # SQM linguistic-codebook term

At fixed temperature ``T`` the three blocks have closed-form updates:

* **assignment** (Gibbs):  ``u_ik = softmax_k(-d_w(i,k)/T)``;
* **prototype**: soft-membership weighted average of the data mean and the nearest
  SQM codeword (lagged anchor) -- generalises the EWA-HAC prototype update;
* **weights**:  ``w_j = softmax_j(-D_j/g)`` with the same scale-stable ``g``.

Deterministic annealing cools ``T`` geometrically.  As ``T -> 0`` the Gibbs
assignment becomes hard ``argmin`` and FE-HAC reduces to EWA-HAC (Proposition 2).
A **kernel** variant replaces ``d_w`` by ``1 - exp(-d_w / 2 sigma^2)`` for
non-convex geometry, and an **auto-K** routine (:func:`fehac_auto_k`) exploits the
phase-transition structure of the free energy to discover the number of clusters.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .hedge_algebra import HedgeAlgebra


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class FEHAC:
    """Free-Energy Hedge-Algebra Clustering (fixed ``K``).

    Parameters
    ----------
    n_clusters : int
    theta, alpha : float
        Hedge-Algebra parameters (build the SQM codebook + semantic space).
    entropy_gamma : float
        Scale-stable feature-weight entropy strength (``0`` -> uniform weights).
    sqm_lambda : float
        SQM linguistic-codebook attraction strength.
    kernel : bool
        If ``True``, use Gaussian-kernel distortion ``1 - exp(-d_w/2 sigma^2)``.
    sigma : float
        Gaussian-kernel bandwidth (only if ``kernel``).
    t0_factor, cooling, t_min_ratio : float
        Deterministic-annealing schedule: ``T0 = t0_factor * mean initial
        distortion``; ``T <- cooling * T`` until ``T < t_min_ratio * T0``.
    inner_iter : int
        Block-update sweeps per temperature.
    init : {"sqm", "kmeans++"}
    decode_depth : int
    """

    def __init__(
        self,
        n_clusters: int = 3,
        theta: float = 0.5,
        alpha: float = 0.5,
        entropy_gamma: Optional[float] = 1.0,
        sqm_lambda: float = 0.01,
        kernel: bool = False,
        sigma: float = 0.5,
        t0_factor: float = 0.2,
        cooling: float = 0.9,
        t_min_ratio: float = 1e-3,
        inner_iter: int = 20,
        tol: float = 1e-5,
        init: str = "sqm",
        decode_depth: int = 2,
        feature_names: Optional[Sequence[str]] = None,
        random_state: Optional[int] = None,
        ha: Optional[HedgeAlgebra] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.theta = theta
        self.alpha = alpha
        self.entropy_gamma = entropy_gamma
        self.sqm_lambda = sqm_lambda
        self.kernel = kernel
        self.sigma = sigma
        self.t0_factor = t0_factor
        self.cooling = cooling
        self.t_min_ratio = t_min_ratio
        self.inner_iter = inner_iter
        self.tol = tol
        self.init = init
        self.decode_depth = decode_depth
        self.feature_names = feature_names
        self.random_state = random_state
        self.ha = ha if ha is not None else HedgeAlgebra(theta=theta, alpha=alpha)

    # ------------------------------------------------------------------ #
    def _semantize(self, X: np.ndarray) -> np.ndarray:
        S = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            S[:, j] = self.ha.semantize(X[:, j], self._lo[j], self._hi[j])
        return S

    def _init_centers(self, S: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        K, F = self.n_clusters, S.shape[1]
        if self.init == "sqm":
            depth = 1
            terms = self.ha.generate_terms(depth)
            while len(terms) < K:
                depth += 1
                terms = self.ha.generate_terms(depth)
            nus = np.array([self.ha.nu(t) for t in terms])
            idx = np.linspace(0, len(nus) - 1, K).round().astype(int)
            return np.tile(nus[idx][:, None], (1, F))
        centers = [S[rng.integers(len(S))]]
        for _ in range(1, K):
            d2 = np.min([np.sum((S - c) ** 2, axis=1) for c in centers], axis=0)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(S[rng.choice(len(S), p=p)])
        return np.array(centers)

    def _sqdist(self, S, centers, w):
        diff = S[:, None, :] - centers[None, :, :]
        return np.sum(w * diff ** 2, axis=2)  # (N, K)

    def _distortion(self, S, centers, w):
        d = self._sqdist(S, centers, w)
        if self.kernel:
            return 1.0 - np.exp(-d / (2.0 * self.sigma ** 2))
        return d

    def _sqm_anchor(self, centers):
        idx = np.argmin(np.abs(centers[:, :, None] - self._sqm_nus[None, None, :]), axis=2)
        return self._sqm_nus[idx]

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray) -> "FEHAC":
        X = np.asarray(X, dtype=float)
        N, F = X.shape
        rng = np.random.default_rng(self.random_state)

        self._lo, self._hi = X.min(axis=0), X.max(axis=0)
        S = self._semantize(X)
        self.semantic_data_ = S

        scale = float(np.mean(np.sum((S - S.mean(axis=0)) ** 2, axis=0)))
        g = self.entropy_gamma if self.entropy_gamma else 0.0
        self._g = g * scale if g > 0 else 0.0
        self._weighting = g > 0
        self._sqm_nus = self.ha.sqm_values(self.decode_depth)

        centers = self._init_centers(S, rng).astype(float)
        w = np.ones(F) / F

        D0 = self._distortion(S, centers, w)
        T0 = self.t0_factor * float(np.mean(D0)) + 1e-12
        T_min = self.t_min_ratio * T0

        self.F_history_: List[float] = []
        u = np.full((N, self.n_clusters), 1.0 / self.n_clusters)
        T = T0
        while T >= T_min:
            for _ in range(self.inner_iter):
                prev = centers.copy()
                # E-step: Gibbs (fuzzy) assignment.
                D = self._distortion(S, centers, w)
                u = _softmax(-D / T, axis=1)              # (N, K)
                # M-step: prototypes (soft-membership + lagged SQM anchor).
                nu_star = self._sqm_anchor(centers)
                if self.kernel:
                    Ksim = np.exp(-self._sqdist(S, centers, w) / (2.0 * self.sigma ** 2))
                    a = u * Ksim                          # (N, K) kernel-weighted resp.
                else:
                    a = u
                resp = a.sum(axis=0)                      # (K,)
                num = a.T @ S                             # (K, F)
                denom = (w[None, :] * resp[:, None]) + self.sqm_lambda
                centers = (w[None, :] * num + self.sqm_lambda * nu_star) / denom
                # weight update (soft dispersion).
                if self._weighting:
                    diff2 = (S[:, None, :] - centers[None, :, :]) ** 2
                    D_j = np.einsum("nk,nkj->j", u, diff2)
                    w = _softmax(-(D_j - D_j.min())[None, :] / self._g, axis=1).ravel()
                self.F_history_.append(self._free_energy(S, u, centers, w, T))
                if np.max(np.abs(centers - prev)) < self.tol:
                    break
            T *= self.cooling

        # Hard partition at the T -> 0 limit (EWA-HAC-style decoding).
        D = self._distortion(S, centers, w)
        labels = np.argmin(D, axis=1)
        self.cluster_centers_ = centers
        self.memberships_ = u
        self.labels_ = labels
        self.feature_weights_ = w
        self.cluster_descriptions_ = self._describe(centers)
        return self

    def _free_energy(self, S, u, centers, w, T):
        D = self._distortion(S, centers, w)
        dist = float(np.sum(u * D))
        nz = u > 0
        part_ent = float(np.sum(u[nz] * np.log(u[nz])))
        wterm = 0.0
        if self._weighting:
            wnz = w > 0
            wterm = self._g * float(np.sum(w[wnz] * np.log(w[wnz])))
        sqm = self.sqm_lambda * float(
            np.sum(np.min((centers[:, :, None] - self._sqm_nus[None, None, :]) ** 2, axis=2))
        )
        return dist + T * part_ent + wterm + sqm

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _describe(self, centers):
        F = centers.shape[1]
        names = self.feature_names if self.feature_names is not None else [f"f{j}" for j in range(F)]
        return [
            {names[j]: self.ha.label(centers[k, j], max_depth=self.decode_depth) for j in range(F)}
            for k in range(self.n_clusters)
        ]


# ---------------------------------------------------------------------------- #
# Automatic K via mass-constrained deterministic annealing (phase transitions).
# ---------------------------------------------------------------------------- #
def fehac_auto_k(
    X: np.ndarray,
    theta: float = 0.5,
    alpha: float = 0.5,
    k_max: int = 12,
    cooling: float = 0.95,
    inner_iter: int = 40,
    split_eps: float = 1e-2,
    merge_tol: float = 5e-3,
    random_state: Optional[int] = None,
    return_centers: bool = False,
):
    """Discover the number of clusters by deterministic-annealing phase transitions.

    Mass-constrained deterministic annealing (Rose): starting from one effective
    centroid, the temperature is cooled geometrically; the centroid with the
    highest critical temperature ``T_c = 2 lambda_max(Sigma_k)`` that exceeds the
    current ``T`` splits (one second-order phase transition per step), and
    near-coincident centroids are merged.  ``K`` is selected as the resolution that
    persists over the widest temperature interval (the most stable plateau of the
    ``K_eff(T)`` trajectory), a principled, knee-free criterion.  Returns
    ``(labels, k_hat, history)`` (and centers if requested).
    """
    ha = HedgeAlgebra(theta=theta, alpha=alpha)
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    S = np.empty_like(X)
    for j in range(X.shape[1]):
        S[:, j] = ha.semantize(X[:, j], lo[j], hi[j])
    N, F = S.shape

    centers = S.mean(axis=0, keepdims=True).copy()
    priors = np.array([1.0])

    cov = np.cov(S.T)
    lam0 = float(np.max(np.linalg.eigvalsh(np.atleast_2d(cov)))) if F > 1 else float(np.var(S))
    T = 4.0 * lam0
    T_min = 1e-4 * lam0
    history = []           # (logT, K_eff)
    snapshots = {}          # K_eff -> (centers, accumulated log-T width)

    while T >= T_min:
        # EM to (approximate) equilibrium at this temperature.
        for _ in range(inner_iter):
            d = np.sum((S[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            u = _softmax(np.log(priors[None, :] + 1e-12) - d / T, axis=1)
            resp = u.sum(axis=0)
            priors = resp / resp.sum()
            new = (u.T @ S) / np.clip(resp[:, None], 1e-12, None)
            shift = np.max(np.abs(new - centers))
            centers = new
            if shift < 1e-6:
                break

        # Merge near-coincident centroids.
        keep = np.ones(len(centers), dtype=bool)
        for i in range(len(centers)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(centers)):
                if keep[j] and np.sum((centers[i] - centers[j]) ** 2) < merge_tol:
                    keep[j] = False
        if not keep.all():
            centers, priors = centers[keep], priors[keep] / priors[keep].sum()

        Keff = len(centers)
        dlogT = -np.log(cooling)
        snap = snapshots.get(Keff)
        snapshots[Keff] = (centers.copy(), (snap[1] if snap else 0.0) + dlogT)
        history.append((np.log(T), Keff))

        if Keff >= k_max:
            break  # stop before the cap dominates the plateau statistics

        # Split one cluster: the largest critical temperature above T, among
        # clusters with enough mass to support a genuine phase transition.
        min_mass = max(2 * F + 2, N / (4 * k_max))
        d = np.sum((S[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        u = _softmax(np.log(priors[None, :] + 1e-12) - d / T, axis=1)
        best_k, best_lam, best_v = -1, 0.0, None
        for k in range(Keff):
            uk = u[:, k]
            m = uk.sum()
            if m < min_mass:
                continue
            Sig = (uk[:, None] * (S - centers[k])).T @ (S - centers[k]) / m
            ev, evec = np.linalg.eigh(Sig)
            if float(ev[-1]) > best_lam:
                best_lam, best_k, best_v = float(ev[-1]), k, evec[:, -1]
        if best_k >= 0 and T < 2.0 * best_lam:
            c = centers[best_k].copy()
            centers = np.vstack([centers, c - split_eps * best_v])
            centers[best_k] = c + split_eps * best_v
            priors = np.append(priors, priors[best_k] / 2)
            priors[best_k] /= 2
            priors = priors / priors.sum()
        T *= cooling

    # Select K = most stable resolution (widest log-T plateau), ignoring K=1.
    candidates = {k: v for k, v in snapshots.items() if k >= 2} or snapshots
    k_hat = max(candidates, key=lambda k: candidates[k][1])
    centers = candidates[k_hat][0]
    d = np.sum((S[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d, axis=1)
    _, labels = np.unique(labels, return_inverse=True)
    k_hat = len(np.unique(labels))
    if return_centers:
        return labels, k_hat, history, centers
    return labels, k_hat, history
