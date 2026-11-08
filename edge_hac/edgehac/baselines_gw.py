"""Baselines of GW-HAC that compute with fuzzy sets or other linguistic / ordinal models (RESEARCH_PLAN.md, B2).

* Fuzzy-linguistic clustering: every feature is described by T triangular fuzzy labels forming a Ruspini partition
  of [0, 1] (memberships sum to 1), centred uniformly or at training quantiles. A sample becomes the concatenated
  membership vector; it is clustered by k-means or fuzzy c-means; a prototype is decoded to the label of maximal
  mean membership. This is "computing with words via fuzzy sets": membership functions, then defuzzification /
  linguistic approximation.
* 2-tuple k-means on an ordinal term set {s_0..s_g}: k-means on Delta^-1 values (the indices), with prototypes reported
  as 2-tuples (s_round(c), c - round(c)) (Herrera & Martinez 2000).
* Gower-PAM: k-medoids (FasterPAM) on Gower's distance with ordinal attributes treated as range-normalised ranks.
* k-means on ranks: ordinal codes used as numbers.
* IMM / ExKMC threshold trees: the authors' library ``ExKMC`` on top of k-means.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


# ----------------------------------------------------------------------------------------------------
# Ruspini triangular partitions
# ----------------------------------------------------------------------------------------------------
def ruspini_centres(T: int, S: np.ndarray | None = None) -> np.ndarray:
    """(F, T) centres of the triangular labels: uniform on [0, 1] (S None, F = 1) or at training quantiles."""
    if S is None:
        return np.linspace(0.0, 1.0, T)[None, :]
    qs = np.linspace(0.0, 1.0, T)
    C = np.quantile(S, qs, axis=0).T                      # (F, T)
    for j in range(C.shape[0]):                           # strictly increasing centres (ties from discrete data)
        for t in range(1, T):
            if C[j, t] <= C[j, t - 1]:
                C[j, t] = C[j, t - 1] + 1e-9
    return C


def memberships(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(N, F, T) Ruspini memberships of S (N, F) for centres C (1 or F, T); each (i, j) row sums to 1."""
    N, F = S.shape
    C = np.broadcast_to(C, (F, C.shape[1]))
    T = C.shape[1]
    U = np.zeros((N, F, T))
    for j in range(F):
        s = np.clip(S[:, j], C[j, 0], C[j, -1])
        t = np.clip(np.searchsorted(C[j], s, side="right") - 1, 0, T - 2)
        lo, hi = C[j, t], C[j, t + 1]
        w = (s - lo) / (hi - lo)
        U[np.arange(N), j, t] = 1.0 - w
        U[np.arange(N), j, t + 1] = w
    return U


@dataclass
class FuzzyLinguisticFit:
    labels: np.ndarray            # (N,)
    words: np.ndarray             # (K, F) decoded label index of each prototype (max mean membership)
    n_params_per_feature: int     # parameters of the label system: 3 per triangular label


def fuzzy_linguistic(S: np.ndarray, K: int, T: int, method: str = "kmeans", placement: str = "uniform",
                     seed: int = 0, m: float = 2.0) -> FuzzyLinguisticFit:
    """Cluster membership vectors; ``method`` in {"kmeans", "fcm"}, ``placement`` in {"uniform", "quantile"}."""
    C = ruspini_centres(T) if placement == "uniform" else ruspini_centres(T, S)
    U = memberships(S, C)
    Z = U.reshape(len(S), -1)
    if method == "kmeans":
        km = KMeans(K, n_init=10, random_state=seed).fit(Z)
        labels, cent = km.labels_, km.cluster_centers_
    elif method == "fcm":
        import skfuzzy
        cntr, u, *_ = skfuzzy.cluster.cmeans(Z.T, K, m, error=1e-5, maxiter=300, seed=seed)
        labels, cent = u.argmax(0), cntr
    else:
        raise ValueError(method)
    words = cent.reshape(K, S.shape[1], T).argmax(2)
    return FuzzyLinguisticFit(labels=labels, words=words, n_params_per_feature=3 * T)


def fuzzy_linguistic_predict(S: np.ndarray, fit_centres: np.ndarray, T: int, C: np.ndarray) -> np.ndarray:
    Z = memberships(S, C).reshape(len(S), -1)
    return ((Z[:, None, :] - fit_centres[None]) ** 2).sum(2).argmin(1)


# ----------------------------------------------------------------------------------------------------
# Ordinal / linguistic baselines
# ----------------------------------------------------------------------------------------------------
def two_tuple_kmeans(codes: np.ndarray, levels: np.ndarray, K: int, seed: int = 0):
    """k-means on Delta^-1 values normalised by the granularity g_j = levels_j - 1; returns labels and 2-tuples."""
    g = np.maximum(np.asarray(levels) - 1, 1)
    Z = codes / g
    km = KMeans(K, n_init=10, random_state=seed).fit(Z)
    beta = km.cluster_centers_ * g
    idx = np.rint(beta).astype(int)
    return km.labels_, (idx, beta - idx)


def gower_pam(codes: np.ndarray, levels: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """k-medoids on Gower's distance; ordinal ranks normalised by their range (Podani's treatment)."""
    import kmedoids
    g = np.maximum(np.asarray(levels) - 1, 1)
    Z = codes / g
    D = np.abs(Z[:, None, :] - Z[None, :, :]).mean(2)           # Gower with all-ordinal attributes
    return np.asarray(kmedoids.fasterpam(D, K, random_state=seed, n_cpu=1).labels)


def kmeans_ranks(codes: np.ndarray, levels: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    g = np.maximum(np.asarray(levels) - 1, 1)
    return KMeans(K, n_init=10, random_state=seed).fit_predict(codes / g)


def kmodes_labels(codes: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    from kmodes.kmodes import KModes
    return KModes(n_clusters=K, init="Huang", n_init=5, random_state=seed).fit_predict(codes)


def threshold_tree(X: np.ndarray, K: int, kind: str = "IMM", seed: int = 0):
    """IMM (K leaves) or ExKMC (2K leaves) via the authors' library, on k-means; returns labels and tree depth."""
    from ExKMC.Tree import Tree
    km = KMeans(K, n_init=10, random_state=seed).fit(X)
    leaves = K if kind == "IMM" else 2 * K
    tree = Tree(k=K, max_leaves=leaves, base_tree="IMM", random_state=seed)
    tree.fit(X, km)
    return tree.predict(X), tree
