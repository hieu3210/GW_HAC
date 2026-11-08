"""Baseline clustering algorithms with a uniform ``fit_predict(X)`` interface.

Included baselines:

* :class:`KMeansBaseline` -- Lloyd's k-means (scikit-learn).
* :class:`FCM` -- standard Fuzzy C-Means (scikit-fuzzy).
* :class:`KernelFCM` -- Gaussian-kernel FCM, distance ``d = 1 - K(x, v)``.
* :class:`EntropyFCM` -- entropy-weighted FCM (Euclidean counterpart of EWA-HAC).
* :class:`SpectralBaseline` -- Spectral Clustering with RBF affinity (scikit-learn).
* :class:`HDBSCANBaseline` -- density-based HDBSCAN; noise points (-1) are
  reassigned to the nearest non-noise cluster centroid so that the output is
  always a complete crisp partition over K clusters.

All expose ``labels_`` after ``fit_predict``, so the benchmark can treat every
method identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from skfuzzy.cluster import cmeans


class KMeansBaseline:
    def __init__(self, n_clusters=3, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit_predict(self, X):
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        self.labels_ = km.fit_predict(X)
        self.cluster_centers_ = km.cluster_centers_
        self.n_iter_ = int(km.n_iter_)  # Lloyd iterations of the retained run
        return self.labels_


class FCM:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        seed = self.random_state if self.random_state is not None else None
        cntr, u, _, _, _, n_iter, _ = cmeans(
            np.asarray(X).T, c=self.n_clusters, m=self.m,
            error=self.tol, maxiter=self.max_iter, seed=seed,
        )
        self.n_iter_ = int(n_iter)
        self.cluster_centers_ = cntr
        self.labels_ = np.argmax(u, axis=0)
        return self.labels_


class EntropyFCM:
    """Entropy-Weighted Fuzzy C-Means (EW-FCM): Euclidean analogue of EWA-HAC."""

    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-4, entropy_gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.entropy_gamma = entropy_gamma
        self.random_state = random_state

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        N, F = X.shape
        rng = np.random.default_rng(self.random_state)
        U = rng.dirichlet(np.ones(self.n_clusters), size=N).T
        W = np.ones(F) / F
        V = np.zeros((self.n_clusters, F))
        # Fixed, data-driven entropy scale (same stabilisation as EWA-HAC).
        scale = float(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=0)))
        g = max(self.entropy_gamma * scale, np.finfo(float).eps)

        self.n_iter_ = 0
        for _ in range(self.max_iter):
            self.n_iter_ += 1
            U_old = U.copy()
            Um = U ** self.m
            for k in range(self.n_clusters):
                V[k] = np.sum(Um[k, :, None] * X, axis=0) / np.sum(Um[k])
            D_ik = np.empty((self.n_clusters, N))
            for k in range(self.n_clusters):
                D_ik[k] = np.sum(W * (X - V[k]) ** 2, axis=1)
            D_ik = np.fmax(D_ik, np.finfo(float).eps)
            temp = D_ik ** (-1.0 / (self.m - 1))
            U = temp / np.sum(temp, axis=0, keepdims=True)

            Um = U ** self.m
            D_j = np.zeros(F)
            for k in range(self.n_clusters):
                D_j += np.sum(Um[k, :, None] * (X - V[k]) ** 2, axis=0)
            w_new = np.exp(-(D_j - D_j.min()) / g)
            W = w_new / w_new.sum()
            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.cluster_centers_ = V
        self.feature_weights_ = W
        self.labels_ = np.argmax(U, axis=0)
        return self.labels_


class KernelFCM:
    """Kernel Fuzzy C-Means with a Gaussian (RBF) kernel; ``d = 1 - K(x, v)``."""

    def __init__(self, n_clusters=3, m=2.0, gamma=0.1, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_predict(self, X):
        X = np.asarray(X, dtype=float)
        N, F = X.shape
        rng = np.random.default_rng(self.random_state)
        U = rng.dirichlet(np.ones(self.n_clusters), size=N).T
        V = np.zeros((self.n_clusters, F))
        for k in range(self.n_clusters):
            V[k] = np.sum((U[k] ** self.m)[:, None] * X, axis=0) / np.sum(U[k] ** self.m)

        self.n_iter_ = 0
        for _ in range(self.max_iter):
            self.n_iter_ += 1
            U_old = U.copy()
            Um = U ** self.m
            K_ik = np.vstack([rbf_kernel(X, V[k:k + 1], gamma=self.gamma).ravel() for k in range(self.n_clusters)])
            for k in range(self.n_clusters):
                weights = Um[k] * K_ik[k]
                V[k] = np.sum(weights[:, None] * X, axis=0) / np.sum(weights)
            K_ik = np.vstack([rbf_kernel(X, V[k:k + 1], gamma=self.gamma).ravel() for k in range(self.n_clusters)])
            D_ik = np.fmax(1.0 - K_ik, np.finfo(float).eps)
            temp = D_ik ** (-1.0 / (self.m - 1))
            U = temp / np.sum(temp, axis=0, keepdims=True)
            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.cluster_centers_ = V
        self.labels_ = np.argmax(U, axis=0)
        return self.labels_


class SpectralBaseline:
    """Spectral Clustering with RBF affinity (scikit-learn)."""

    def __init__(self, n_clusters=3, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit_predict(self, X):
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="rbf",
            gamma=self.gamma,
            random_state=self.random_state,
            n_init=10,
        )
        self.labels_ = sc.fit_predict(X)
        return self.labels_


class HDBSCANBaseline:
    """HDBSCAN density-based clustering.

    Noise points (label -1) are reassigned to the nearest cluster centroid so
    the output is always a complete crisp partition over the K largest clusters.
    """

    def __init__(self, n_clusters=3, min_cluster_size=15):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, X):
        import hdbscan

        X = np.asarray(X, dtype=float)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        raw = clusterer.fit_predict(X)

        # Keep only the K largest clusters; merge smaller ones into the nearest
        # large cluster so the output matches the expected number of clusters.
        valid = raw[raw >= 0]
        if len(valid) == 0:
            self.labels_ = np.zeros(len(X), dtype=int)
            return self.labels_

        counts = np.bincount(valid)
        top_k = np.argsort(counts)[::-1][: self.n_clusters]
        remap = {old: new for new, old in enumerate(sorted(top_k))}

        labels = np.full(len(X), -1, dtype=int)
        for i, lbl in enumerate(raw):
            if lbl in remap:
                labels[i] = remap[lbl]

        # Assign unassigned points to the nearest cluster centroid.
        if np.any(labels == -1):
            assigned = labels >= 0
            if assigned.any():
                centroids = np.array(
                    [X[labels == k].mean(axis=0) for k in range(len(top_k))]
                )
                for i in np.where(~assigned)[0]:
                    labels[i] = int(np.argmin(np.sum((centroids - X[i]) ** 2, axis=1)))
            else:
                labels[:] = 0

        self.labels_ = labels
        return self.labels_
