"""Edge-relevant comparison methods, all operating on the same semantic coordinates ``S`` in [0, 1].

* :func:`kmeans_float`     Lloyd k-means (float reference, scikit-learn)
* :class:`PTQKMeans`       post-training b-bit uniform quantization of k-means (inputs b bits, centroids 8 bits),
                           integer nearest-centroid  -- the standard TinyML recipe (Jacob et al. 2018)
* :class:`IMMTree`         Iterative Mistake Minimization threshold tree (Moshkovitz et al. 2020), integer
                           thresholds on the B-bit input codes; k leaves, one per k-means centre
* :func:`fcm`              fuzzy c-means (float), for context only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.cluster import KMeans


def kmeans_float(S: np.ndarray, K: int, seed: int = 0):
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(S)
    return km.cluster_centers_, km.labels_


def nearest_centroid(S: np.ndarray, C: np.ndarray, chunk: int = 4096):
    """Float nearest centroid (lowest index on ties), labels and margin of squared distances."""
    N = len(S)
    lab = np.empty(N, dtype=np.int64)
    mar = np.empty(N)
    for i in range(0, N, chunk):
        d = ((S[i:i + chunk, None, :] - C[None]) ** 2).sum(axis=2)
        lab[i:i + chunk] = d.argmin(axis=1)
        srt = np.sort(d, axis=1)
        mar[i:i + chunk] = srt[:, 1] - srt[:, 0]
    return lab, mar


# ------------------------------------------------------------------------------------------------
@dataclass
class PTQKMeans:
    """Uniform post-training quantization of nearest-centroid inference."""

    centers_q: np.ndarray            # (K, F) uint8 (8-bit centroids on [0, 255])
    in_bits: int

    @classmethod
    def from_float(cls, C: np.ndarray, in_bits: int = 8) -> "PTQKMeans":
        return cls(centers_q=np.rint(np.clip(C, 0, 1) * 255).astype(np.int64), in_bits=in_bits)

    def codes(self, S: np.ndarray) -> np.ndarray:
        top = (1 << self.in_bits) - 1
        return np.rint(np.clip(S, 0, 1) * top).astype(np.int64)

    def scores(self, S: np.ndarray, chunk: int = 4096) -> np.ndarray:
        x = self.codes(S)
        # bring b-bit codes onto the 8-bit scale of the centroids with an integer shift/multiply
        top = (1 << self.in_bits) - 1
        x8 = (x * 255 + top // 2) // top                       # integer rescale (rounding), exact in int arithmetic
        out = np.empty((len(S), len(self.centers_q)), dtype=np.int64)
        for i in range(0, len(S), chunk):
            d = x8[i:i + chunk, None, :] - self.centers_q[None]
            out[i:i + chunk] = (d * d).sum(axis=2)
        return out

    def predict(self, S: np.ndarray):
        sc = self.scores(S)
        lab = sc.argmin(axis=1)
        srt = np.sort(sc, axis=1)
        return lab, (srt[:, 1] - srt[:, 0])


# ------------------------------------------------------------------------------------------------
@dataclass
class IMMTree:
    """Threshold tree with ``K`` leaves built by IMM; thresholds live in the B-bit code domain."""

    feat: List[int] = field(default_factory=list)       # per node: feature index (internal) or -1
    thr: List[int] = field(default_factory=list)        # code threshold: go right iff code >= thr
    left: List[int] = field(default_factory=list)
    right: List[int] = field(default_factory=list)
    leaf: List[int] = field(default_factory=list)       # cluster id for leaves, -1 otherwise
    B: int = 12

    @classmethod
    def fit(cls, S: np.ndarray, centers: np.ndarray, labels: np.ndarray, B: int = 12) -> "IMMTree":
        t = cls(B=B)
        top = (1 << B) - 1
        t._build(S, centers, labels, np.arange(len(centers)), top)
        return t

    def _new(self):
        for a in (self.feat, self.thr, self.left, self.right, self.leaf):
            a.append(-1)
        return len(self.feat) - 1

    def _build(self, S, C, lab, cset, top) -> int:
        node = self._new()
        if len(cset) == 1:
            self.leaf[node] = int(cset[0])
            return node
        m = np.isin(lab, cset)
        Sx, lx = S[m], lab[m]
        Cc = C[cset]
        best = None
        for j in range(C.shape[1]):
            cj = np.unique(Cc[:, j])
            if len(cj) < 2:
                continue
            cand = 0.5 * (cj[1:] + cj[:-1])
            xj = Sx[:, j]
            cx = C[lx, j]
            for th in cand:
                mistakes = int(np.sum((xj >= th) != (cx >= th)))
                if best is None or mistakes < best[0]:
                    best = (mistakes, j, th)
        if best is None:                                  # identical centres on every feature
            half = len(cset) // 2
            thr_code, j = 0, 0
            lcs, rcs = cset[:half], cset[half:]
        else:
            _, j, th = best
            thr_code = int(np.clip(np.ceil(th * top), 1, top))
            side = Cc[:, j] >= thr_code / top
            lcs, rcs = cset[~side], cset[side]
            if len(lcs) == 0 or len(rcs) == 0:            # rounding collapsed the cut
                half = len(cset) // 2
                lcs, rcs = cset[:half], cset[half:]
        self.feat[node], self.thr[node] = int(j), int(thr_code)
        self.left[node] = self._build(S, C, lab, lcs, top)
        self.right[node] = self._build(S, C, lab, rcs, top)
        return node

    def predict_codes(self, codes: np.ndarray):
        """Route integer codes; returns labels and the number of comparisons per sample."""
        feat, thr = np.array(self.feat), np.array(self.thr)
        left, right, leaf = np.array(self.left), np.array(self.right), np.array(self.leaf)
        N = len(codes)
        node = np.zeros(N, dtype=np.int64)
        steps = np.zeros(N, dtype=np.int64)
        active = leaf[node] < 0
        while active.any():
            idx = np.where(active)[0]
            f = feat[node[idx]]
            go_r = codes[idx, f] >= thr[node[idx]]
            node[idx] = np.where(go_r, right[node[idx]], left[node[idx]])
            steps[idx] += 1
            active = leaf[node] < 0
        return leaf[node], steps

    @property
    def n_nodes(self) -> int:
        return len(self.feat)

    def depth(self) -> int:
        def d(n):
            return 0 if self.leaf[n] >= 0 else 1 + max(d(self.left[n]), d(self.right[n]))
        return d(0)

    def memory_bytes(self) -> int:
        """Internal node: feature (1-2 B) + threshold (2 B) + 2 child offsets (1 B each if < 256 nodes)."""
        internal = sum(1 for l in self.leaf if l < 0)
        F_bytes = 1 if max(self.feat) < 256 else 2
        child = 1 if self.n_nodes < 256 else 2
        return internal * (F_bytes + 2 + 2 * child)


def fcm(S: np.ndarray, K: int, m: float = 2.0, iters: int = 100, tol: float = 1e-5, seed: int = 0):
    """Plain fuzzy c-means (float); returns centres and hard labels."""
    rng = np.random.default_rng(seed)
    N = len(S)
    U = rng.random((N, K)); U /= U.sum(1, keepdims=True)
    for _ in range(iters):
        W = U ** m
        C = (W.T @ S) / np.maximum(W.sum(0)[:, None], 1e-12)
        d = ((S[:, None, :] - C[None]) ** 2).sum(2) + 1e-12
        inv = d ** (-1.0 / (m - 1))
        Un = inv / inv.sum(1, keepdims=True)
        if np.max(np.abs(Un - U)) < tol:
            U = Un
            break
        U = Un
    return C, U.argmax(1)
