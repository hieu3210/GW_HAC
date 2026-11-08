"""W-HAC: clustering *directly on linguistic words* of per-feature hedge algebras (THEORY_WORD.md).

A numeric feature value is semantized to the nearest word of its vocabulary (classical SQM, specificity s);
every sample becomes a vector of words, every cluster prototype a vector of words, and all computation
(assignment, prototype update, prediction) uses only the SQM values nu_j(word) -- i.e. integer word indices and
small tables. The fuzziness parameters (theta_j, alpha_j) in (0, 1) are free and are optimised by PSO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.metrics import silhouette_score

from .ha_classic import n_words, vocabulary
from .sqm import vocabulary as sqm_vocabulary
from .lattice import Lattice
from .pso import PSOResult, pso

EPS = 1e-12


# ----------------------------------------------------------------------------------------------------
# Vocabularies and semantization
# ----------------------------------------------------------------------------------------------------
@dataclass
class Vocab:
    """Per-feature vocabularies: ``V[j]`` = ascending SQM values, ``names[j]`` = word codes."""

    theta: np.ndarray            # (F,)
    alpha: np.ndarray            # (F,)
    spec: int
    variant: str = "lts"            # "lts" (legacy closed forms, M2b) or the consistent "little" / "rather" (sqm.py, GW-HAC)
    V: np.ndarray = field(init=False)           # (F, T)
    thr: np.ndarray = field(init=False)         # (F, T-1) cell boundaries used to semantize a value
    vmid: np.ndarray = field(init=False)        # (F, T-1) midpoints of consecutive SQM values (nearest-word rule)
    names: List[List[str]] = field(init=False)

    def __post_init__(self) -> None:
        F = len(self.theta)
        T = n_words(self.spec)
        self.V = np.empty((F, T))
        self.names = []
        cache = {}
        for j in range(F):
            key = (round(float(self.theta[j]), 12), round(float(self.alpha[j]), 12))
            if key not in cache:
                cache[key] = (vocabulary(key[0], key[1], self.spec) if self.variant == "lts"
                              else sqm_vocabulary(key[0], key[1], self.spec, self.variant))
            v, nm = cache[key]
            self.V[j] = v
            self.names.append(nm)
        self.vmid = 0.5 * (self.V[:, 1:] + self.V[:, :-1])
        self.thr = self.vmid.copy()

    @property
    def T(self) -> int:
        return self.V.shape[1]

    def lattices(self) -> List[Lattice]:
        return [Lattice(v=self.V[j].copy(), kind="words") for j in range(len(self.theta))]


class CustomVocab:
    """Non-HA word sets used as baselines: explicit embedding values ``V`` and semantization boundaries ``thr``."""

    def __init__(self, V: np.ndarray, thr: np.ndarray, names=None):
        self.V = np.asarray(V, float)
        self.thr = np.asarray(thr, float)
        self.vmid = 0.5 * (self.V[:, 1:] + self.V[:, :-1])
        F, T = self.V.shape
        self.theta = np.full(F, np.nan)
        self.alpha = np.full(F, np.nan)
        self.names = names or [[f"w{t}" for t in range(T)] for _ in range(F)]

    @property
    def T(self) -> int:
        return self.V.shape[1]

    def lattices(self) -> List[Lattice]:
        return [Lattice(v=self.V[j].copy(), kind="words") for j in range(len(self.V))]


def uniform_vocab(F: int, T: int) -> CustomVocab:
    """Equal-width words, embedding (t + 1/2)/T (ordinal, non-HA)."""
    v = (np.arange(T) + 0.5) / T
    return CustomVocab(np.tile(v, (F, 1)), np.tile(np.arange(1, T) / T, (F, 1)))


def quantile_vocab(S: np.ndarray, T: int) -> CustomVocab:
    """Equal-frequency words from training quantiles, ordinal embedding (t + 1/2)/T."""
    F = S.shape[1]
    qs = np.arange(1, T) / T
    thr = np.stack([np.quantile(S[:, j], qs) for j in range(F)])
    v = (np.arange(T) + 0.5) / T
    return CustomVocab(np.tile(v, (F, 1)), thr)


def make_vocab(theta, alpha, F: int, spec: int, variant: str = "lts") -> Vocab:
    th = np.full(F, theta) if np.isscalar(theta) else np.asarray(theta, float)
    al = np.full(F, alpha) if np.isscalar(alpha) else np.asarray(alpha, float)
    return Vocab(theta=th, alpha=al, spec=spec, variant=variant)


def semantize_words(S: np.ndarray, voc: Vocab) -> np.ndarray:
    """Word index (N, F) of semantic coordinates ``S`` in [0, 1]: nearest SQM value (ties -> upper word)."""
    N, F = S.shape
    q = np.empty((N, F), dtype=np.int32)
    shared = bool(np.all(voc.thr == voc.thr[0]))
    if shared:
        q[:] = np.searchsorted(voc.thr[0], S, side="right")
    else:
        for j in range(F):
            q[:, j] = np.searchsorted(voc.thr[j], S[:, j], side="right")
    return q


def embed(q: np.ndarray, voc: Vocab) -> np.ndarray:
    """SQM values of a word-index matrix (N, F)."""
    return voc.V[np.arange(q.shape[1])[None, :], q]


# ----------------------------------------------------------------------------------------------------
# Lloyd-type clustering in the word domain
# ----------------------------------------------------------------------------------------------------
def _dist(nuX: np.ndarray, nuP: np.ndarray, chunk: int = 2048, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Squared (optionally feature-weighted) SQM distances, shape (N, K)."""
    N = len(nuX)
    out = np.empty((N, len(nuP)))
    for i in range(0, N, chunk):
        d = nuX[i:i + chunk, None, :] - nuP[None]
        out[i:i + chunk] = np.einsum("nkf,nkf->nk", d, d) if w is None else np.einsum("nkf,f->nk", d * d, w)
    return out


def _nearest_word(mean_nu: np.ndarray, voc: Vocab) -> np.ndarray:
    """Word index of the vocabulary value nearest to ``mean_nu`` (K, F)."""
    K, F = mean_nu.shape
    out = np.empty((K, F), dtype=np.int32)
    for j in range(F):
        out[:, j] = np.searchsorted(voc.vmid[j], mean_nu[:, j], side="right")
    return out


def init_prototypes(K: int, F: int, T: int) -> np.ndarray:
    """Deterministic SQM-style initialisation: prototype k at vocabulary rank round(k (T-1)/(K-1)) in every feature."""
    ranks = np.rint(np.linspace(0, T - 1, K)).astype(np.int32) if K > 1 else np.array([T // 2], dtype=np.int32)
    return np.tile(ranks[:, None], (1, F))


def kmeanspp_prototypes(nuX: np.ndarray, voc: Vocab, K: int, rng: np.random.Generator) -> np.ndarray:
    idx = [int(rng.integers(len(nuX)))]
    d2 = ((nuX - nuX[idx[0]]) ** 2).sum(1)
    for _ in range(1, K):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        idx.append(int(rng.choice(len(nuX), p=p)))
        d2 = np.minimum(d2, ((nuX - nuX[idx[-1]]) ** 2).sum(1))
    return _nearest_word(nuX[idx], voc)


@dataclass
class WordFit:
    P: np.ndarray            # (K, F) prototype word indices
    labels: np.ndarray       # (N,)
    J: float                 # sum of squared SQM distances
    n_iter: int


def fit_words(q: np.ndarray, voc: Vocab, K: int, init: Optional[np.ndarray] = None, max_iter: int = 50,
              weights: Optional[np.ndarray] = None) -> WordFit:
    """Alternating minimisation of ``J = sum_i ||nu(x_i) - nu(P_c(i))||^2`` over assignments and prototype words.

    Prototype update (THEORY_WORD, Lemma W1): the word nearest to the mean SQM value of the members, feature-wise.
    Ties in the assignment go to the lowest index unless the incumbent cluster is tied (then it is kept), so the
    iteration cannot cycle and terminates in finitely many steps (Theorem W2).
    """
    N, F = q.shape
    nuX = embed(q, voc)
    P = init_prototypes(K, F, voc.T) if init is None else init.copy()
    labels = None
    J_prev = np.inf
    n = 0
    for n in range(1, max_iter + 1):
        nuP = voc.V[np.arange(F)[None, :], P]
        D = _dist(nuX, nuP, w=weights)
        new = np.argmin(D, axis=1)
        if labels is not None:
            rows = np.arange(N)
            keep = D[rows, labels] <= D[rows, new] + EPS
            new[keep] = labels[keep]
        J = float(D[np.arange(N), new].sum())
        changed = labels is None or not np.array_equal(new, labels)
        labels = new
        if not changed and J >= J_prev - 1e-15:
            break
        J_prev = J
        mean = np.empty((K, F))
        for k in range(K):
            m = labels == k
            if m.any():
                mean[k] = nuX[m].mean(0)
            else:                                         # empty cluster -> re-seed at the worst-fitted sample
                far = int(np.argmax(D[np.arange(N), labels]))
                mean[k] = nuX[far]
        P = _nearest_word(mean, voc)
        if not changed:
            break
    nuP = voc.V[np.arange(F)[None, :], P]
    D = _dist(nuX, nuP, w=weights)
    labels = np.argmin(D, axis=1)
    return WordFit(P=P, labels=labels, J=float(D[np.arange(N), labels].sum()), n_iter=n)


def fit_words_counts(q: np.ndarray, voc: Vocab, K: int, init: Optional[np.ndarray] = None, max_iter: int = 50,
                     weights: Optional[np.ndarray] = None) -> WordFit:
    """Same procedure as :func:`fit_words`, but the prototype update uses only the integer counters
    ``n[k, j, t]`` (number of members of cluster k whose feature j is word t): the mean SQM value of a cluster is
    ``sum_t n[k,j,t] nu_j(t) / n_k`` (THEORY_WORD, Prop. W4).  No raw sample is touched by the update, so a device
    can train from a stream with K*F*T counters.  Returns exactly the same partition as ``fit_words``."""
    N, F = q.shape
    T = voc.T
    nuX = embed(q, voc)
    P = init_prototypes(K, F, T) if init is None else init.copy()
    labels = None
    J_prev = np.inf
    n = 0
    fidx = np.arange(F)
    for n in range(1, max_iter + 1):
        nuP = voc.V[fidx[None, :], P]
        D = _dist(nuX, nuP, w=weights)
        new = np.argmin(D, axis=1)
        if labels is not None:
            rows = np.arange(N)
            keep = D[rows, labels] <= D[rows, new] + EPS
            new[keep] = labels[keep]
        J = float(D[np.arange(N), new].sum())
        changed = labels is None or not np.array_equal(new, labels)
        labels = new
        if not changed and J >= J_prev - 1e-15:
            break
        J_prev = J
        counts = np.zeros((K, F, T), dtype=np.int64)               # the only statistic kept for the update
        for j in range(F):
            np.add.at(counts[:, j, :], (labels, q[:, j]), 1)
        nk = counts.sum(axis=2)[:, :1]
        mean = np.einsum("kjt,jt->kj", counts, voc.V) / np.maximum(nk, 1)
        empty = nk[:, 0] == 0
        if empty.any():
            far = int(np.argmax(D[np.arange(N), labels]))
            mean[empty] = nuX[far]
        P = _nearest_word(mean, voc)
        if not changed:
            break
    D = _dist(nuX, voc.V[fidx[None, :], P], w=weights)
    labels = np.argmin(D, axis=1)
    return WordFit(P=P, labels=labels, J=float(D[np.arange(N), labels].sum()), n_iter=n)


def best_of(q: np.ndarray, voc: Vocab, K: int, n_init: int = 5, seed: int = 0, max_iter: int = 50,
            weights: Optional[np.ndarray] = None) -> WordFit:
    """Best (lowest J) of the deterministic SQM initialisation and ``n_init - 1`` k-means++-seeded runs."""
    best = fit_words(q, voc, K, max_iter=max_iter, weights=weights)
    nuX = embed(q, voc)
    rng = np.random.default_rng(seed)
    for _ in range(n_init - 1):
        f = fit_words(q, voc, K, init=kmeanspp_prototypes(nuX, voc, K, rng), max_iter=max_iter, weights=weights)
        if f.J < best.J:
            best = f
    return best


def predict_words(q: np.ndarray, voc: Vocab, P: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    F = q.shape[1]
    D = _dist(embed(q, voc), voc.V[np.arange(F)[None, :], P], w=weights)
    return np.argmin(D, axis=1)


# ----------------------------------------------------------------------------------------------------
# PSO calibration of (theta, alpha)
# ----------------------------------------------------------------------------------------------------
BOUND = (0.05, 0.95)


def valid_partition(labels: np.ndarray, K: int) -> bool:
    cnt = np.bincount(labels, minlength=K)
    return bool(np.all(cnt >= max(2, int(np.ceil(len(labels) / (10 * K))))))


class SilhouetteFitness:
    """Label-free fitness: Silhouette (precomputed distances on a fixed sub-sample) of the word partition,
    measured in the *continuous* semantic space; invalid partitions are penalised."""

    def __init__(self, S: np.ndarray, K: int, spec: int, per_feature: bool, n_sub: int = 700, seed: int = 0,
                 max_iter: int = 30):
        rng = np.random.default_rng(seed)
        self.idx = rng.choice(len(S), min(n_sub, len(S)), replace=False)
        self.S = S[self.idx]
        sq = (self.S ** 2).sum(1)
        self.Dp = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * self.S @ self.S.T, 0))
        np.fill_diagonal(self.Dp, 0)
        self.K, self.spec, self.pf, self.max_iter = K, spec, per_feature, max_iter
        self.F = S.shape[1]
        self.evals = 0

    def decode(self, x: np.ndarray):
        if self.pf:
            return x[:self.F], x[self.F:]
        return float(x[0]), float(x[1])

    def labels(self, x: np.ndarray):
        th, al = self.decode(x)
        voc = make_vocab(th, al, self.F, self.spec)
        q = semantize_words(self.S, voc)
        return fit_words(q, voc, self.K, max_iter=self.max_iter).labels

    def __call__(self, x: np.ndarray) -> float:
        self.evals += 1
        lab = self.labels(x)
        if not valid_partition(lab, self.K) or len(np.unique(lab)) < 2:
            return 1.0
        return -float(silhouette_score(self.Dp, lab, metric="precomputed"))


class AgreementFitness:
    """Label-free fitness: adjusted Rand index between the word partition and a reference (float) partition of the
    same training samples -- i.e. how faithfully the words reproduce the trained float model."""

    def __init__(self, S: np.ndarray, ref: np.ndarray, K: int, spec: int, per_feature: bool, n_sub: int = 1500,
                 seed: int = 0, max_iter: int = 30):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(S), min(n_sub, len(S)), replace=False)
        self.S, self.ref = S[idx], ref[idx]
        self.K, self.spec, self.pf, self.max_iter, self.F = K, spec, per_feature, max_iter, S.shape[1]
        self.evals = 0

    def decode(self, x):
        return (x[:self.F], x[self.F:]) if self.pf else (float(x[0]), float(x[1]))

    def __call__(self, x: np.ndarray) -> float:
        from sklearn.metrics import adjusted_rand_score
        self.evals += 1
        th, al = self.decode(x)
        voc = make_vocab(th, al, self.F, self.spec)
        lab = fit_words(semantize_words(self.S, voc), voc, self.K, max_iter=self.max_iter).labels
        return -float(adjusted_rand_score(self.ref, lab))


def calibrate(S_train: np.ndarray, K: int, spec: int, per_feature: bool = False, n_particles: int = 16,
              n_iter: int = 25, seed: int = 0, n_sub: int = 700, objective: str = "silhouette",
              ref: Optional[np.ndarray] = None):
    """PSO over (theta, alpha) (shared) or (theta_j, alpha_j)_j; returns (theta, alpha, PSOResult).

    ``objective``: ``"silhouette"`` (label-free quality) or ``"agreement"`` (ARI with the reference partition ``ref``)."""
    if objective == "silhouette":
        fit = SilhouetteFitness(S_train, K, spec, per_feature, n_sub=n_sub, seed=seed)
    else:
        fit = AgreementFitness(S_train, ref, K, spec, per_feature, n_sub=max(n_sub, 1500), seed=seed)
    F = S_train.shape[1]
    d = 2 * F if per_feature else 2
    x0 = np.full(d, 0.5)
    res = pso(fit, [BOUND[0]] * d, [BOUND[1]] * d, n_particles=n_particles, n_iter=n_iter, seed=seed, x0=x0)
    th, al = fit.decode(res.x)
    return th, al, res


# ----------------------------------------------------------------------------------------------------
# Interpretation helpers
# ----------------------------------------------------------------------------------------------------
def distinguishing_words(P: np.ndarray, q_train: np.ndarray) -> np.ndarray:
    """Boolean (K, F): prototype word differs from the feature's most frequent training word."""
    K, F = P.shape
    T = int(max(q_train.max(), P.max())) + 1
    mode = np.array([np.bincount(q_train[:, j], minlength=T).argmax() for j in range(F)])
    return P != mode[None, :]


def describe(P: np.ndarray, voc: Vocab, q_train: np.ndarray, feature_names: Sequence[str] | None = None):
    """Per-cluster list of (feature, word) for the distinguishing features, e.g. ('petal length', 'Very Large')."""
    from .ha_classic import readable
    F = P.shape[1]
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(F)]
    dm = distinguishing_words(P, q_train)
    out = []
    for k in range(P.shape[0]):
        out.append([(names[j], readable(voc.names[j][P[k, j]])) for j in range(F) if dm[k, j]])
    return out


# ----------------------------------------------------------------------------------------------------
# Categorical baseline: k-modes (Hamming distance on equal-width bins; ignores the order of the words)
# ----------------------------------------------------------------------------------------------------
def bins_equal_width(S: np.ndarray, T: int) -> np.ndarray:
    return np.minimum((np.clip(S, 0, 1) * T).astype(np.int32), T - 1)


def kmodes(B: np.ndarray, K: int, T: int, n_init: int = 5, seed: int = 0, max_iter: int = 50):
    """Huang's k-modes with random distinct-row initialisation; returns (modes (K, F), labels, cost)."""
    rng = np.random.default_rng(seed)
    N, F = B.shape
    best = None
    for _ in range(n_init):
        modes = B[rng.choice(N, K, replace=False)].copy()
        labels = None
        for _it in range(max_iter):
            D = np.stack([(B != modes[k]).sum(1) for k in range(K)], axis=1)
            new = np.argmin(D, axis=1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for k in range(K):
                m = labels == k
                if m.any():
                    for j in range(F):
                        modes[k, j] = np.bincount(B[m, j], minlength=T).argmax()
        D = np.stack([(B != modes[k]).sum(1) for k in range(K)], axis=1)
        labels = np.argmin(D, axis=1)
        cost = float(D[np.arange(N), labels].sum())
        if best is None or cost < best[2]:
            best = (modes.copy(), labels, cost)
    return best


def kmodes_predict(B: np.ndarray, modes: np.ndarray) -> np.ndarray:
    return np.argmin(np.stack([(B != m).sum(1) for m in modes], axis=1), axis=1)
