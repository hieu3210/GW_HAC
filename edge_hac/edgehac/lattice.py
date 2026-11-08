"""Scalar quantization lattices on the semantic axis [0, 1].

A :class:`Lattice` is a sorted set of reconstruction points ``v`` together with the nearest-point
cell structure (thresholds = midpoints).  Three families are provided:

* :func:`ha_lattice`      -- the Hedge-Algebra SQM lattice (THEORY.md, Lemmas 1-2); the only family
                             whose points carry linguistic names and have a closed-form covering radius.
* :func:`uniform_lattice` -- equally spaced points (equals the HA lattice for theta = alpha = 1/2).
* :func:`lloyd_max_lattice` -- data-adaptive 1-D k-means (Max 1960 / Lloyd 1982); the optimal-MSE
                             benchmark and the scalar special case of product quantization.

The HA core (:class:`haclustering.hedge_algebra.HedgeAlgebra`) is the only dependency on the rest of the
repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from haclustering.hedge_algebra import HedgeAlgebra, Term


@dataclass(frozen=True)
class Lattice:
    """Reconstruction points ``v`` (strictly increasing, inside [0, 1]) with nearest-point cells."""

    v: np.ndarray                      # (T,)
    kind: str = "custom"
    theta: Optional[float] = None
    alpha: Optional[float] = None
    depth: Optional[int] = None
    completed: bool = False
    names: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        v = np.asarray(self.v, dtype=float)
        if v.ndim != 1 or len(v) < 2 or np.any(np.diff(v) <= 0):
            raise ValueError("lattice points must be strictly increasing")
        if v[0] < 0 or v[-1] > 1:
            raise ValueError("lattice points must lie in [0, 1]")
        object.__setattr__(self, "v", v)

    # -- structure ---------------------------------------------------------------------------
    @property
    def T(self) -> int:
        return len(self.v)

    @property
    def bits(self) -> int:
        """Bits needed to store a cell index."""
        return int(np.ceil(np.log2(self.T)))

    @property
    def thresholds(self) -> np.ndarray:
        """Cell boundaries ``m_t = (v_t + v_{t+1}) / 2`` (length T-1)."""
        return 0.5 * (self.v[1:] + self.v[:-1])

    @property
    def cell_radius(self) -> np.ndarray:
        """Largest distance from ``v_t`` to a point of its own cell, per cell (length T)."""
        m = np.concatenate([[0.0], self.thresholds, [1.0]])
        return np.maximum(self.v - m[:-1], m[1:] - self.v)

    @property
    def radius(self) -> float:
        """Covering radius ``max_s min_t |s - v_t|`` of the lattice on [0, 1]."""
        return float(self.cell_radius.max())

    def encode(self, s: np.ndarray) -> np.ndarray:
        """Nearest-point index of (real-valued) semantic coordinates; ties go to the upper cell."""
        return np.searchsorted(self.thresholds, np.asarray(s, dtype=float), side="right")

    def decode(self, q: np.ndarray) -> np.ndarray:
        return self.v[np.asarray(q)]

    def mse(self, s: np.ndarray) -> float:
        return float(np.mean((np.asarray(s) - self.v[self.encode(s)]) ** 2))

    def phrase(self, t: int) -> str:
        return self.names[t] if self.names else f"L{t}"


# ------------------------------------------------------------------------------------------------
# Families
# ------------------------------------------------------------------------------------------------
def ha_lattice(theta: float = 0.5, alpha: float = 0.5, depth: int = 2, completed: bool = False) -> Lattice:
    """SQM lattice of the canonical HA {Little, Very} up to hedge depth ``depth``.

    ``completed=True`` adds the constants ``0, W, 1`` so that the points are exactly the boundaries of
    the depth-(d+1) fuzziness partition (Lemma 1): ``2^(d+2) + 1`` points.  Otherwise only the
    ``2^(d+2) - 2`` generator/hedge terms are used (the term set of the EWA-HAC paper).
    """
    ha = HedgeAlgebra(theta=theta, alpha=alpha)
    terms = ha.generate_terms(depth)                       # sorted by nu
    pts = [(ha.nu(t), ha.term_phrase(t)) for t in terms]
    if completed:
        pts += [(0.0, "Zero"), (theta, "Medium"), (1.0, "One")]
        pts.sort(key=lambda p: p[0])
    v = np.array([p[0] for p in pts])
    return Lattice(v=v, kind="ha", theta=theta, alpha=alpha, depth=depth, completed=completed,
                   names=tuple(p[1] for p in pts))


def uniform_lattice(T: int) -> Lattice:
    """Mid-rise uniform lattice with ``T`` points: ``v_t = (t + 1/2) / T``."""
    return Lattice(v=(np.arange(T) + 0.5) / T, kind="uniform")


def lloyd_max_lattice(s: np.ndarray, T: int, iters: int = 50) -> Lattice:
    """1-D Lloyd-Max quantizer with ``T`` levels fitted to the samples ``s`` (deterministic).

    Initialised at evenly spaced sample quantiles; duplicate levels (discrete data) are merged and the
    remaining slots are re-seeded at the largest-error cell so that exactly ``T`` distinct points
    result whenever the sample has at least ``T`` distinct values.
    """
    s = np.sort(np.asarray(s, dtype=float))
    uniq = np.unique(s)
    if len(uniq) <= T:                                      # fewer distinct values than levels
        v = uniq.copy()
        extra = T - len(v)
        if extra > 0:                                       # pad with evenly spaced unused points
            cand = np.setdiff1d(np.linspace(0, 1, 4 * T + 1), v)
            v = np.sort(np.concatenate([v, cand[np.linspace(0, len(cand) - 1, extra).astype(int)]]))
        return Lattice(v=v, kind="lloyd_max")
    v = np.quantile(s, (np.arange(T) + 0.5) / T)
    v = np.unique(v)
    while len(v) < T:                                       # quantile ties: add midpoints of widest gap
        g = np.argmax(np.diff(np.concatenate([[0.0], v, [1.0]])))
        edges = np.concatenate([[0.0], v, [1.0]])
        v = np.sort(np.append(v, 0.5 * (edges[g] + edges[g + 1])))
    for _ in range(iters):
        m = 0.5 * (v[1:] + v[:-1])
        idx = np.searchsorted(m, s, side="right")
        cnt = np.bincount(idx, minlength=T)
        sm = np.bincount(idx, weights=s, minlength=T)
        newv = np.where(cnt > 0, sm / np.maximum(cnt, 1), v)
        newv = np.sort(newv)
        if np.max(np.abs(newv - v)) < 1e-10:
            v = newv
            break
        v = newv
    # enforce strict monotonicity (degenerate cells)
    for i in range(1, len(v)):
        if v[i] <= v[i - 1]:
            v[i] = np.nextafter(v[i - 1], 2.0)
    return Lattice(v=np.clip(v, 0.0, 1.0), kind="lloyd_max")


# ------------------------------------------------------------------------------------------------
# Closed-form quantities from THEORY.md
# ------------------------------------------------------------------------------------------------
def phi(theta: float, alpha: float, m: int) -> float:
    """``phi_m = max_{|y|=m} fm(y) = max(theta, 1-theta) * max(alpha, 1-alpha)^m``."""
    return max(theta, 1.0 - theta) * max(alpha, 1.0 - alpha) ** m


def ha_radius_bound(theta: float, alpha: float, depth: int, completed: bool) -> float:
    """Lemma 2: covering radius of the HA lattice (equality if completed, upper bound otherwise)."""
    f = phi(theta, alpha, depth + 1)
    return 0.5 * f if completed else f


def n_levels(depth: int, completed: bool) -> int:
    return 2 ** (depth + 2) + (1 if completed else -2)
