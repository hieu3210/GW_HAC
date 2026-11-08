"""Standard particle swarm optimisation (same update rule and defaults as the user's pyLTS / CO-LTS code).

    V <- w V + c1 r1 (P_i - X) + c2 r2 (P_g - X),   X <- clip(X + V)         (w = 0.4, c1 = c2 = 2.0)

Minimises ``f`` over a box; deterministic given ``seed``.  The objective of the word-domain clustering is
piecewise constant in (theta, alpha) (a word assignment only changes when a cell boundary crosses a data value),
which is why a gradient-free optimiser is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass
class PSOResult:
    x: np.ndarray
    fun: float
    history: list          # best value after each iteration
    n_eval: int


def pso(f: Callable[[np.ndarray], float], lo: Sequence[float], hi: Sequence[float], n_particles: int = 16,
        n_iter: int = 25, omega: float = 0.4, c1: float = 2.0, c2: float = 2.0, seed: int = 0,
        x0: np.ndarray | None = None) -> PSOResult:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, (n_particles, d))
    if x0 is not None:                                   # optional warm start for particle 0
        X[0] = np.clip(x0, lo, hi)
    V = rng.uniform(-(hi - lo), hi - lo, (n_particles, d)) * 0.1
    pbest = X.copy()
    pval = np.array([f(x) for x in X])
    n_eval = n_particles
    g = int(np.argmin(pval))
    gbest, gval = pbest[g].copy(), float(pval[g])
    hist = [gval]
    for _ in range(n_iter):
        r1, r2 = rng.random((n_particles, d)), rng.random((n_particles, d))
        V = omega * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lo, hi)
        for i in range(n_particles):
            v = f(X[i])
            n_eval += 1
            if v < pval[i]:
                pval[i], pbest[i] = v, X[i].copy()
                if v < gval:
                    gval, gbest = float(v), X[i].copy()
        hist.append(gval)
    return PSOResult(x=gbest, fun=gval, history=hist, n_eval=n_eval)
