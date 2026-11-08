"""Lattice design: the per-feature lattices compared in the study (THEORY.md, Sec. 3).

Variants (all with the same number of levels ``T = 2^(d+2) - 2``, i.e. the HA term set of depth ``d``):

``uniform``   equally spaced levels (the HA lattice for theta = alpha = 1/2 up to the two removed constants)
``ha_train``  HA lattice with the (theta, alpha) selected when training EWA-HAC (no extra fitting)
``ha_opt``    one HA lattice minimising the prototype-aware distortion J_L over a (theta, alpha) grid
``ha_opt_pf`` per-feature HA lattices minimising J_L (separable)
``lloyd_max`` per-feature 1-D Lloyd-Max levels (MSE-optimal; scalar product quantization)
"""
from __future__ import annotations

from itertools import product
from typing import List

import numpy as np

from .fit import FittedHA
from .lattice import Lattice, ha_lattice, lloyd_max_lattice, n_levels, uniform_lattice

GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
VARIANTS = ("uniform", "ha_train", "ha_opt", "ha_opt_pf", "lloyd_max")


def distortion_weights(fit: FittedHA) -> np.ndarray:
    """``g_j = w_j^2 * sum_{a<c} (c_aj - c_cj)^2``  (prototype-aware weight of feature j)."""
    C, w = fit.centers, fit.weights
    d2 = (C[:, None, :] - C[None, :, :]) ** 2
    return w ** 2 * d2.sum(axis=(0, 1)) / 2.0


def _mse_per_feature(S: np.ndarray, lat: Lattice) -> np.ndarray:
    q = lat.encode(S)
    return np.mean((S - lat.v[q]) ** 2, axis=0)


def _subsample(S: np.ndarray, n: int = 4000) -> np.ndarray:
    if len(S) <= n:
        return S
    return S[np.random.default_rng(0).choice(len(S), n, replace=False)]


def build(variant: str, fit: FittedHA, S_train: np.ndarray, depth: int) -> List[Lattice]:
    """Per-feature lattice list for ``variant`` (training data only; no labels)."""
    F = fit.F
    T = n_levels(depth, completed=False)
    if variant == "uniform":
        return [uniform_lattice(T)] * F
    if variant == "ha_train":
        return [ha_lattice(fit.theta, fit.alpha, depth)] * F
    S = _subsample(S_train)
    if variant in ("ha_opt", "ha_opt_pf"):
        g = distortion_weights(fit)
        mse = {}
        lats = {}
        for th, al in product(GRID, GRID):
            lat = ha_lattice(th, al, depth)
            lats[(th, al)] = lat
            mse[(th, al)] = _mse_per_feature(S, lat)
        if variant == "ha_opt":
            best = min(mse, key=lambda k: float(np.dot(g, mse[k])))
            return [lats[best]] * F
        keys = list(mse)
        M = np.stack([mse[k] for k in keys])            # (49, F)
        pick = np.argmin(M, axis=0)                     # separable: per-feature minimiser (g_j > 0 cancels)
        return [lats[keys[i]] for i in pick]
    if variant == "lloyd_max":
        return [lloyd_max_lattice(S[:, j], T) for j in range(F)]
    raise ValueError(variant)


def distortion(variant_lattices: List[Lattice], S: np.ndarray, g: np.ndarray) -> float:
    """``J_L`` of a lattice list on data ``S``."""
    mse = np.array([np.mean((S[:, j] - variant_lattices[j].v[variant_lattices[j].encode(S[:, j])]) ** 2)
                    for j in range(S.shape[1])])
    return float(np.dot(g, mse))
