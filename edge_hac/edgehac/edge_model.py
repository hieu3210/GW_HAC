"""Compilation of a trained EWA-HAC model into an integer-only Edge-HAC model (THEORY.md, Sec. 2).

:func:`compile_edge` produces an :class:`EdgeModel` holding only integer tables:

* ``tau``      integer cell thresholds in the B-bit input-code domain,
* LUT form:    ``lut[k, j, t]``  (b_L-bit)                          -> multiplier-free scoring,
* MAC form:    ``U[k, j]`` (b_W-bit), ``lev[j, t]``, ``beta[k]``    -> dot-product scoring,
* certificate: ``R_global[k]`` and the integer tables of the adaptive rule.

All rounding is conservative (upward) wherever it enters a certificate, so Theorem 1 holds for the
tables exactly as stored (and as shipped to the C kernels).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .fit import FittedHA
from .lattice import Lattice

NUM_SLACK = 1e-9          # numerical slack added to every radius (float64 evaluation of thresholds)
ADAPT_SHIFT = 12          # fixed-point shift of the per-cell radii in the adaptive certificate


@dataclass
class EdgeModel:
    realization: str                    # "lut" | "mac"
    K: int
    F: int
    T: int
    B: int                              # input code bits
    lattice_bits: int
    shared_lattice: bool
    tau: np.ndarray                     # (F, T-1) int32 thresholds on input codes
    # LUT realization
    lut: Optional[np.ndarray] = None    # (K, F, T) uint
    lut_bits: int = 0
    # MAC realization
    U: Optional[np.ndarray] = None      # (K, F) uint
    lev: Optional[np.ndarray] = None    # (F, T) int32
    beta: Optional[np.ndarray] = None   # (K,) int64
    weight_bits: int = 0
    p: int = 0
    # certificate
    scale: float = 1.0
    R_global: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    A_adapt: Optional[np.ndarray] = None   # (K, F) uint32
    E_adapt: Optional[np.ndarray] = None   # (F, T) uint16 radii << ADAPT_SHIFT
    Dmax_adapt: Optional[np.ndarray] = None  # (K,) int64
    acc_bits: int = 32
    meta: dict = field(default_factory=dict)

    # ---- memory accounting (bytes), see THEORY.md Sec. 4 --------------------------------------
    def memory_bytes(self, adaptive: bool = False) -> dict:
        in_bytes = int(np.ceil(self.B / 8))
        thr = (self.T - 1) * (1 if self.shared_lattice else self.F) * in_bytes
        out = {"thresholds": thr}
        if self.realization == "lut":
            out["lut"] = int(self.K * self.F * self.T * np.ceil(self.lut_bits / 8))
        else:
            out["weights"] = int(self.K * self.F * np.ceil(self.weight_bits / 8))
            out["levels"] = int(self.T * (1 if self.shared_lattice else self.F) * 2)
            out["bias"] = int(4 * self.K)
        out["cert_global"] = int(4 * self.K)
        if adaptive:
            out["cert_adaptive"] = int(self.K * self.F * 2 + self.T * (1 if self.shared_lattice else self.F) * 2
                                       + 4 * self.K)
        out["total"] = int(sum(out.values()))
        return out


def _nonneg_ceil(x: np.ndarray) -> np.ndarray:
    return np.ceil(np.maximum(x, 0.0) - 1e-12).astype(np.int64)


def compile_edge(fit: FittedHA, lattices: Sequence[Lattice], B: int = 12, realization: str = "lut",
                 lut_bits: int = 8, weight_bits: int = 8, p_max: int = 8, exact_inputs: bool = False) -> EdgeModel:
    """Compile ``fit`` + per-feature ``lattices`` into integer tables."""
    K, F = fit.centers.shape
    if len(lattices) != F:
        raise ValueError("one lattice per feature required")
    T = lattices[0].T
    if any(l.T != T for l in lattices):
        raise ValueError("all lattices must have the same number of levels")
    shared = all(np.array_equal(l.v, lattices[0].v) for l in lattices)
    V = np.stack([l.v for l in lattices])                       # (F, T)
    w, C = fit.weights, fit.centers
    top = (1 << B) - 1
    uB = 1.0 / (2.0 * top)

    # --- input stage: integer thresholds on the code domain ---------------------------------
    tau = np.stack([np.ceil(l.thresholds * top - 1e-9) for l in lattices]).astype(np.int32)
    tau = np.maximum.accumulate(tau, axis=1)

    # --- per-cell radii (distance units), global radii ---------------------------------------
    if exact_inputs:                      # the model itself lives on the words: s = nu(word), no input quantization error
        eps_cell = np.full((F, T), NUM_SLACK)
    else:
        eps_cell = np.stack([l.cell_radius for l in lattices]) + uB + NUM_SLACK    # (F, T)
    eps_j = eps_cell.max(axis=1)                                              # (F,)
    vbar = V.max(axis=1)

    model = EdgeModel(realization=realization, K=K, F=F, T=T, B=B, lattice_bits=int(np.ceil(np.log2(T))),
                      shared_lattice=shared, tau=tau)

    # separations needed by the certificate
    Dsep = np.abs(C[:, None, :] - C[None, :, :])                # (K, K, F)
    Delta = np.where(np.eye(K, dtype=bool)[:, :, None], 0.0, Dsep).max(axis=1)   # (K, F) max_c |c_aj - c_cj|

    if realization == "lut":
        ent = w[None, :, None] * (V[None, :, :] - C[:, :, None]) ** 2            # (K, F, T)
        top_l = (1 << lut_bits) - 1
        sigma = top_l / max(ent.max(), 1e-300)
        model.lut = np.rint(sigma * ent).astype(np.int64)
        model.lut_bits = lut_bits
        model.scale = float(sigma)
        delta = np.full((K, K), float(F))
        model.acc_bits = int(lut_bits + np.ceil(np.log2(F + 1)))
    elif realization == "mac":
        A = 2.0 * w[None, :] * C                                                  # (K, F)
        Bk = np.sum(w[None, :] * C ** 2, axis=1)                                  # (K,)
        top_w = (1 << weight_bits) - 1
        Gamma = top_w / max(A.max(), 1e-300)
        U = np.rint(Gamma * A).astype(np.int64)
        # largest p <= p_max for which the int32 accumulator cannot overflow
        p = p_max
        while p > 2:
            worst = U.sum(axis=1).max() * (1 << p) + np.rint(Gamma * (1 << p) * Bk.max())
            if worst < 2 ** 31 - 1:
                break
            p -= 1
        lev = np.rint((1 << p) * V).astype(np.int64)
        beta = np.rint(Gamma * (1 << p) * Bk).astype(np.int64)
        model.U, model.lev, model.beta = U, lev, beta
        model.weight_bits, model.p = weight_bits, p
        model.scale = float(Gamma * (1 << p))
        dU = np.abs(U[:, None, :] - U[None, :, :]).sum(axis=2)                   # (K, K)
        delta = 1.0 + 0.5 * dU + (1 << p) * vbar.sum()
        model.acc_bits = 32
    else:
        raise ValueError(realization)

    # --- certificates -------------------------------------------------------------------------
    rho_glob = 2.0 * np.einsum("f,acf,f->ac", w, Dsep, eps_j)                    # (K, K)
    tot = model.scale * rho_glob + delta
    np.fill_diagonal(tot, -np.inf)
    model.R_global = np.ceil(tot.max(axis=1) + 1e-9).astype(np.int64)            # certified iff margin > R
    dmax = np.where(np.eye(K, dtype=bool), 0.0, delta).max(axis=1)
    model.Dmax_adapt = np.ceil(dmax + 1e-9).astype(np.int64)
    model.A_adapt = np.ceil(model.scale * 2.0 * w[None, :] * Delta).astype(np.int64)
    model.E_adapt = np.ceil(eps_cell * (1 << ADAPT_SHIFT)).astype(np.int64)
    model.meta = dict(eps_j=eps_j, eps_cell=eps_cell, Delta=Delta, w=w, C=C, delta=delta)
    return model
