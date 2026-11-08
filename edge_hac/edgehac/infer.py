"""Reference (NumPy) integer-only inference for :class:`EdgeModel` with certificate evaluation.

This is the executable specification of the C kernels in ``c/edgehac.c``: both must agree bit-for-bit
on labels, integer margins and certificate flags (checked by ``tests/test_c_kernels.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .edge_model import ADAPT_SHIFT, EdgeModel


def to_codes(S: np.ndarray, B: int) -> np.ndarray:
    """Sensor front-end: B-bit code of semantic coordinates, ``round(s (2^B - 1))``."""
    top = (1 << B) - 1
    return np.rint(np.asarray(S, dtype=float) * top).astype(np.int32)


def encode_indices(model: EdgeModel, codes: np.ndarray) -> np.ndarray:
    """Lattice cell index per feature via the integer thresholds (``#{tau <= code}``)."""
    N, F = codes.shape
    q = np.empty((N, F), dtype=np.int32)
    if model.shared_lattice:
        q[:] = np.searchsorted(model.tau[0], codes, side="right")
    else:
        for j in range(F):
            q[:, j] = np.searchsorted(model.tau[j], codes[:, j], side="right")
    return q


def integer_scores(model: EdgeModel, q: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Integer scores ``S~_k`` (N, K); smaller is better."""
    N, F = q.shape
    out = np.empty((N, model.K), dtype=np.int64)
    fidx = np.arange(F)
    for i in range(0, N, chunk):
        qi = q[i:i + chunk]
        if model.realization == "lut":
            g = model.lut[:, fidx[None, :], qi]                  # (K, n, F)
            out[i:i + chunk] = g.sum(axis=2).T
        else:
            lv = model.lev[fidx[None, :], qi]                    # (n, F)
            out[i:i + chunk] = model.beta[None, :] - lv @ model.U.T
    return out


@dataclass
class EdgeResult:
    labels: np.ndarray            # integer arg-min (lowest index on ties)
    margin: np.ndarray            # S~_(2) - S~_(1)  (integer units)
    cert_global: np.ndarray       # bool
    cert_adaptive: np.ndarray     # bool
    R_adaptive: np.ndarray        # per-sample adaptive threshold
    q: np.ndarray
    cert_pairwise: np.ndarray = None   # bool; exact per-pair rule (analysis reference, ADDENDUM_1)
    S: np.ndarray = None               # integer scores (N, K)


def run(model: EdgeModel, codes: np.ndarray) -> EdgeResult:
    return run_indices(model, encode_indices(model, codes))


def run_indices(model: EdgeModel, q: np.ndarray) -> EdgeResult:
    """Inference from lattice/word indices (the word-domain models start here: there is no input encoder)."""
    S = integer_scores(model, q)
    labels = np.argmin(S, axis=1)
    srt = np.partition(S, 1, axis=1)[:, :2]
    srt.sort(axis=1)
    margin = srt[:, 1] - srt[:, 0]
    Rg = model.R_global[labels]
    # adaptive rule in integer arithmetic (round-up), identical to the C kernel
    fidx = np.arange(model.F)
    E = model.E_adapt[fidx[None, :], q]                          # (N, F)
    acc = np.einsum("nf,nf->n", model.A_adapt[labels].astype(np.int64), E.astype(np.int64))
    Ra = ((acc + (1 << ADAPT_SHIFT) - 1) >> ADAPT_SHIFT) + model.Dmax_adapt[labels]
    return EdgeResult(labels=labels, margin=margin, cert_global=margin > Rg, cert_adaptive=margin > Ra,
                      R_adaptive=Ra, q=q, cert_pairwise=pairwise_certificate(model, S, labels, q), S=S)


def pairwise_certificate(model: EdgeModel, S: np.ndarray, labels: np.ndarray, q: np.ndarray,
                         chunk: int = 1024) -> np.ndarray:
    """Exact per-pair rule: ``S~_c - S~_a > ceil(scale * 2 sum_j w_j eps_{j,q_j} |c_aj - c_cj| + delta_ac)`` for all c != a.

    This is the tightest certificate implied by Lemma 4 and the integer error budget (ADDENDUM_1); it costs
    K*F extra operations and is used as an analysis reference for the cheaper global/adaptive rules.
    """
    w, C = model.meta["w"], model.meta["C"]
    N, F = q.shape
    fidx = np.arange(F)
    ok = np.empty(N, dtype=bool)
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        a = labels[sl]
        eq = model.meta["eps_cell"][fidx[None, :], q[sl]]                        # (n, F)
        sep = np.abs(C[a][:, None, :] - C[None, :, :])                           # (n, K, F)
        rho = 2.0 * np.einsum("nkf,nf->nk", sep, w[None, :] * eq)               # (n, K)
        need = np.ceil(model.scale * rho + model.meta["delta"][a] + 1e-9)
        gap = S[sl] - S[sl][np.arange(len(a)), a][:, None]                       # S~_c - S~_a
        gap[np.arange(len(a)), a] = np.iinfo(np.int64).max
        ok[sl] = np.all(gap > need, axis=1)
    return ok
