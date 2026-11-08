"""Property tests of the v2 helpers (run: python -m pytest v2/tests -q, from edge_hac/)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v2lib as L                                           # noqa: E402
from edgehac.whac import best_of, semantize_words           # noqa: E402

RNG = np.random.default_rng(0)


def toy(N=400, F=4, K=3):
    C = RNG.uniform(.2, .8, (K, F))
    return np.clip(C[RNG.integers(K, size=N)] + RNG.normal(0, .08, (N, F)), 0, 1), K


def test_best_of_v2_equals_best_of():
    S, K = toy()
    voc = L.GWI.interval_vocab(np.full(4, .4), np.full(4, .6), 3)
    q = semantize_words(S, voc)
    w = L.uniform_w(4)
    a = best_of(q, voc, K, n_init=5, seed=3, weights=w)
    b, _ = L.best_of_v2(q, voc, K, seed=3, w=w)
    assert np.array_equal(a.P, b.P) and a.J == b.J


def test_decomposition_identity():
    S, K = toy()
    from sklearn.cluster import KMeans
    lab = KMeans(K, n_init=5, random_state=0).fit(S).labels_
    for th, al, lvl in ((.5, .5, 2), (.35, .62, 3), (.6, .4, 5)):
        d = L.decomposition(S, lab, L.GWI.interval_vocab(np.full(4, th), np.full(4, al), lvl), L.uniform_w(4))
        assert d["identity_err"] < 1e-12


def test_theta_hat_between_median_and_half():
    for a_, b_ in ((2, 5), (5, 2), (1, 3), (.5, .5)):
        S = RNG.beta(a_, b_, (5000, 1))
        th = L.theta_hat(S)[0]
        lo, hi = sorted([np.median(S), .5])
        assert lo - 2e-3 <= th <= hi + 2e-3


def test_parent_rank_is_shift():
    voc3 = L.GWI.interval_vocab(np.array([.3]), np.array([.65]), 3)
    voc1 = L.GWI.interval_vocab(np.array([.3]), np.array([.65]), 1)
    names3, names1 = voc3.names[0], voc1.names[0]
    for t, w in enumerate(names3):
        assert names1[t >> 2] == w[2:]


def test_with_rep_monotone_and_cells_unchanged():
    S, _ = toy()
    voc = L.family_vocab("Beta", 3, S, {})
    for rep in ("mid", "mean"):
        v = L.with_rep(voc, rep, S)
        assert np.all(np.diff(v.V, axis=1) >= 0) and np.array_equal(v.thr, voc.thr)


def test_nearest_assignment_uses_midpoints():
    voc = L.with_assign(L.GWI.interval_vocab(np.array([.5]), np.array([.5]), 2), "nearest")
    assert np.allclose(voc.thr[0], 0.5 * (voc.V[0, 1:] + voc.V[0, :-1]))


def test_prepare_split_modes():
    X = np.column_stack([RNG.exponential(1, 300), RNG.normal(0, 1, 300), np.ones(300)])
    X[5, 1] = np.nan
    tr, te = np.arange(150), np.arange(150, 300)
    for mode in ("minmax", "log1p", "rank"):
        S_tr, S_te, keep = L.prepare_split(X, tr, te, mode)
        assert keep.tolist() == [True, True, False]
        assert np.isfinite(S_tr).all() and S_tr.min() >= 0 and S_tr.max() <= 1 and S_te.min() >= 0 and S_te.max() <= 1


def test_certified_granularity_no_violations():
    S, K = toy(600)
    w = L.uniform_w(4)
    cache = {}
    voc = L.family_vocab("HA-grid", 4, S, cache)
    f, _, _, q, _ = L.word_model(S, S, voc, K, 0, w)
    mstar, per_m = L.certified_granularity(q, voc, f.P, w, 4, lambda l: L.family_vocab("HA-grid", l, S, cache))
    assert all(p["viol"] == 0 for p in per_m) and mstar.min() >= 0 and mstar.max() <= 3


def test_translation_certificate_no_violations():
    S, K = toy(600)
    from sklearn.cluster import KMeans
    Z = KMeans(K, n_init=5, random_state=0).fit(S).cluster_centers_
    for lvl in (2, 3, 4):
        c = L.translation_certificate(S, Z, L.uniform_w(4), L.GWI.interval_vocab(np.full(4, .45), np.full(4, .55), lvl))
        assert not np.any(c["cert"] & c["changed"]) and not np.any(c["cert_static"] & c["changed"])


def test_Q_sign_predicts_alpha_direction():
    S = RNG.beta(2, 5, (20000, 1))
    th = L.theta_hat(S)
    Q = L.Q_hat(S, th, 6)[0]
    _, al = L.calibrate_grid(S, 6, theta_fixed=th)
    assert np.sign(al[0] - .5) in (0.0, -np.sign(Q))
