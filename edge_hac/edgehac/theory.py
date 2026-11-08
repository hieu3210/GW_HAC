"""Closed-form quantities of THEORY.md used to compare theory with measurements."""
from __future__ import annotations

import numpy as np

from .edge_model import EdgeModel
from .lattice import ha_radius_bound, phi


def eta_bar(model: EdgeModel) -> float:
    """``max_a R_a / scale``: the margin scale below which the certificate may fail (Theorem 2)."""
    return float(model.R_global.max() / model.scale)


def margin_events(margin_exact: np.ndarray, labels_edge: np.ndarray, labels_ref: np.ndarray,
                  cert: np.ndarray, eta: float) -> dict:
    """Check the inclusions of Theorem 2 on data; returns counts of violations (must be 0)."""
    mis = labels_edge != labels_ref
    return {
        "n": int(len(labels_ref)),
        "mislabel": int(mis.sum()),
        "abstain": int((~cert).sum()),
        "viol_certified_wrong": int((cert & mis).sum()),                  # Theorem 1
        "viol_mislabel_incl": int((mis & (margin_exact > eta + 1e-12)).sum()),      # {err} subset {M <= eta}
        "viol_abstain_incl": int(((~cert) & (margin_exact > 2 * eta + 1e-12)).sum()),  # {abst} subset {M <= 2 eta}
    }


def fit_tail_exponent(margin_exact: np.ndarray, lo_q: float = 0.002, hi_q: float = 0.2):
    """Estimate (C, gamma) in ``P(M <= t) ~ C t^gamma`` from the empirical margin CDF (log-log LS)."""
    m = np.sort(margin_exact[margin_exact > 0])
    if len(m) < 200:
        return np.nan, np.nan
    qs = np.geomspace(max(lo_q, 20 / len(m)), hi_q, 20)
    t = np.quantile(m, qs)
    ok = t > 0
    x, y = np.log(t[ok]), np.log(qs[ok])
    gamma, logC = np.polyfit(x, y, 1)
    return float(np.exp(logC)), float(gamma)


def abstention_bound_closed_form(theta, alpha, depth, completed, B, F, sigma, Delta_bar, C, gamma) -> float:
    """Corollary 3 (LUT form): ``C (2 eta)^gamma`` with ``eta <= 2 (r + u_B) Delta_bar + (F + 1)/sigma``."""
    r = ha_radius_bound(theta, alpha, depth, completed)
    uB = 1.0 / (2 * ((1 << B) - 1))
    eta = 2 * (r + uB) * Delta_bar + (F + 1) / sigma
    return float(C * (2 * eta) ** gamma)
