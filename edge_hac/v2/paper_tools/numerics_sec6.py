"""Numerical verification of the high-resolution theory (Section 6 of the paper: Table 1, Fig. 1 and the in-text
numbers).  Pure NumPy/SciPy; needs no dataset.  Re-implemented from the specification in Section 6, because the
script that produced the submitted figure was not archived (see README, "Section 6").

    python paper_tools/numerics_sec6.py [--out ../../out] [--quick]          (run from edge_hac/v2; ~1-3 min)

Quadrature.  Midpoint rule with n nodes (4e5; 8e5 for the hedge-asymmetry panel).  f is taken constant on each node (its
midpoint value) and the moments of s^k f are integrated exactly, so D is continuous in (theta, alpha).
Optimal M-point quantizers: Lloyd's algorithm from the companding quantizer with point density ~ f^(1/3), stopped
at relative change 1e-9 of the distortion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import beta as Beta, norm

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------------------------
# Densities on [0, 1]
# ---------------------------------------------------------------------------------------------------------------
def _mix_pdf(s):
    return 0.7 * norm.pdf(s, 0.2, 0.07) + 0.3 * norm.pdf(s, 0.8, 0.1)


def _mix_cdf(s):
    return 0.7 * norm.cdf(s, 0.2, 0.07) + 0.3 * norm.cdf(s, 0.8, 0.1)


_ZMIX = _mix_cdf(1.0) - _mix_cdf(0.0)
_ZEXP = 1 - np.exp(-6.0)
DENSITIES = {
    "Uniform": (lambda s: np.ones_like(s), lambda s: np.clip(s, 0, 1)),
    "Beta(2,5)": (lambda s: Beta.pdf(s, 2, 5), lambda s: Beta.cdf(s, 2, 5)),
    "Beta(1,3)": (lambda s: Beta.pdf(s, 1, 3), lambda s: Beta.cdf(s, 1, 3)),
    "Trunc. exp.": (lambda s: 6 * np.exp(-6 * s) / _ZEXP, lambda s: (1 - np.exp(-6 * s)) / _ZEXP),
    "Mixture": (lambda s: _mix_pdf(s) / _ZMIX, lambda s: (_mix_cdf(s) - _mix_cdf(0.0)) / _ZMIX),
}


class Quad:
    """f is replaced by its value at the node midpoint on each of n equal nodes (midpoint rule); the moments
    of s^k f, k = 0, 1, 2, are then integrated exactly, also over partial nodes, so cell integrals are exact for
    piecewise-constant f and D is continuous in (theta, alpha)."""

    def __init__(self, f, n):
        self.n = n
        self.edges = np.arange(n + 1) / n
        s = (np.arange(n) + 0.5) / n
        self.fv = f(s)
        self.w = self.fv / n                          # node masses
        e = self.edges
        self.C = [np.concatenate([[0.0], np.cumsum(self.fv * (e[1:] ** (k + 1) - e[:-1] ** (k + 1)) / (k + 1))])
                  for k in range(3)]

    def cum(self, b, k):
        b = np.asarray(b, float)
        i = np.clip(np.floor(b * self.n).astype(int), 0, self.n - 1)
        e = self.edges[i]
        return self.C[k][i] + self.fv[i] * (b ** (k + 1) - e ** (k + 1)) / (k + 1)

    def moments(self, bnd):
        return [np.diff(self.cum(bnd, k)) for k in range(3)]   # mass, first and second moment per cell

    def distortion(self, bnd, reps):
        m0, m1, m2 = self.moments(bnd)
        return float(np.sum(m2 - 2 * reps * m1 + reps ** 2 * m0))


# ---------------------------------------------------------------------------------------------------------------
# HA cells (Proposition 1(b)): nu(x) splits I(x); the Little child is adjacent to the parent value.
# ---------------------------------------------------------------------------------------------------------------
def ha_cells(theta, alpha, level):
    """Boundaries (2^l + 1), SQM representatives (2^l) and hedge imbalance H = n_L - n_V (2^l) of level l."""
    beta = 1.0 - alpha
    a = np.array([0.0, theta])
    L = np.array([theta, 1.0 - theta])
    sg = np.array([-1, 1])                           # side of the cell relative to its parent value
    H = np.zeros(2, int)
    for _ in range(level - 1):
        low = np.where(sg > 0, alpha * L, beta * L)  # lower piece: L child if the cell lies above its parent
        hl = np.where(sg > 0, 1, -1)                 # hedge of the lower child: +1 for L, -1 for V
        a = np.stack([a, a + low], 1).ravel()
        L = np.stack([low, L - low], 1).ravel()
        H = np.stack([H + hl, H - hl], 1).ravel()
        sg = np.tile([-1, 1], len(sg))
    reps = a + np.where(sg > 0, alpha * L, beta * L)
    return np.append(a, 1.0), reps, H


def D_ha(q, theta, alpha, level):
    b, r, _ = ha_cells(theta, alpha, level)
    return q.distortion(b, r)


def Q_ell(q, theta, level):
    """Mass-weighted hedge imbalance (eq. Q) at alpha = 1/2."""
    b, _, H = ha_cells(theta, 0.5, level)
    m0 = q.moments(b)[0]
    side = np.where(b[:-1] < theta - 1e-15, theta ** 2, (1 - theta) ** 2)
    return float(np.sum(side * H * m0))


def lloyd(q, Finv_comp, M, tol=1e-9, itmax=200000):
    """Optimal M-point quantizer by Lloyd's algorithm from the companding start."""
    r = Finv_comp((np.arange(M) + 0.5) / M)
    Dold = np.inf
    for _ in range(itmax):
        b = np.concatenate([[0.0], 0.5 * (r[1:] + r[:-1]), [1.0]])
        m0, m1, _ = q.moments(b)
        r = np.where(m0 > 0, m1 / np.maximum(m0, 1e-300), 0.5 * (b[1:] + b[:-1]))
        D = q.distortion(b, r)
        if abs(Dold - D) <= tol * D:
            break
        Dold = D
    return D


def companding(q, f, level):
    """Nested companding quantizer: G^{-1} of the dyadic cells, midpoint representatives (Theorem 3(c))."""
    g = np.cbrt(q.w * q.n)
    G = np.concatenate([[0.0], np.cumsum(g)]) / g.sum()
    Ginv = lambda u: np.interp(u, G, q.edges)
    b = Ginv(np.arange(2 ** level + 1) / 2 ** level)
    b[0], b[-1] = 0.0, 1.0
    return q.distortion(b, 0.5 * (b[1:] + b[:-1])), Ginv


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[2] / "out"))
    ap.add_argument("--quick", action="store_true", help="1e5 nodes, fewer levels (smoke test)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    n1, n2 = (100_000, 200_000) if a.quick else (400_000, 800_000)
    res = {"n_nodes": n1, "n_nodes_alpha": n2, "densities": {}}

    # ---- Table 1: optimal Medium and vocabulary price ---------------------------------------------------------
    rows = []
    for name, (f, F) in DENSITIES.items():
        q = Quad(f, n1)
        med = brentq(lambda t: F(t) - 0.5, 1e-9, 1 - 1e-9)
        Phi = lambda t: t ** 2 * F(t) + (1 - t) ** 2 * (1 - F(t))
        bal = lambda t: F(t) - (1 - t) + f(np.array([t]))[0] * (t - 0.5)   # Phi'/2 = 0, eq. (balance)
        lo, hi = min(med, 0.5), max(med, 0.5)
        th = 0.5 if hi - lo < 1e-12 else brentq(bal, lo - 1e-9, hi + 1e-9)
        grid = np.round(np.arange(0.005, 1.0, 0.005), 3)
        Dg = [D_ha(q, t, 0.5, 12) for t in grid]
        tg = float(grid[int(np.argmin(Dg))])
        I3 = float(np.sum(np.cbrt(q.w * q.n)) / q.n) ** 3
        P = Phi(th)
        lim_err = abs(4 ** 12 * D_ha(q, th, 0.5, 12) / (P / 3) - 1)
        red = 1 - 4 * P
        rec = red / (1 - I3) if 1 - I3 > 1e-9 else float("nan")
        r = dict(density=name, median=med, theta_star=th, grid_l12=tg, reduction=red, price_ha=4 * P / I3,
                 price_dyadic=1 / I3, recovered=rec, I3=I3, rel_err_limit_l12=lim_err,
                 between=bool(lo - 1e-9 <= th <= hi + 1e-9), grid_within_step=bool(abs(tg - th) <= 0.005))
        rows.append(r)
        res["densities"][name] = r
        print(f"{name:12s} med {med:.3f} theta* {th:.3f} grid {tg:.3f} 1-4Phi {100*red:5.1f}% "
              f"4Phi/I3 {4*P/I3:.3f} 1/I3 {1/I3:.3f} recovered {100*rec if rec==rec else 0:4.0f}%  "
              f"|4^12 D/(Phi/3)-1| {lim_err:.1e}")
    tex = [r"\begin{tabular}{lccccccc}", r"\toprule",
           r"Density & median & $\theta^\ast$ & grid ($\ell{=}12$) & $1-4\Phi^\ast$ & $4\Phi^\ast/I_3$ & $1/I_3$ & recovered\\",
           r"\midrule"]
    for r in rows:
        u = r["density"] == "Uniform"
        dname = r["density"].replace("Trunc. exp.", "Trunc.\\ exp.")
        red = "0" if u else "%.1f\\%%" % (100 * r["reduction"])
        rec = "--" if u else "%.0f\\%%" % (100 * r["recovered"])
        tex.append("%s & %.3f & %.3f & %.3f & %s & %.3f & %.3f & %s\\\\" % (
            dname, r["median"], r["theta_star"], r["grid_l12"], red, r["price_ha"], r["price_dyadic"], rec))
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "tab1_medium.tex").write_text("\n".join(tex) + "\n")

    # ---- Hedge asymmetry: gradient check (l = 8, 10, 12) and scaled optimum (Fig. 1b) -------------------------
    levels_b = list(range(3, 13)) if not a.quick else [3, 4, 6, 8]
    figb = {}
    grad = {}
    gains = {}
    for name in ["Beta(2,5)", "Beta(1,3)", "Trunc. exp.", "Mixture"]:
        f, F = DENSITIES[name]
        q = Quad(f, n2)
        th = res["densities"][name]["theta_star"]
        Phi = th ** 2 * F(th) + (1 - th) ** 2 * (1 - F(th))
        h = 1e-4
        grad[name] = {}
        for l in ([8, 10, 12] if not a.quick else [8]):
            num = (D_ha(q, th, 0.5 + h, l) - D_ha(q, th, 0.5 - h, l)) / (2 * h)
            grad[name][l] = num / (4 ** (1 - l) * Q_ell(q, th, l) / 3)
        sc, pr = [], []
        gains[name] = {}
        for l in levels_b:
            m = minimize_scalar(lambda al: D_ha(q, th, al, l), bounds=(0.3, 0.7), method="bounded",
                                options={"xatol": 1e-9})
            sc.append(l * (m.x - 0.5))
            pr.append(-Q_ell(q, th, l) / (6 * Phi))
            d0 = D_ha(q, th, 0.5, l)
            gains[name][l] = (d0 - m.fun) / d0
        figb[name] = dict(levels=levels_b, scaled=sc, predicted=pr)
        print(f"{name:12s} grad ratio {', '.join(f'l={k}: {v:.3f}' for k, v in grad[name].items())}; "
              f"l*(alpha*-1/2) at l={levels_b[-1]}: {sc[-1]:+.3f} (pred {pr[-1]:+.3f}); "
              f"gain l=3 {100*gains[name].get(3, np.nan):.2f}%, l={levels_b[-1]} {100*gains[name][levels_b[-1]]:.2f}%")
    res["gradient_ratio"] = grad
    res["fig1b"] = figb
    res["alpha_gain"] = gains

    # ---- Price for the truncated exponential (Fig. 1a) --------------------------------------------------------
    f, F = DENSITIES["Trunc. exp."]
    q = Quad(f, n1)
    I3 = res["densities"]["Trunc. exp."]["I3"]
    th = res["densities"]["Trunc. exp."]["theta_star"]
    levels_a = list(range(2, 11)) if not a.quick else [2, 3, 4, 5, 6]
    fa = dict(levels=levels_a, ha=[], dyadic=[], companding=[], theta_opt=[], alpha_opt=[], ha_continuous=[])
    # "HA with optimal parameters": best (theta, alpha) on the grid of step 0.005 on [0.005, 0.995]^2 per level
    # (this reproduces theta = 0.33 at l = 2, 0.315 for l >= 7 and the price 1.269 at l = 10 quoted in the paper);
    # the continuous optimum (Nelder-Mead) is stored as ha_continuous.
    G = np.round(np.arange(0.005, 1.0, 0.005 if not a.quick else 0.02), 3)
    for l in levels_a:
        Dc, Ginv = companding(q, f, l)
        Ds = lloyd(q, Ginv, 2 ** l)
        M = np.array([[D_ha(q, t, al, l) for al in G] for t in G])
        i, j = np.unravel_index(int(M.argmin()), M.shape)
        m = minimize(lambda p: D_ha(q, p[0], p[1], l), [G[i], G[j]], method="Nelder-Mead",
                     options=dict(xatol=1e-7, fatol=1e-16, maxiter=4000))
        fa["ha"].append(M[i, j] / Ds)
        fa["ha_continuous"].append(m.fun / Ds)
        fa["dyadic"].append(D_ha(q, 0.5, 0.5, l) / Ds)
        fa["companding"].append(Dc / Ds)
        fa["theta_opt"].append(float(G[i]))
        fa["alpha_opt"].append(float(G[j]))
        print(f"price l={l:2d}: HA(opt, grid) {fa['ha'][-1]:.3f} (theta {G[i]:.3f}, alpha {G[j]:.3f}; continuous "
              f"{fa['ha_continuous'][-1]:.3f}) dyadic {fa['dyadic'][-1]:.3f} companding {fa['companding'][-1]:.3f}")
    fa["limits"] = dict(ha=4 * (th ** 2 * F(th) + (1 - th) ** 2 * (1 - F(th))) / I3, dyadic=1 / I3, companding=1.0)
    res["fig1a"] = fa

    # ---- Objective decomposition on synthetic Gaussian clusters (Proposition 6) -------------------------------
    res["decomposition"] = decomposition_example(levels=range(2, 7))

    # ---- Figure 1 ---------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.4))
    cols = {"ha": "#2a78d6", "dyadic": "#8a8a84", "companding": "#d6732a"}
    labs = {"ha": r"HA, optimal $(\theta,\alpha)$", "dyadic": r"dyadic HA, $\theta=\alpha=1/2$", "companding": r"$f^{1/3}$ compander (nested)"}
    for k in ("dyadic", "ha", "companding"):
        ax[0].plot(fa["levels"], fa[k], "o-", ms=2.5, lw=0.8, color=cols[k], label=labs[k])
        ax[0].axhline(fa["limits"][k], ls="--", lw=0.6, color=cols[k])
    ax[0].set_xlabel(r"level $\ell$"); ax[0].set_ylabel(r"price $\mathcal{P}_\ell$"); ax[0].set_title("(a) Price of the linguistic vocabulary", fontsize=7.5)
    ax[0].legend(frameon=False, fontsize=6)
    cb = {"Beta(2,5)": "#2a78d6", "Trunc. exp.": "#d6732a", "Mixture": "#3a9a5b", "Beta(1,3)": "#9a3a8a"}
    for name in FIG_B:
        d = figb[name]
        ax[1].plot(d["levels"], d["scaled"], "-", lw=0.9, color=cb[name], label=name)
        ax[1].plot(d["levels"], d["predicted"], "--", lw=0.7, color=cb[name])
    ax[1].axhline(0, lw=0.4, color="#bbbbbb")
    ax[1].set_xlabel(r"level $\ell$"); ax[1].set_ylabel(r"$\ell(\alpha^\ast_\ell-\frac{1}{2})$"); ax[1].set_title(r"(b) Optimal hedge asymmetry, scaled by $\ell$", fontsize=7.5)
    ax[1].legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.3))
    for x in ax:
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3, h_pad=1.0)
    fig.savefig(out / "figures" / "theory_verification.pdf")
    (out / "tables" / "numerics_sec6.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out/'tables'/'tab1_medium.tex'}, {out/'figures'/'theory_verification.pdf'}, "
          f"{out/'tables'/'numerics_sec6.json'}")


FIG_B = ["Beta(2,5)", "Trunc. exp.", "Mixture"]   # the three non-uniform densities plotted in Fig. 1(b)


def decomposition_example(levels, seed=0, N=3000, d=4, K=4):
    """Four Gaussian clusters in [0,1]^4: check of the exact decomposition J_l - J_N = T1 + T2 + T3.

    The paper does not fix the cluster means and spread; this uses means drawn uniformly in [0.2, 0.8]^4,
    SD 0.06, clipped to [0, 1], HA-theta parameters per feature and the k-means partition (n_init = 10).
    The identity holds for any configuration; the ratios |T2|/(T1+T3) depend on it."""
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.2, 0.8, (K, d))
    y = rng.integers(0, K, N)
    S = np.clip(mu[y] + rng.normal(0, 0.06, (N, d)), 0, 1)
    lab = KMeans(K, n_init=10, random_state=seed).fit_predict(S)
    w = np.full(d, 1 / d)
    th = []
    for j in range(d):                                # empirical Phi minimizer (Remark 1)
        x = np.sort(S[:, j]); Fe = np.arange(1, N + 1) / N
        cand = np.unique(np.clip(x, 1e-6, 1 - 1e-6))
        Fc = np.searchsorted(x, cand, side="left") / N
        th.append(float(cand[np.argmin(cand ** 2 * Fc + (1 - cand) ** 2 * (1 - Fc))]))
    rows = []
    for l in levels:
        V = np.empty_like(S)
        grids = []
        for j in range(d):
            b, r, _ = ha_cells(th[j], 0.5, l)
            idx = np.clip(np.searchsorted(b, S[:, j], side="right") - 1, 0, len(r) - 1)
            V[:, j] = r[idx]
            grids.append(r)
        e = V - S
        JN = T1 = T2 = T3 = Jl = 0.0
        for k in range(K):
            m = lab == k
            z = S[m].mean(0); eb = e[m].mean(0); vb = V[m].mean(0)
            P = np.array([g[np.argmin(np.abs(g - vb[j]))] for j, g in enumerate(grids)])   # Lemma 4 (word mean)
            JN += np.sum(w * (S[m] - z) ** 2)
            T1 += np.sum(w * (e[m] - eb) ** 2)
            T2 += 2 * np.sum(w * (S[m] - z) * e[m])
            T3 += m.sum() * np.sum(w * (vb - P) ** 2)
            Jl += np.sum(w * (V[m] - P) ** 2)
        Dh = np.sum(w * e ** 2) / N
        rows.append(dict(level=l, identity_error=abs(Jl - JN - (T1 + T2 + T3)) / JN, T2_ratio=abs(T2) / (T1 + T3),
                         gap_over_ND=abs(Jl - JN) / JN / (N * Dh / JN)))
    for r in rows:
        print(f"decomposition l={r['level']}: |identity err|/J_N {r['identity_error']:.1e}, |T2|/(T1+T3) "
              f"{r['T2_ratio']:.2f}, relative gap / (N D/J_N) {r['gap_over_ND']:.2f}")
    return rows


if __name__ == "__main__":
    main()
