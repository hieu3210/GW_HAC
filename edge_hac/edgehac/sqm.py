"""Quantifying semantics of the two-generator / two-hedge algebra H = {N, V} (THEORY_UNIFIED.md, Section T0).

Words use the pyLTS encoding, outermost hedge first: generators ``-`` (Small) and ``+`` (Large), the negative hedge
``L`` (read *Little* or *Rather* depending on the algebra), the positive hedge ``V`` (Very), neutral ``W``.
``theta = fm(c-)``, ``alpha = mu(N)``, ``beta = 1 - alpha``.

The value of a word is fixed by the *relative sign table* Sgn(h, k) (h strengthens / weakens the modification of k),
not by the hedge's name.  Interval structure (the children of every word lie on both sides of it) forces
Sgn(N, k) = -Sgn(V, k); with Sgn(V, V) = +1 exactly two consistent algebras remain:

* ``"little"`` -- Sgn(V, N) = +1, Sgn(N, N) = -1, Sgn(N, V) = -1: the table of Nguyen & Wechler (1990), Example 1
  (V positive and L negative w.r.t. V and L).  Ho--Long omega = beta for both hedges, so kappa(V) = kappa(N) = alpha.
  Equals ``haclustering.hedge_algebra`` (published EWA-HAC) and HA_PLMR_VIO's "Rather".
* ``"rather"`` -- Sgn(V, N) = -1, Sgn(N, N) = +1, Sgn(N, V) = -1: Very *weakens* Rather.  omega(Vx) = beta,
  omega(Nx) = alpha, so kappa(V) = alpha, kappa(N) = beta.  One-hedge values = LTS closed forms (4.1)--(4.7).

Common recursion: nu(W) = theta, nu(c-) = theta - alpha theta, nu(c+) = theta + alpha (1 - theta),
nu(hx) = nu(x) + Sgn(hx) fm(hx) kappa(h).

``"lts"`` (legacy, ``pyLTS`` / ``ha_classic``) uses the kappa of ``rather`` with the sign recursion of ``little``; it is
an SQM only for words with at most one hedge (kept to reproduce study M2b).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

VARIANTS = ("little", "rather", "lts")
_NAMES = {"-": "Small", "+": "Large", "W": "Middle", "V": "Very"}
_NEG_NAME = {"little": "Little", "rather": "Rather", "lts": "Rather"}
# relative sign Sgn(h, k) of the outer hedge h w.r.t. the inner hedge k
_TABLE = {"little": {("V", "V"): 1, ("V", "L"): 1, ("L", "V"): -1, ("L", "L"): -1},
          "rather": {("V", "V"): 1, ("V", "L"): -1, ("L", "V"): -1, ("L", "L"): 1}}


def _check(theta: float, alpha: float, variant: str) -> None:
    if not (0.0 < theta < 1.0 and 0.0 < alpha < 1.0):
        raise ValueError("theta and alpha must lie in (0, 1)")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")


def sign(x: str, variant: str = "little") -> int:
    """Sgn(x) in {-1, 0, +1}: Sgn(c-+) = -+1; on a generator V keeps and N reverses the sign; deeper,
    Sgn(h'hx) = Sgn(h', h) Sgn(hx) with the variant's table (``lts`` uses the ``little`` table)."""
    if x == "W":
        return 0
    if len(x) == 1:
        return -1 if x == "-" else 1
    if len(x) == 2:
        return sign(x[1:]) if x[0] == "V" else -sign(x[1:])
    table = _TABLE["rather" if variant == "rather" else "little"]
    return table[(x[0], x[1])] * sign(x[1:], variant)


def fm(x: str, theta: float, alpha: float) -> float:
    """Fuzziness measure fm(x) = mu(h_k)...mu(h_1) fm(c); fm(W) = 0."""
    if x == "W":
        return 0.0
    f = theta if x[-1] == "-" else 1.0 - theta
    for h in x[:-1]:
        f *= alpha if h == "L" else 1.0 - alpha
    return f


def nu(x: str, theta: float, alpha: float, variant: str = "little") -> float:
    """Quantifying semantics nu(x) of word x."""
    _check(theta, alpha, variant)
    return _nu(x, float(theta), float(alpha), variant)


@lru_cache(maxsize=200_000)
def _nu(x: str, theta: float, alpha: float, variant: str) -> float:
    if x == "W":
        return theta
    if x == "-":
        return theta - alpha * theta
    if x == "+":
        return theta + alpha * (1.0 - theta)
    kappa = alpha if (x[0] == "V" or variant == "little") else 1.0 - alpha
    return _nu(x[1:], theta, alpha, variant) + sign(x, variant) * fm(x, theta, alpha) * kappa


def lts_offset(x: str, theta: float, alpha: float) -> float:
    """Closed form of nu_lts(x) - nu_little(x) = (1 - 2 alpha) * sum over the L-prefixes y of x of Sgn(y) fm(y)."""
    d = 0.0
    for k in range(len(x) - 1):
        y = x[k:]
        if y[0] == "L":
            d += sign(y) * fm(y, theta, alpha)
    return (1.0 - 2.0 * alpha) * d


def interval(x: str, theta: float, alpha: float, variant: str = "little") -> Tuple[float, float]:
    """Fuzziness interval I(x) = [a, b], b - a = fm(x).  I(Np), I(Vp) partition I(p); the child h p lies below p iff
    Sgn(hp) = -1 (``lts`` has no interval semantics of its own; its intervals are those of ``little``)."""
    if variant == "lts":
        variant = "little"
    if x == "W":
        return theta, theta
    a, b = (0.0, theta) if x[-1] == "-" else (theta, 1.0)
    for k in range(len(x) - 2, -1, -1):          # apply hedges innermost first
        p = x[k + 1:]
        f = b - a
        width = (alpha if x[k] == "L" else 1.0 - alpha) * f
        if sign(x[k] + p, variant) < 0:          # this child is the lower part of I(p)
            b = a + width
        else:
            a = b - width
    return a, b


def parent(x: str) -> str:
    """Hedge projection pi(hx) = x (drop the outermost hedge); generators and W are fixed points."""
    return x[1:] if len(x) > 1 else x


@lru_cache(maxsize=None)
def word_list(spec: int) -> Tuple[str, ...]:
    """Words with at most ``spec`` hedges plus W: T = 2^(spec+2) - 1 words (spec = 1: the 7 LTS words)."""
    words = ["-", "W", "+"]
    frontier = ["-", "+"]
    for _ in range(spec):
        frontier = [h + w for w in frontier for h in ("L", "V")]
        words += frontier
    return tuple(words)


def level_words(k: int) -> Tuple[str, ...]:
    """Words of length exactly k (k - 1 hedges); their intervals partition [0, 1] (W excluded)."""
    frontier = ["-", "+"]
    for _ in range(k - 1):
        frontier = [h + w for w in frontier for h in ("L", "V")]
    return tuple(frontier)


def linguistic_order(spec: int, theta: float, alpha: float, variant: str = "little") -> List[str]:
    """Words sorted by the HA order of the variant's algebra (= order of their intervals; ``lts`` -> ``little``)."""
    v = "little" if variant == "lts" else variant

    def split_point(w: str) -> float:          # common endpoint of I(Lw) and I(Vw): computed from intervals only
        if w == "W":
            return theta
        l, r = interval("L" + w, theta, alpha, v), interval("V" + w, theta, alpha, v)
        return l[1] if abs(l[1] - r[0]) < abs(r[1] - l[0]) else l[0]

    return sorted(word_list(spec), key=split_point)


def vocabulary(theta: float, alpha: float, spec: int, variant: str = "little") -> Tuple[np.ndarray, List[str]]:
    """Ascending semantic values and word codes of the vocabulary of specificity ``spec``."""
    ws = word_list(spec)
    vals = np.array([nu(w, theta, alpha, variant) for w in ws])
    order = np.argsort(vals, kind="stable")
    return vals[order], [ws[i] for i in order]


def covering_radius(points: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    """sup_{s in [lo, hi]} min_t |s - p_t| for sorted points (nearest-point quantizer)."""
    p = np.sort(np.asarray(points, float))
    gaps = np.diff(p)
    return float(max(p[0] - lo, hi - p[-1], 0.5 * gaps.max() if gaps.size else 0.0))


def radius_ho_long(theta: float, alpha: float, spec: int, completed: bool = False) -> float:
    """Closed-form covering radius of the vocabulary of either consistent algebra, ``little`` or ``rather``
    (Proposition T0.4; both have the same level partition lengths and the same extreme cells).

    Vocabulary points are the interior boundaries of the level-(spec+2) partition.  Without the constants 0, 1
    the two extreme cells I(V^(spec+1) c-+) are covered from one side only:
        r = max(theta, 1-theta) * max(beta^(spec+1), m / 2),  m = alpha beta^spec if beta >= alpha else alpha^(spec+1);
    with the constants (completed lattice) r = max(theta, 1-theta) * max(alpha, beta)^(spec+1) / 2.
    """
    beta = 1.0 - alpha
    g = max(theta, 1.0 - theta)
    n = spec + 1
    if completed:
        return 0.5 * g * max(alpha, beta) ** n
    m = alpha * beta ** spec if beta >= alpha else alpha ** n
    return g * max(beta ** n, 0.5 * m)


def readable(word: str, variant: str = "little") -> str:
    """Human-readable phrase, e.g. ``VL+`` -> ``Very Little Large`` (``little``) or ``Very Rather Large``."""
    return " ".join(_NEG_NAME[variant] if c == "L" else _NAMES[c] for c in word)
