"""Classical (Ho--Long) SQM of the two-generator / two-hedge algebra, as used in the published LTS papers.

Words are encoded as in pyLTS: generators ``-`` (Small) and ``+`` (Large), hedges ``L`` (Rather, negative) and
``V`` (Very, positive), neutral element ``W``; e.g. ``VL+`` = Very Rather Large.  With ``theta = fm(c-)``,
``alpha = mu(L)``, ``beta = 1 - alpha``:

    fm(c-) = theta, fm(c+) = 1 - theta, fm(hx) = mu(h) fm(x)
    nu(W) = theta,  nu(c-) = theta - alpha fm(c-),  nu(c+) = theta + alpha fm(c+)
    nu(hx) = nu(x) + Sgn(hx) fm(hx) (1 - omega(hx)),   omega(Vx) = beta,  omega(Lx) = alpha

For theta = alpha = 1/2 this coincides with the interval ("split-point") construction of
``haclustering.hedge_algebra``; for alpha != 1/2 they differ on terms that carry the negative hedge
(e.g. nu(L-) = theta (1 - alpha^2) here against theta (1 - alpha + alpha^2) there).  This module reproduces
the closed forms (4.1)-(4.7) of Hieu et al., J. Comput. Sci. Cybern. 36(2), 2020 and the ``pyLTS`` code.

Vocabulary of *specificity* s: generators and hedged terms of depth <= s plus ``W``  ->  2^(s+2) - 1 words
(s = 1: the 7 words Very Small ... Very Large).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np

LABELS_7 = {"V-": "Very Small", "-": "Small", "L-": "Rather Small", "W": "Middle", "L+": "Rather Large",
            "+": "Large", "V+": "Very Large"}
_NAMES = {"-": "Small", "+": "Large", "W": "Middle", "L": "Rather", "V": "Very"}


def _sign(x: str) -> int:
    if len(x) == 1:
        return -1 if x == "-" else 1
    s = _sign(x[1:])
    return s if x[0] == "V" else -s


def _fm(x: str, theta: float, alpha: float) -> float:
    if x == "W":
        return 0.0
    if len(x) == 1:
        return theta if x == "-" else 1.0 - theta
    mu = alpha if x[0] == "L" else 1.0 - alpha
    return mu * _fm(x[1:], theta, alpha)


def sqm(x: str, theta: float, alpha: float) -> float:
    if x == "W":
        return theta
    if x == "-":
        return theta - alpha * theta
    if x == "+":
        return theta + alpha * (1.0 - theta)
    beta = 1.0 - alpha
    omega = beta if x[0] == "V" else alpha
    return sqm(x[1:], theta, alpha) + _sign(x) * _fm(x, theta, alpha) * (1.0 - omega)


@lru_cache(maxsize=None)
def word_list(specificity: int) -> Tuple[str, ...]:
    """All words of the vocabulary of the given specificity (unsorted, deterministic order)."""
    words = ["-", "W", "+"]
    frontier = ["-", "+"]
    for _ in range(specificity):
        nxt = []
        for w in frontier:
            for h in ("L", "V"):
                nxt.append(h + w)
        words += nxt
        frontier = nxt
    return tuple(words)


def vocabulary(theta: float, alpha: float, specificity: int) -> Tuple[np.ndarray, List[str]]:
    """Sorted SQM values and word codes of the vocabulary (ascending ``nu``)."""
    ws = word_list(specificity)
    vals = np.array([sqm(w, theta, alpha) for w in ws])
    order = np.argsort(vals, kind="stable")
    return vals[order], [ws[i] for i in order]


def readable(word: str) -> str:
    """Human-readable phrase, e.g. ``VL+`` -> ``Very Rather Large``; the 7 base words use the paper's labels."""
    if word in LABELS_7:
        return LABELS_7[word]
    return " ".join(_NAMES[c] for c in word[:-1]) + " " + _NAMES[word[-1]]


def n_words(specificity: int) -> int:
    return 2 ** (specificity + 2) - 1
