# -*- coding: utf-8 -*-
"""
ti.py — TI envelope / Tmax engine
=========================================
Grossman et al. maximum modulation envelope (verified to match the Sim4Life formula,
FOUNDATION §TI).

  modulation envelope along a direction n:  T_n(x) = 2·min(|n·E1|, |n·E2|)  (envelope=True)
  maximum modulation envelope (Tmax):       T_max(x) = max_n T_n(x)
      = 2·max_theta min(a1, a2), sampling 120 directions in the plane spanned by E1 and E2.

Sign-independent and direction-optimal. For optimisation the factor of 2 is irrelevant, so it
can be switched off with envelope=False. For a fixed direction use directional_env(E1,E2,n) —
that is the path for an anatomical axis such as nL.

★For ratio sweeps: gram3() + tmax_gram() are **mathematically identical** to tmax() but avoid
repeating the vector work every time the current ratio r changes (the classic exhaustive
search relies on this). See gram3 for details.
"""
import numpy as np
from . import config as C

_TH = np.linspace(0, np.pi, C.N_DIR, endpoint=False)
_CO = np.cos(_TH)[None, :]
_SI = np.sin(_TH)[None, :]


def tmax(E1, E2, envelope=True):
    """Direction-optimal Tmax = 2·max_n min(|n·E1|, |n·E2|). E1, E2 are (...,3); returns (...).

    Grossman closed form: identical to a 120-direction search but exact and about 10x faster.
    Batched inputs ((B,M,3)) are supported.
    Sort so |A| >= |B|, make the angle acute (the envelope is invariant under |·|), then use
    2·nb when nb <= na·cos(alpha) and 2·|A x B| / |A - B| otherwise."""
    E1 = np.asarray(E1, float); E2 = np.asarray(E2, float)
    n1 = np.linalg.norm(E1, axis=-1); n2 = np.linalg.norm(E2, axis=-1)
    sw = (n2 > n1)[..., None]
    A = np.where(sw, E2, E1); B = np.where(sw, E1, E2)                     # |A| ≥ |B|
    na = np.linalg.norm(A, axis=-1); nb = np.linalg.norm(B, axis=-1)
    dot = (A * B).sum(-1)
    B = np.where((dot < 0)[..., None], -B, B); dot = np.abs(dot)          # make acute (the envelope is invariant under |·|)
    cosa = dot / (na * nb + 1e-30)
    cross = np.linalg.norm(np.cross(A, B), axis=-1)
    diff = np.linalg.norm(A - B, axis=-1) + 1e-30
    t = np.where(nb <= na * cosa, nb, cross / diff)
    t = np.where((na < 1e-30) | (nb < 1e-30), 0.0, t)
    return 2.0 * t if envelope else t


def gram3(E1, E2):
    """(E1,E2) → the **four scalars** Tmax needs: (u, w, a, c2). E1, E2 are (...,3), each
    output is (...).

      u = |E1|²,  w = |E2|²,  a = |E1·E2|,  c2 = |E1×E2|² = u·w − a²

    Why separate them — tmax(E1, r·E2) has a closed form in these four alone. Scaling E2 by r
    turns them into (u, r²w, r·a, r²c2), i.e. **pure scalar scaling**. So sweeping the current
    ratio r needs the (...,3) vector work exactly once (this function), and the ratio loop is
    all (...) scalar arithmetic. In the classic exhaustive search (2.75 M metas x 13 ratios)
    that is a 5-6x difference.

    c2 is obtained as u·w - a² rather than by forming the cross product. The near-parallel
    regime, where that subtraction could cancel, is caught by the nb² <= |dot| branch in
    tmax_gram and never reaches the result — checked against tmax() on real leadfields, worst
    relative error 7.6e-14 in float64."""
    u = np.einsum('...i,...i->...', E1, E1)
    w = np.einsum('...i,...i->...', E2, E2)
    a = np.abs(np.einsum('...i,...i->...', E1, E2))
    return u, w, a, np.maximum(u * w - a * a, 0.0)


def tmax_gram(g, r=1.0, envelope=True):
    """Compute tmax(E1, r·E2) from the gram3(E1,E2) result `g` — the same closed form as
    tmax(). The |A| >= |B| ordering is handled by min/max and the acute-angle step is already
    done via a = |dot|:
      nb² = min(u, r²w),  |dot| = r·a,  |A×B|² = r²c2,  |A−B|² = u + r²w − 2r·a.
    2·nb when nb² <= |dot|, otherwise 2·|A x B| / |A - B|."""
    u, w, a, c2 = g
    r = float(r); r2 = r * r
    v = r2 * w; s = r * a
    m = np.minimum(u, v)
    den = u + v - 2.0 * s
    # On the m > s branch den > 0 is guaranteed (den = 0 iff the two vectors are equal, which
    # would imply m = s).
    t2 = np.where(m <= s, m, r2 * c2 / np.where(den > 0.0, den, np.inf))
    t = np.sqrt(t2)
    return 2.0 * t if envelope else t


def directional_env(E1, E2, n, envelope=True):
    """Modulation envelope along a fixed direction n (e.g. an anatomical axis).
    n is (3,) or (M,3)."""
    n = np.asarray(n, dtype=np.float64)
    if n.ndim == 1:
        p1 = np.abs(E1 @ n); p2 = np.abs(E2 @ n)
    else:
        p1 = np.abs((E1 * n).sum(1)); p2 = np.abs((E2 * n).sum(1))
    t = np.minimum(p1, p2)
    return 2.0 * t if envelope else t


def dir_grid(m=160):
    """Quasi-uniform directions on a hemisphere (Fibonacci). A hemisphere suffices because of
    the |·| symmetry. Returns (m,3)."""
    k = np.arange(m) + 0.5
    ph = np.arccos(1.0 - k / m)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


def tmax_timeavg(pairs, duties, m=256):
    """The **isotropic equivalent** Tmax for time-multiplexed (time-averaged) drive.

        max_n̂  Σ_k w_k · 2·min(|n̂·E1_k|, |n̂·E2_k|)

    ⚠ Do **not** compute this as `sum_k w_k · tmax(E1_k, E2_k)`. That adds maxima taken along
    a different direction in every slot, so it overestimates (it is an upper bound). Averaging
    over time per direction first, and only then maximising over directions, is what a neuron
    actually experiences.

    `tmax()` is exact because it has a closed form; the time average does not, so directions
    are sampled. With K = 1 we delegate straight to `tmax()` and keep the exact value (no
    sampling error).
    """
    pairs = list(pairs); duties = [float(w) for w in duties]
    if len(pairs) == 1:
        return duties[0] * tmax(*pairs[0])
    acc = None
    for v in dir_grid(m):
        s = None
        for w, (E1, E2) in zip(duties, pairs):
            e = w * directional_env(E1, E2, v)
            s = e if s is None else s + e
        acc = s if acc is None else np.maximum(acc, s)
    return acc


def carrier_max(E1, E2):
    """Peak carrier field, max |E1 ± E2| — the instantaneous safety metric."""
    return np.maximum(np.linalg.norm(E1 + E2, axis=1), np.linalg.norm(E1 - E2, axis=1))


def carrier_sq(E1, E2):
    """Summed per-channel power (|E1|² + |E2|²)/2 — the cumulative/heating safety metric
    (matches Cassara 2025)."""
    return 0.5 * ((E1 ** 2).sum(1) + (E2 ** 2).sum(1))
