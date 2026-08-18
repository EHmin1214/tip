# -*- coding: utf-8 -*-
"""
huang.py — reproduction of the IFS array optimisation of Huang, Datta & Parra 2020
(J. Neural Eng. 17 036023)
================================================================================
A peer-reviewed reference method, kept for comparison against ours.

From the paper:
  · modulation depth along a direction e = 2·min(|e^T A s1|, |e^T A s2|), where s1 and s2 are
    the per-electrode current distributions of the two carriers.
  · **maximum modulation = the HD-TES fused solution** (s1 = s2, 100% modulation) — TI offers
    no strength advantage; this is proved analytically in eq. 8-9.
  · **focality optimisation** (eq. 18): maximise the target modulation depth subject to
    **off-target modulation power <= Pmax** (L2), an L1 current budget, and sum(s) = 0. The
    min() over off-target voxels is non-convex, hence SQP. Pmax titrates strength against
    focality.
    (A convex relaxation collapses back to the fused solution — paper Appendix A. The focality
    gain exists only in the non-convex min().)

Our implementation: scipy SLSQP with a softmin (a smooth approximation of the non-smooth
min), initialised from the fused solution (the same strategy as the paper), with the electrode
set reduced to the GEVD top-K. The direction e is the mean target projection (our n).
"""
import numpy as np
from scipy.optimize import minimize
from .. import config as C
from .selective import _gevd_pool


def _dir_lf(lf, els, idx, n):
    """Directional leadfield (len(idx), K): n·E_i at each voxel."""
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    return np.stack([lf.elec_field(e, idx) @ n for e in els], axis=1)


def optimize_huang(lf, target, allowed, n, pmax_rel=0.25, select_k=32,
                   tgt_scan=400, off_scan=600, budget=None, beta=25.0, seed=7,
                   init=None, maxiter=600, verbose=False):
    # A larger select_k means more array degrees of freedom and better deep focality (the
    # paper's point). 32 balances speed against convergence; beyond that the pool is reduced
    # to the GEVD top-K.
    """Huang 2020 focality IFS optimisation.

    `pmax_rel` caps the off-target modulation power, expressed as a multiple of the fused
    solution — smaller means more focal.
    Returns dict(huang=True, els, s1, s2, i_total, ...), i.e. the two carrier current
    distributions."""
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    #  ★프로토콜이 단일 진실. 예전엔 `C.ITOTAL` 을 직접 읽어, 규약을 바꿔도 이 방법만
    #  총전류 규약으로 고정돼 있었다.
    from .. import protocol as _P
    _p = _P.current()
    budget = (_p.budget if _p.current_norm == "total" else C.ITOTAL) if budget is None else budget
    pool = [e for e in allowed if lf.has(e)]
    els = pool if len(pool) <= select_k else _gevd_pool(lf, target, pool, select_k)
    K = len(els)
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    tr = tgt if len(tgt) <= tgt_scan else np.sort(rng.choice(tgt, tgt_scan, replace=False))
    orr = off if len(off) <= off_scan else np.sort(rng.choice(off, off_scan, replace=False))
    Tt = _dir_lf(lf, els, tr, n); Co = _dir_lf(lf, els, orr, n)      # (Nt,K),(No,K)
    v = Tt.mean(0)                                                    # (K,) target projection
                                                                      # (the paper's e)

    def smin(a, b):                                                  # numerically stable softmin
        m = np.minimum(a, b)
        return m - np.log(np.exp(-beta * (a - m)) + np.exp(-beta * (b - m))) / beta

    def sabs(x):
        return np.sqrt(x * x + 1e-9)

    def negobj(x):
        s1, s2 = x[:K], x[K:]
        return -2.0 * smin(sabs(v @ s1), sabs(v @ s2))

    def offpow(x):
        s1, s2 = x[:K], x[K:]
        mm = smin(sabs(Co @ s1), sabs(Co @ s2))
        return float((4.0 * mm * mm).mean())

    # Initialise from the fused solution (HD-TES maximum strength): put current on the
    # electrodes with the largest and smallest projection
    s0 = np.zeros(K); s0[np.argmax(v)] = budget / 2; s0[np.argmin(v)] = -budget / 2
    x0 = init if init is not None else np.concatenate([s0, s0])
    ref_off = offpow(np.concatenate([s0, s0]))                       # off power of the fused
                                                                     # solution, the reference
    pmax = pmax_rel * ref_off
    cons = [{"type": "eq", "fun": lambda x: x[:K].sum()},
            {"type": "eq", "fun": lambda x: x[K:].sum()},
            {"type": "ineq", "fun": lambda x: budget - sabs(x[:K]).sum()},    # L1(s1)≤budget
            {"type": "ineq", "fun": lambda x: budget - sabs(x[K:]).sum()},    # L1(s2)≤budget
            {"type": "ineq", "fun": lambda x: pmax - offpow(x)}]              # off-target modulation power
    res = minimize(negobj, x0, method="SLSQP", constraints=cons,
                   options={"maxiter": maxiter, "ftol": 1e-9})
    s1, s2 = res.x[:K], res.x[K:]
    i_total = 0.5 * (np.abs(s1).sum() + np.abs(s2).sum())            # total injected current
                                                                     # (what enters equals what exits)
    # Directional modulation metrics, in the same format as the distributed mode so the GUI
    # rendering is reused
    mt = 2.0 * np.minimum(np.abs(Tt @ s1), np.abs(Tt @ s2))
    mo = 2.0 * np.minimum(np.abs(Co @ s1), np.abs(Co @ s2))
    M1d = float(np.median(mt))
    M2d = float((np.sqrt((mt ** 2).mean()) / max(np.sqrt((mo ** 2).mean()), 1e-12)) ** 2)
    M3d = float(100.0 * (mo > np.percentile(mt, 50)).mean())
    c0 = {els[i]: round(float(s1[i]), 4) for i in range(K) if abs(s1[i]) > 1e-4}
    c1 = {els[i]: round(float(s2[i]), 4) for i in range(K) if abs(s2[i]) > 1e-4}
    best = dict(huang=True, currents={"ch0": c0, "ch1": c1}, direction=[float(x) for x in n],
                M1_dir=round(M1d, 4), M2_dir=round(M2d, 3), M3_dir=round(M3d, 1),
                target_mod=round(M1d, 4), i_total=round(float(i_total), 2),
                pmax_rel=float(pmax_rel), converged=bool(res.success),
                els=list(els), s1=s1, s2=s2)                        # s1/s2/els are for scripts; app.py pops them before serialising
    if verbose:
        print(f"[huang] pmax_rel={pmax_rel} conv={res.success} M1d={M1d:.4f} M2d={M2d:.2f} "
              f"I_tot={i_total:.2f} nE0={len(c0)} nE1={len(c1)}")
    return best
