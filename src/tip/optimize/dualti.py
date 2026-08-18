# -*- coding: utf-8 -*-
"""
dualti.py — 4-channel dual TI (2+2), jointly optimised
=============================================
Four channels (eight electrodes) = two independent TI systems:
  system A: ch1(f1) / ch2(f2)  → envelope env_A = Tmax(E_A1, E_A2)
  system B: ch3(f3) / ch4(f4)  → envelope env_B = Tmax(E_B1, E_B2)
The two bands (f1, f2 versus f3, f4) are separated, so the systems do not interfere — any
cross term beats at a high frequency.
Combined stimulation = env_A + env_B, reinforcing at the target. Frequency does not affect
amplitude under the quasi-static approximation, so the optimisation only chooses the eight
electrodes; the frequencies are a protocol label.

Optimisation is a **joint exhaustive search**. A and B are not found separately — greedy would
lock in "best plus runner-up". Instead **every (A,B) montage pair with disjoint electrodes** is
scored by the WP of the combined envelope, so the 2nd + 3rd montages can beat 1st + 4th.
An exhaustive search over eight electrodes at once explodes as C(K,4)·C(K-4,4), so when the
allowed set is large it is first reduced to the top `max_elec` by 3D-GEVD importance (this is
logged). Finally the two current ratios of the winning pair are polished jointly in 2-D.
**Total injected current is fixed at ITOTAL across both systems** (ITOTAL/2 each) with
<= IMAX per electrode, so this is directly comparable to classic at the same current budget.
"""
import itertools
import numpy as np
from .. import ti, metrics
from ..config import ITOTAL
from .classic import _ratio_grid, _cnorm, _cnorm_vec, channel_currents
from .multichannel import _gevd3d

#  ★per-system budget. Was a module constant `ITOTAL/2`, which silently forced a total-current
#  rule even when `config.CURRENT_NORM` said otherwise — the two systems then drew 1.00 mA
#  while classic drew 2.00 mA under the same "fair" table (measured 2026-08-18).
#  Now it follows `protocol.current()`: under a total rule each system gets half the budget;
#  under `max_channel` there is no total to split, so `None` lets `_cnorm` pin the larger
#  channel per system, exactly as classic does.
def dual_budget():
    """Per-system current budget, or None when the rule pins channels instead of a total."""
    from .. import protocol as _P
    p = _P.current()
    return (p.budget / 2.0) if p.current_norm == "total" else None


class _DualBudgetCompat(float):
    """Backwards compatibility: `DUAL_BUDGET` was imported as a float by gui/plan/benchmark.

    Those call sites pass it straight to `channel_currents(r, DUAL_BUDGET)`. Keeping it a real
    float keeps them working, while `dual_budget()` is the value the optimiser now uses."""


DUAL_BUDGET = _DualBudgetCompat(ITOTAL / 2.0)


# ---- app.py compatibility: the combined envelope, for reports and the 3D field ----
def _cfields(lf, m, idx):
    """The two channel fields of one dual-system montage, normalised to a per-system current
    of ITOTAL/2."""
    a, b = m["ch1"]; c, d = m["ch2"]; r = m.get("ratio", 1.0)
    i1, i2 = channel_currents(r, dual_budget())
    return (i1 * (lf.elec_field(a, idx) - lf.elec_field(b, idx)),
            i2 * (lf.elec_field(c, idx) - lf.elec_field(d, idx)))


def combined_env(lf, best, idx):
    """Combined dual-TI envelope = env_A + env_B (array)."""
    eA = ti.tmax(*_cfields(lf, best["systemA"], idx))
    eB = ti.tmax(*_cfields(lf, best["systemB"], idx))
    return eA + eB


def _env_diff(lf, ch1, ch2, r, idx):
    """Envelope of a single dual system, normalised to a per-system current of ITOTAL/2."""
    i1, i2 = channel_currents(r, dual_budget())
    return ti.tmax(i1 * (lf.elec_field(ch1[0], idx) - lf.elec_field(ch1[1], idx)),
                   i2 * (lf.elec_field(ch2[0], idx) - lf.elec_field(ch2[1], idx)))


def optimize_dual_ti(lf, target, allowed=None, weights=(0.5, 0.5, 0.5), pctl=50,
                     verbose=True, progress=None, max_elec=12, tgt_scan=700, off_scan=1400,
                     tgt_refine=4000, off_refine=6000, ratio_n=7, seed=42, **_):
    """
    Joint exhaustive search: every (A,B) montage pair with disjoint electrodes is scored
    globally by the combined WP.
    max_elec : cap on how many electrodes enter the exhaustive search; anything above it is
               reduced by 3D-GEVD importance.
    return   : dict(best={dual, systemA, systemB, M1/M2/M3, M1_A, M1_B}, allowed, n_pairs)
    """
    names = [e for e in (allowed if allowed is not None else list(lf.names)) if lf.has(e)]
    if len(names) < 8:
        raise ValueError(f"dual TI (4 channels) needs at least 8 electrodes (got {len(names)})")
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)

    def sub(a, k):
        return a if (not k or len(a) <= k) else np.sort(rng.choice(a, k, replace=False))
    tgt_s = sub(tgt, tgt_scan); off_s = sub(off, off_scan)

    # ---- reduce the electrode set: an 8-electrode exhaustive search explodes as
    #      C(K,4)·C(K-4,4), so keep the top `max_elec` by 3D-GEVD importance ----
    if len(names) > max_elec:
        Ft3 = np.stack([lf.elec_field(e, tgt_s) for e in names])
        Fo3 = np.stack([lf.elec_field(e, off_s) for e in names])
        _, cstar, _ = _gevd3d(Ft3, Fo3)
        keep = sorted(np.argsort(-np.abs(cstar))[:max_elec].tolist())
        n0 = len(names); names = [names[i] for i in keep]
        if verbose:
            print(f"[Dual TI] electrodes {n0}→{len(names)} (top 3D-GEVD importance, to keep the "
                  f"8-electrode exhaustive search feasible)")
    if progress: progress(0.2, "몽타주 사전계산")
    K = len(names)

    Ft = np.stack([lf.elec_field(e, tgt_s) for e in names])   # (K, Nt, 3)
    Fo = np.stack([lf.elec_field(e, off_s) for e in names])   # (K, No, 3)
    ratios = _ratio_grid(ratio_n)

    # ---- enumerate montages (4 electrodes in 2 pairs), pick each one's solo-optimal ratio,
    #      and precompute the envelopes used for combining ----
    metas = []   # (pa=(i,i), pb=(i,i))
    for a, b, c, d in itertools.combinations(range(K), 4):
        metas += [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    p1a = np.fromiter((m[0][0] for m in metas), int); p1b = np.fromiter((m[0][1] for m in metas), int)
    p2a = np.fromiter((m[1][0] for m in metas), int); p2b = np.fromiter((m[1][1] for m in metas), int)
    D1t = Ft[p1a] - Ft[p1b]; D2t = Ft[p2a] - Ft[p2b]   # (M, Nt, 3)
    D1o = Fo[p1a] - Fo[p1b]; D2o = Fo[p2a] - Fo[p2b]   # (M, No, 3)
    M = len(metas)

    # Compute each montage's solo metrics across the whole ratio grid and take the ratio that
    # maximises solo WP — the same criterion classic uses
    R = len(ratios)
    M1g = np.empty((M, R)); M2g = np.empty((M, R)); M3g = np.empty((M, R))
    for ri, r in enumerate(ratios):
        i1 = _cnorm(r, dual_budget()); i2 = r * i1
        et = ti.tmax(i1 * D1t, i2 * D2t); eo = ti.tmax(i1 * D1o, i2 * D2o)   # (M,Nt),(M,No)
        M1g[:, ri] = np.median(et, axis=1)
        rt = np.sqrt((et ** 2).mean(1)); ro = np.sqrt((eo ** 2).mean(1))
        M2g[:, ri] = (rt / np.maximum(ro, 1e-12)) ** 2
        thr = np.percentile(et, pctl, axis=1); M3g[:, ri] = 100.0 * (eo > thr[:, None]).mean(1)
    sx1 = M1g.max() or 1.0; sx2 = M2g.max() or 1.0; sx3 = M3g.max() or 1.0
    soloWP = weights[0] * M1g / sx1 + weights[1] * M2g / sx2 - weights[2] * M3g / sx3
    ridx = soloWP.argmax(1); rbest = ratios[ridx]
    ar = np.arange(M)
    M1solo = M1g[ar, ridx]; M2solo = M2g[ar, ridx]
    # Envelopes at the chosen ratio, (M, Nt) and (M, No), computed in one pass
    i1 = _cnorm_vec(rbest, dual_budget()); i2 = rbest * i1
    ET = ti.tmax(i1[:, None, None] * D1t, i2[:, None, None] * D2t)   # (M, Nt)
    EO = ti.tmax(i1[:, None, None] * D1o, i2[:, None, None] * D2o)   # (M, No)
    masks = np.fromiter(((1 << m[0][0]) | (1 << m[0][1]) | (1 << m[1][0]) | (1 << m[1][1]) for m in metas), int, M)
    del D1t, D2t, D1o, D2o

    # Normalise against a fixed reference: the population max is unstable against outliers, so
    # use the solo optimum instead. M3 is self-normalised on [0,100].
    M1ref = float(M1solo.max()) or 1.0; M2ref = float(M2solo.max()) or 1.0

    def jwp(m1, m2, m3):
        return weights[0] * m1 / M1ref + weights[1] * m2 / M2ref - weights[2] * m3 / 100.0

    # ---- joint: combined-envelope WP for every (A,B) pair with disjoint electrodes
    #      (i < j, vectorised over j) ----
    bestWP = -1e18; ia = ib = 0
    idxall = np.arange(M)
    for i in range(M):
        j = idxall[(idxall > i) & ((masks & masks[i]) == 0)]
        if not len(j):
            continue
        ct = ET[i] + ET[j]        # (nj, Nt)
        co = EO[i] + EO[j]        # (nj, No)
        thr = np.percentile(ct, pctl, axis=1)
        m1 = np.median(ct, axis=1)
        rt = np.sqrt((ct ** 2).mean(1)); ro = np.sqrt((co ** 2).mean(1))
        m2 = (rt / np.maximum(ro, 1e-12)) ** 2
        m3 = 100.0 * (co > thr[:, None]).mean(1)
        wp = jwp(m1, m2, m3)
        k = int(np.argmax(wp))
        if wp[k] > bestWP:
            bestWP = float(wp[k]); ia, ib = i, int(j[k])
        if progress and (i % 64 == 0): progress(0.4 + 0.45 * i / M, "조인트 전수")

    def _mont(idx):
        (x1, x2), (y1, y2) = metas[idx]
        return dict(ch1=(names[x1], names[x2]), ch2=(names[y1], names[y2]), ratio=float(rbest[idx]))
    A = _mont(ia); B = _mont(ib)

    # ---- joint 2-D ratio polish: maximise combined WP over (rA, rB) on the refine subsample ----
    if progress: progress(0.9, "비율 폴리싱")
    tgt_r = sub(tgt, tgt_refine); off_r = sub(off, off_refine)
    dA = _mont(ia); dB = _mont(ib)
    from scipy.optimize import minimize
    # Cache the four pair field differences so the polish loop does not call elec_field again
    cache = {}
    def _envc(tag, m, r, tr, of):
        if tag not in cache:
            cache[tag] = (m["ch1"], m["ch2"],
                          lf.elec_field(m["ch1"][0], tr) - lf.elec_field(m["ch1"][1], tr),
                          lf.elec_field(m["ch2"][0], tr) - lf.elec_field(m["ch2"][1], tr),
                          lf.elec_field(m["ch1"][0], of) - lf.elec_field(m["ch1"][1], of),
                          lf.elec_field(m["ch2"][0], of) - lf.elec_field(m["ch2"][1], of))
        _, _, d1t, d2t, d1o, d2o = cache[tag]
        i1, i2 = channel_currents(r, dual_budget())
        return ti.tmax(i1 * d1t, i2 * d2t), ti.tmax(i1 * d1o, i2 * d2o)

    def _neg_wp(x):
        etA, eoA = _envc("A", dA, x[0], tgt_r, off_r)
        etB, eoB = _envc("B", dB, x[1], tgt_r, off_r)
        ct = etA + etB; co = eoA + eoB
        m1 = float(np.median(ct)); m2 = metrics.M2(ct, co)
        m3 = 100.0 * float(np.mean(co > np.percentile(ct, pctl)))
        return -jwp(m1, m2, m3)

    res = minimize(_neg_wp, [A["ratio"], B["ratio"]], method="Nelder-Mead",
                   options={"xatol": 5e-3, "fatol": 1e-4, "maxiter": 120})
    rA, rB = float(res.x[0]), float(res.x[1])
    A["ratio"] = round(rA, 4); B["ratio"] = round(rB, 4)

    # ---- final metrics (refine subsample, polished ratios) ----
    etA, eoA = _envc("A", dA, rA, tgt_r, off_r); etB, eoB = _envc("B", dB, rB, tgt_r, off_r)
    ct = etA + etB; co = eoA + eoB
    fM1 = float(np.median(ct)); fM2 = metrics.M2(ct, co)
    fM3 = 100.0 * float(np.mean(co > np.percentile(ct, pctl)))
    M1_A = float(np.median(etA)); M1_B = float(np.median(etB))

    best = dict(dual=True, systemA=A, systemB=B,
                M1=fM1, M2=fM2, M3=fM3, M1_A=M1_A, M1_B=M1_B)
    if verbose:
        print(f"[Dual TI joint] electrodes {K} | montages {M} | best pair WP={bestWP:.3f}")
        print(f"  A {A['ch1']}×{A['ch2']}@{rA:.2f}  B {B['ch1']}×{B['ch2']}@{rB:.2f}")
        print(f"  solo M1: A={M1_A:.3f} B={M1_B:.3f} | combined M1={fM1:.3f} M2={fM2:.2f} "
              f"M3={fM3:.1f}%  (strength gain x{fM1/max(M1_A,1e-9):.2f})")
    if progress: progress(0.97, "완료")
    return dict(best=best, allowed=names, n_pairs=M,
                M1ref=M1ref, M2ref=M2ref, bestWP=bestWP)


def optimize_ncarrier(lf, target, N, allowed=None, weights=(0.5, 0.5, 0.5), pctl=50,
                      max_elec=None, tgt_scan=700, off_scan=1400, ratio_n=7, seed=42, verbose=True):
    """N-carrier TI = N independent TI systems combined, each at ITOTAL/N.

    Because the total current is fixed, sum_k env_k(ITOTAL/N) = (1/N) sum_k env_k(full), i.e.
    **the mean of the N montage envelopes** — the same quantity as an N-slot time average.
    N = 2 is seeded from the joint pair (top-200 pairwise); after that, disjoint montages are
    forward-added by maximising the combined-mean WP. The return uses the timemux-avg
    structure so `benchmark` can evaluate it automatically."""
    names = [e for e in (allowed if allowed is not None else list(lf.names)) if lf.has(e)]
    if max_elec is None:
        max_elec = min(len(names), 4 * N + 6)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)

    def sub(a, k):
        return a if (not k or len(a) <= k) else np.sort(rng.choice(a, k, replace=False))
    tgt_s = sub(tgt, tgt_scan); off_s = sub(off, off_scan)
    if len(names) > max_elec:
        Ft3 = np.stack([lf.elec_field(e, tgt_s) for e in names])
        Fo3 = np.stack([lf.elec_field(e, off_s) for e in names])
        _, cstar, _ = _gevd3d(Ft3, Fo3)
        keep = sorted(np.argsort(-np.abs(cstar))[:max_elec].tolist())
        names = [names[i] for i in keep]
    K = len(names)
    Ft = np.stack([lf.elec_field(e, tgt_s) for e in names])
    Fo = np.stack([lf.elec_field(e, off_s) for e in names])
    ratios = _ratio_grid(ratio_n)
    metas = []
    for a, b, c, d in itertools.combinations(range(K), 4):
        metas += [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    p1a = np.fromiter((m[0][0] for m in metas), int); p1b = np.fromiter((m[0][1] for m in metas), int)
    p2a = np.fromiter((m[1][0] for m in metas), int); p2b = np.fromiter((m[1][1] for m in metas), int)
    D1t = Ft[p1a] - Ft[p1b]; D2t = Ft[p2a] - Ft[p2b]
    D1o = Fo[p1a] - Fo[p1b]; D2o = Fo[p2a] - Fo[p2b]
    M = len(metas)
    R = len(ratios); M1g = np.empty((M, R)); M2g = np.empty((M, R)); M3g = np.empty((M, R))
    for ri, r in enumerate(ratios):
        i1, i2 = channel_currents(r)   # full ITOTAL budget
        et = ti.tmax(i1 * D1t, i2 * D2t); eo = ti.tmax(i1 * D1o, i2 * D2o)
        M1g[:, ri] = np.median(et, 1)
        rt = np.sqrt((et ** 2).mean(1)); ro = np.sqrt((eo ** 2).mean(1))
        M2g[:, ri] = (rt / np.maximum(ro, 1e-12)) ** 2
        thr = np.percentile(et, pctl, axis=1); M3g[:, ri] = 100.0 * (eo > thr[:, None]).mean(1)
    sx1 = M1g.max() or 1.0; sx2 = M2g.max() or 1.0; sx3 = M3g.max() or 1.0
    soloWP = weights[0] * M1g / sx1 + weights[1] * M2g / sx2 - weights[2] * M3g / sx3
    ridx = soloWP.argmax(1); rbest = ratios[ridx]; ar = np.arange(M)
    i1 = _cnorm_vec(rbest); i2 = rbest * i1
    ET = ti.tmax(i1[:, None, None] * D1t, i2[:, None, None] * D2t)   # (M,Nt) full-budget env
    EO = ti.tmax(i1[:, None, None] * D1o, i2[:, None, None] * D2o)
    masks = np.fromiter(((1 << m[0][0]) | (1 << m[0][1]) | (1 << m[1][0]) | (1 << m[1][1]) for m in metas), int, M)
    M1ref = float(M1g[ar, ridx].max()) or 1.0; M2ref = float(M2g[ar, ridx].max()) or 1.0

    def jwp(m1, m2, m3):
        return weights[0] * m1 / M1ref + weights[1] * m2 / M2ref - weights[2] * m3 / 100.0

    def _wp_avg(sum_t, sum_o, cand, n_now):
        ct = (sum_t[None, :] + ET[cand]) / n_now
        co = (sum_o[None, :] + EO[cand]) / n_now
        thr = np.percentile(ct, pctl, axis=1)
        m1 = np.median(ct, 1)
        rt = np.sqrt((ct ** 2).mean(1)); ro = np.sqrt((co ** 2).mean(1))
        m2 = (rt / np.maximum(ro, 1e-12)) ** 2
        m3 = 100.0 * (co > thr[:, None]).mean(1)
        return jwp(m1, m2, m3)

    idxall = np.arange(M); sol = np.median(ET, 1)   # solo M1 (for solo-best pick)
    sel = []; sel_mask = 0
    if N >= 2:   # seed from the joint pair (pairwise over the top-200 solo-WP montages)
        top = np.argsort(-soloWP[ar, ridx])[:min(M, 200)]
        best = (-1e18, top[0], top[0])
        for i in top:
            js = top[(masks[top] & int(masks[i])) == 0]; js = js[js != i]
            if not len(js):
                continue
            wp = _wp_avg(ET[i], EO[i], js, 2)
            k = int(np.argmax(wp))
            if wp[k] > best[0]:
                best = (float(wp[k]), int(i), int(js[k]))
        sel = [best[1], best[2]]
    else:
        sel = [int(np.argmax(jwp(np.median(ET, 1),
                                 (np.sqrt((ET ** 2).mean(1)) / np.maximum(np.sqrt((EO ** 2).mean(1)), 1e-12)) ** 2,
                                 100.0 * (EO > np.percentile(ET, pctl, 1)[:, None]).mean(1))))]
    for j in sel:
        sel_mask |= int(masks[j])
    sum_t = ET[sel].sum(0); sum_o = EO[sel].sum(0)
    while len(sel) < N:   # forward-add
        avail = idxall[(masks & sel_mask) == 0]
        if not len(avail):
            break
        wp = _wp_avg(sum_t, sum_o, avail, len(sel) + 1)
        j = int(avail[int(np.argmax(wp))])
        sel.append(j); sel_mask |= int(masks[j]); sum_t = sum_t + ET[j]; sum_o = sum_o + EO[j]
    comps = []
    for j in sel:
        (x1, x2), (y1, y2) = metas[j]
        comps.append(dict(ch1=(names[x1], names[x2]), ch2=(names[y1], names[y2]), ratio=float(rbest[j])))
    if verbose:
        print(f"[N-carrier N={N}] electrodes {K} | montages {M} | selected {len(comps)}", flush=True)
    return dict(timemux=True, components=comps, duties=[1.0 / len(comps)] * len(comps),
                combine="avg", ncarrier=len(comps))
