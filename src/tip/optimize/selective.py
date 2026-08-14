# -*- coding: utf-8 -*-
"""
selective.py — (beta) selectivity-first optimisation
==========================================
**Experimental.** Fully isolated from the established methods (classic, dual, gevd /
distributed) — nothing outside this file is touched.

Motivation (NEURON RESULTS §3 Q3): **both** the field-optimal and the AF-optimal montage fail
on selectivity — an off-target axon fires at a lower threshold than the target one
(off:orthoU threshold 511 versus target:axis 958). Two reasons: an off-target axon can lie
along **any direction**, and it can be recruited by **either of two mechanisms** (a termination
responds to field magnitude, a fibre of passage to the AF).
The established metrics only look at the envelope along the single target axis n, so they miss
this leakage entirely.

Approach — dual mechanism, worst direction:
  · target: isotropic Tmax (how easily it can be driven) plus AF along the axis n (fibres of
    passage).
  · off:    isotropic Tmax (which automatically covers the **worst-direction termination**)
            plus AF taken as the **worst over several directions** (a fibre of passage in any
            orientation).
  · selectivity = **whichever mechanism is weaker**: M2 = min(M2_field, M2_af) and
    M3 = max(M3_field, M3_af). Being focal in one mechanism while leaking through the other
    earns a low score, which is the honest outcome.
WP = w1·M1hat + w2·M2hat - w3·M3hat.
Note: this is not neuron-in-the-loop; it is a conservative field proxy.
"""
import itertools
import numpy as np
from .. import ti as TI
from .classic import _ratio_grid, channel_currents
from ..fieldsample import af_proj_elec


def _orthonormal(n):
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u) + 1e-30
    return n, u, np.cross(n, u)


def _gevd_pool(lf, target, pool, k, seed=7):
    """The top-k electrodes in the allowed pool by 3D GEVD importance."""
    from .multichannel import _gevd3d
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    ts = tgt if len(tgt) <= 700 else np.sort(rng.choice(tgt, 700, replace=False))
    ob = off if len(off) <= 1400 else np.sort(rng.choice(off, 1400, replace=False))
    Ft = np.stack([lf.elec_field(e, ts) for e in pool]); Fo = np.stack([lf.elec_field(e, ob) for e in pool])
    _, cstar, _ = _gevd3d(Ft, Fo)
    return [pool[i] for i in sorted(np.argsort(-np.abs(cstar))[:k].tolist())]


def dual_metrics(lf, target, best, n, pctl=50, tgt_scan=700, off_scan=1400, seed=7):
    """**Dual-mechanism, worst-direction** metrics for any two-pair montage.

    Returns dict(M2_field, M2_af, M3_field, M3_af, M1); the off-target AF is the worst over
    three directions (n, u, w).
    Used to compare selectivity on one yardstick (field_opt, af_opt and selective alike) and
    for GUI diagnostics."""
    n0, u, w = _orthonormal(n); dirs = [n0, u, w]
    (a, b) = best["ch1"]; (c, d) = best["ch2"]; i1, i2 = channel_currents(float(best.get("ratio", 1.0)))
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    tr = tgt if len(tgt) <= tgt_scan else np.sort(rng.choice(tgt, tgt_scan, replace=False))
    orr = off if len(off) <= off_scan else np.sort(rng.choice(off, off_scan, replace=False))

    def pf(idx):
        return (i1 * (lf.elec_field(a, idx) - lf.elec_field(b, idx)),
                i2 * (lf.elec_field(c, idx) - lf.elec_field(d, idx)))
    ft = TI.tmax(*pf(tr)); fo = TI.tmax(*pf(orr))
    At = af_proj_elec(lf, [a, b, c, d], n0, tr)
    at = 2.0 * np.minimum(np.abs(i1 * (At[0] - At[1])), np.abs(i2 * (At[2] - At[3])))
    ao = np.maximum.reduce([
        (lambda A: 2.0 * np.minimum(np.abs(i1 * (A[0] - A[1])), np.abs(i2 * (A[2] - A[3]))))(
            af_proj_elec(lf, [a, b, c, d], dd, orr)) for dd in dirs])
    return dict(
        M1=round(float(np.median(ft)), 4),
        M2_field=round(float((np.sqrt((ft ** 2).mean()) / max(np.sqrt((fo ** 2).mean()), 1e-12)) ** 2), 3),
        M2_af=round(float((np.sqrt((at ** 2).mean()) / max(np.sqrt((ao ** 2).mean()), 1e-12)) ** 2), 3),
        M3_field=round(100.0 * float((fo > np.percentile(ft, pctl)).mean()), 1),
        M3_af=round(100.0 * float((ao > np.percentile(at, pctl)).mean()), 1))


def optimize_selective(lf, target, allowed, n, weights=(0.5, 0.5, 0.5), pctl=50,
                       select_k=12, ratio_n=7, tgt_scan=700, off_scan=1400,
                       chunk=200, seed=7, progress=None, verbose=False):
    """(beta) Dual-mechanism, worst-direction selectivity optimisation.
    Returns a classic-style dict(ch1, ch2, ratio) plus diagnostics."""
    def prog(f, s):
        if progress:
            progress(f, s)

    n0, u, w = _orthonormal(n); dirs = [n0, u, w]
    pool = [e for e in allowed if lf.has(e)]
    names = pool if len(pool) <= select_k else _gevd_pool(lf, target, pool, select_k)
    K = len(names)
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    tr = tgt if len(tgt) <= tgt_scan else np.sort(rng.choice(tgt, tgt_scan, replace=False))
    orr = off if len(off) <= off_scan else np.sort(rng.choice(off, off_scan, replace=False))

    prog(0.35, "필드·AF 사전계산")
    Vt = np.stack([lf.elec_field(e, tr) for e in names])              # (K,Nt,3) for isotropic Tmax
    Vo = np.stack([lf.elec_field(e, orr) for e in names])             # (K,No,3)
    AFt = af_proj_elec(lf, names, n0, tr)                             # (K,Nt) target AF along n
    AFo = np.stack([af_proj_elec(lf, names, d, orr) for d in dirs])   # (3,K,No) off AF, 3 directions

    metas = []
    for a, b, c, d in itertools.combinations(range(K), 4):
        metas += [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    p1a = np.array([m[0][0] for m in metas]); p1b = np.array([m[0][1] for m in metas])
    p2a = np.array([m[1][0] for m in metas]); p2b = np.array([m[1][1] for m in metas])
    ratios = _ratio_grid(ratio_n); M = len(metas); R = len(ratios)
    oM1 = np.zeros((R, M)); oM2f = np.zeros((R, M)); oM2a = np.zeros((R, M))
    oM3f = np.zeros((R, M)); oM3a = np.zeros((R, M))

    prog(0.55, f"선택성 전수 ({M}×{R})")
    for ri, r in enumerate(ratios):
        i1, i2 = channel_currents(r)
        for s0 in range(0, M, chunk):
            sl = slice(s0, min(s0 + chunk, M))
            a1, b1, a2, b2 = p1a[sl], p1b[sl], p2a[sl], p2b[sl]
            # isotropic field Tmax, target and off-target
            ft = TI.tmax(i1 * (Vt[a1] - Vt[b1]), i2 * (Vt[a2] - Vt[b2]))       # (m,Nt)
            fo = TI.tmax(i1 * (Vo[a1] - Vo[b1]), i2 * (Vo[a2] - Vo[b2]))       # (m,No)
            # AF envelope along the target axis n
            at = 2.0 * np.minimum(np.abs(i1 * (AFt[a1] - AFt[b1])), np.abs(i2 * (AFt[a2] - AFt[b2])))
            # off-target AF per direction → take the worst
            ao = np.maximum.reduce([
                2.0 * np.minimum(np.abs(i1 * (AFo[di][a1] - AFo[di][b1])),
                                 np.abs(i2 * (AFo[di][a2] - AFo[di][b2]))) for di in range(len(dirs))])
            oM1[ri, sl] = np.median(ft, 1)
            oM2f[ri, sl] = (np.sqrt((ft ** 2).mean(1)) / np.maximum(np.sqrt((fo ** 2).mean(1)), 1e-12)) ** 2
            oM2a[ri, sl] = (np.sqrt((at ** 2).mean(1)) / np.maximum(np.sqrt((ao ** 2).mean(1)), 1e-12)) ** 2
            oM3f[ri, sl] = 100.0 * (fo > np.percentile(ft, pctl, axis=1)[:, None]).mean(1)
            oM3a[ri, sl] = 100.0 * (ao > np.percentile(at, pctl, axis=1)[:, None]).mean(1)

    M1 = oM1.ravel(); M2 = np.minimum(oM2f, oM2a).ravel(); M3 = np.maximum(oM3f, oM3a).ravel()
    mx1 = M1.max() or 1.0; mx2 = M2.max() or 1.0; mx3 = M3.max() or 1.0
    WP = weights[0] * M1 / mx1 + weights[1] * M2 / mx2 - weights[2] * M3 / mx3
    bi = int(np.argmax(WP)); ri, mi = bi // M, bi % M; r = float(ratios[ri])
    (a, b), (c, d) = metas[mi]
    best = dict(ch1=(names[a], names[b]), ch2=(names[c], names[d]), ratio=r, selective=True,
                M1=round(float(oM1[ri, mi]), 4),
                M2=round(float(min(oM2f[ri, mi], oM2a[ri, mi])), 3),
                M2_field=round(float(oM2f[ri, mi]), 3), M2_af=round(float(oM2a[ri, mi]), 3),
                M3_field=round(float(oM3f[ri, mi]), 1), M3_af=round(float(oM3a[ri, mi]), 1),
                pool=names)
    if verbose:
        print(f"[selective β] {best['ch1']}×{best['ch2']} r={r:.2f} | "
              f"M2_field={best['M2_field']} M2_af={best['M2_af']} (min={best['M2']}) | "
              f"M3_field={best['M3_field']} M3_af={best['M3_af']}")
    return best
