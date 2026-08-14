# -*- coding: utf-8 -*-
"""
classic.py — classic two-channel TI, exhaustive search (Phase 1, the baseline oracle)
========================================================================
Exhaustively searches [two channel pairs + the channel current ratio] over the allowed
electrode set and returns the M1/M2/M3 Pareto front. For a small allowed set (the K electrodes
the user picked) this is the **global optimum**.

A montage is four electrodes split into two pairs (channel 1 = pair A at 1 mA, channel 2 =
pair B at r mA). Tmax is built on |·|, so it is invariant to polarity within a pair and to
swapping the channels — which leaves exactly three bipartitions of each 4-subset, times a
sweep over r.

Two-stage for speed: (1) rank every montage on a coarse subsample of target and off-target
voxels, then (2) recompute the Pareto set plus the top-WP candidates precisely against the
full off-target pool. `exact=True` collapses this to a single precise (slow) stage.

★The scan stage creates **no Python objects at all** (70 electrodes = 2,750,685 metas x 13
ratios = 35.75 M candidates). Combinations and metas are (n,4) int32 arrays and the metrics
are (R,B) float32 arrays; converting back to name tuples happens only for the few thousand
top candidates that go on to refine.
The ratio sweep runs through ti.gram3 / tmax_gram — one vector pass plus 13 scalar passes.
"""
import itertools
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from .. import ti, metrics
from ..config import ITOTAL, IMAX
from .. import config as _C          # imported as a module so the constants can be switched
                                     # at runtime (a `from ... import X` would bind at import)


def _cnorm(r, budget=None):
    """Current i1 on channel 1 (with i2 = r·i1). Tmax is first order in current, so i1 is
    simply the scale factor on M1; M2 and M3 are unaffected.

    Two normalisations (`config.CURRENT_NORM`; the reasoning lives in that file):
      · "max_channel" — larger channel = ICH_MAX → i1 = ICH_MAX/max(1,r). **tip.lite convention**
      · "total"       — i1 + i2 = ITOTAL         → i1 = budget/(1+r).    for fair cross-method
                                                                          comparison
    ⚠ Passing `budget=` explicitly **always** means a total-current budget (dual TI uses that
    to split a budget per system).
    The per-electrode cap IMAX applies in both cases."""
    r = float(r)
    cap = IMAX / max(1.0, r)
    if budget is None and getattr(_C, "CURRENT_NORM", "total") == "max_channel":
        return min(_C.ICH_MAX / max(1.0, r), cap)
    b = ITOTAL if budget is None else budget
    return min(b / (1.0 + r), cap)


def _cnorm_vec(r, budget=None):
    """Vectorised `_cnorm` over a batch of r."""
    r = np.asarray(r, float)
    cap = IMAX / np.maximum(1.0, r)
    if budget is None and getattr(_C, "CURRENT_NORM", "total") == "max_channel":
        return np.minimum(_C.ICH_MAX / np.maximum(1.0, r), cap)
    b = ITOTAL if budget is None else budget
    return np.minimum(b / (1.0 + r), cap)


def channel_currents(r, budget=None):
    """The actual channel currents (i1, i2): i1 = _cnorm(r, budget), i2 = r·i1
    (sum = budget, per electrode <= IMAX)."""
    i1 = _cnorm(r, budget)
    return i1, r * i1


def pareto_front(M1, M2, M3):
    """Non-dominated mask (bool) for (M1 up, M2 up, M3 down)."""
    obj = np.column_stack([-np.asarray(M1, float), -np.asarray(M2, float), np.asarray(M3, float)])
    n = len(obj); keep = np.ones(n, bool)
    for i in range(n):
        d = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        d[i] = False
        if d.any():
            keep[i] = False
    return keep


def weighted_performance(M1, M2, M3, w=(0.5, 0.5, 0.5)):
    M1, M2, M3 = (np.asarray(x, float) for x in (M1, M2, M3))
    mx1 = M1.max() or 1.0; mx2 = M2.max() or 1.0; mx3 = M3.max() or 1.0
    return w[0] * M1 / mx1 + w[1] * M2 / mx2 - w[2] * M3 / mx3


def hypervolume(M1, M2, M3, n=100000, seed=0):
    """Normalised hypervolume of the Pareto front — the tip.lite quality indicator.
    M1 up, M2 up, M3 down, each normalised to [0,1] by its own min and max.
    Returns 0-1; larger means a wider, better-converged front."""
    obj = np.column_stack([np.asarray(M1, float), np.asarray(M2, float), -np.asarray(M3, float)])
    lo = obj.min(0); span = obj.max(0) - lo; span[span < 1e-12] = 1.0
    o = (obj - lo) / span                     # 0 (worst) to 1 (best), per axis
    S = np.random.default_rng(seed).random((n, 3))
    dom = np.zeros(n, bool)
    for f in o:
        dom |= np.all(S <= f, axis=1)
    return float(dom.mean())


def _ratio_grid(n):
    r = np.linspace(0.3, 1.0, n)
    return np.unique(np.round(np.concatenate([r, 1.0 / r]), 4))


def _metrics(E1t, E2t, E1o, E2o, p=50):
    """M1 = median over the target, M2 = (RMS ratio)², M3 = fraction (%) of off-target voxels
    with Tmax above the target p-percentile.
    `p` is the isopercentile; the default 50 means the median."""
    tt = ti.tmax(E1t, E2t); to = ti.tmax(E1o, E2o); thr = np.percentile(tt, p)
    return float(np.median(tt)), metrics.M2(tt, to), 100.0 * float(np.mean(to > thr))


# ---------- building the exhaustive-search indices (without Python lists) ----------
def _combos4(K, names, pos, min_dist_mm):
    """Allowed 4-electrode combinations as an (n,4) int32 array. The itertools output is never
    materialised as a list — at 70 electrodes C(70,4) = 916,895 tuples alone would cost
    around 100 MB."""
    n4 = math.comb(K, 4)
    C4 = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(K), 4)),
                     np.int32, 4 * n4).reshape(n4, 4)
    if min_dist_mm <= 0 or not pos:
        return C4
    P = np.full((K, 3), np.nan)
    for i, nm in enumerate(names):
        if pos.get(nm) is not None:
            P[i] = pos[nm]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    # Electrodes with unknown coordinates (NaN) are exempt from the constraint — same
    # convention as the scalar `ok_dist`
    near = np.less(D, min_dist_mm, out=np.zeros(D.shape, bool), where=~np.isnan(D))
    keep = np.ones(len(C4), bool)
    for i, j in itertools.combinations(range(4), 2):
        keep &= ~near[C4[:, i], C4[:, j]]
    return C4[keep]


def _metas(C4):
    """4-combinations → (B,4) int32 = (ch1+, ch1-, ch2+, ch2-).
    Tmax is built on |·|, so it is invariant to polarity within a pair and to swapping the
    channels — three bipartitions cover everything.
    Ordering is combination-major, bipartition-minor (as in the original implementation)."""
    E = np.empty((len(C4), 3, 4), np.int32)
    E[:, 0] = C4[:, (0, 1, 2, 3)]
    E[:, 1] = C4[:, (0, 2, 1, 3)]
    E[:, 2] = C4[:, (0, 3, 1, 2)]
    return E.reshape(-1, 4)


def _scan_workers(B, nr, npt, n_jobs):
    """How many threads to scan with. numpy ufuncs release the GIL, so threads are enough —
    unlike processes they avoid copying the fields and paying spawn cost — and throughput
    saturates around 8 because of memory bandwidth.
    For small pools (the GUI default is <= 14 electrodes) creating the pool costs more than it
    saves, so use 1."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    if B * nr * npt < 3e8:
        return 1
    return min(8, max(1, (os.cpu_count() or 2) - 1))


def optimize_classic(lf, target, allowed=None, ratio_n=7, ratio_fine=25, max_pairs=60,
                     tgt_scan=2500, tgt_refine=5000, off_scan=3000, off_refine=20000, n_refine=60,
                     min_dist_mm=0.0, weights=(0.5, 0.5, 0.5), pctl=50,
                     exact=False, seed=42, verbose=True, progress=None, n_jobs=None):
    """
    allowed : list of allowed electrode names (None = all of them) — the "available electrodes"
    n_jobs  : scan threads (None = auto, 1 = sequential). The result does not depend on this.
    return  : dict(montages, best, n_eval, n_pareto, hypervolume, allowed)
    """
    names = [e for e in (allowed if allowed is not None else list(lf.names)) if lf.has(e)]
    if len(names) < 4:
        raise ValueError(f"fewer than 4 allowed electrodes: {names}")
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)
    def sub(a, k):
        return a if (not k or len(a) <= k) else np.sort(rng.choice(a, k, replace=False))

    tgt_s = tgt if exact else sub(tgt, tgt_scan)
    off_s = sub(off, off_refine) if exact else sub(off, off_scan)
    Fts = {e: lf.elec_field(e, tgt_s) for e in names}
    Fos = {e: lf.elec_field(e, off_s) for e in names}

    pos = getattr(lf, "pos", {}) or {}
    ratios = _ratio_grid(ratio_n)
    nr = len(ratios)
    Fta = np.stack([Fts[e] for e in names])   # (K, Nt, 3)
    Foa = np.stack([Fos[e] for e in names])   # (K, No, 3)
    MET = _metas(_combos4(len(names), names, pos, min_dist_mm))    # (B,4) electrode indices
    B = len(MET)
    if B == 0:
        raise ValueError(f"no 4-electrode combination satisfies min_dist_mm={min_dist_mm}")
    n_eval = B * nr

    # ── scan: write the metrics straight into (R,B) float32 arrays ────────────
    # Accumulating Python tuples would mean 35.75 M of them (~2 GB) at 70 electrodes; as
    # arrays it is 3 x 143 MB.
    # A single (B,N,3) array would be tens of GB, so montages are processed in chunks. The
    # reduction is per row, so the chunk size cannot affect the result (and therefore neither
    # can the thread count).
    npt = max(len(tgt_s), len(off_s), 1)
    nw = _scan_workers(B, nr, npt, n_jobs)
    CHUNK = max(256, int(1.2e8 / (npt * 24 * nw)))
    # float32 is enough — scan values only drive the **ranking** that selects refine
    # candidates, and the reported M1/M2/M3 are recomputed in float64 by refine. Under
    # `exact` these are the final numbers, so float64 is used instead.
    sdt = np.float64 if exact else np.float32
    S1 = np.empty((nr, B), sdt); S2 = np.empty((nr, B), sdt); S3 = np.empty((nr, B), sdt)

    def _scan(c0):
        c1 = min(B, c0 + CHUNK)
        q = MET[c0:c1]
        # ratio-independent vector work happens once per chunk; the ratio loop below is all
        # (b,N) scalar arithmetic
        gt = ti.gram3(Fta[q[:, 0]] - Fta[q[:, 1]], Fta[q[:, 2]] - Fta[q[:, 3]])   # (b,Nt)
        go = ti.gram3(Foa[q[:, 0]] - Foa[q[:, 1]], Foa[q[:, 2]] - Foa[q[:, 3]])   # (b,No)
        for ri, r in enumerate(ratios):
            tt = ti.tmax_gram(gt, r); to = ti.tmax_gram(go, r)
            med = np.median(tt, axis=1)
            thr = med if pctl == 50 else np.percentile(tt, pctl, axis=1)   # identical when p = 50
            rt = np.sqrt((tt ** 2).mean(1)); ro = np.sqrt((to ** 2).mean(1))
            S1[ri, c0:c1] = med * _cnorm(r)
            S2[ri, c0:c1] = (rt / np.maximum(ro, 1e-12)) ** 2
            S3[ri, c0:c1] = 100.0 * (to > thr[:, None]).mean(1)

    starts = range(0, B, CHUNK)
    if nw > 1:
        # numpy ufuncs release the GIL, so this scales with threads. Chunks write disjoint
        # columns, so no lock is needed.
        with ThreadPoolExecutor(nw) as ex:
            futs = [ex.submit(_scan, c0) for c0 in starts]
            for k, f in enumerate(as_completed(futs)):
                f.result()
                if progress: progress(0.1 + 0.6 * (k + 1) / len(futs), "전수탐색")
    else:
        for k, c0 in enumerate(starts):
            if progress: progress(0.1 + 0.6 * k / len(starts), "전수탐색")
            _scan(c0)

    # ---- refine stage: unless `exact`, recompute only the Pareto set plus the top-WP
    #      candidates against the full off-target pool ----
    if progress:
        progress(0.82, "정밀화")
    if not exact:
        # ★Normalisation constants are taken over the **entire** scan. Taking them per chunk
        # would scramble the ranking.
        # Calling weighted_performance() directly would promote to float64 and cost another
        # 3 x 286 MB, so the same expression is evaluated in float32 (these values only feed
        # the scan ranking).
        mx1 = float(S1.max()) or 1.0; mx2 = float(S2.max()) or 1.0; mx3 = float(S3.max()) or 1.0
        WP = (weights[0] * S1 / mx1).ravel()
        WP += (weights[1] * S2 / mx2).ravel()
        WP -= (weights[2] * S3 / mx3).ravel()
        ntop = min(WP.size, 1500)                        # top WP only — keeps the O(n²)
                                                         # Pareto pass affordable
        topw = (np.arange(WP.size) if ntop == WP.size
                else np.argpartition(-WP, ntop - 1)[:ntop])
        topw = topw[np.argsort(-WP[topw], kind="stable")]
        f1 = S1.ravel()[topw]; f2 = S2.ravel()[topw]; f3 = S3.ravel()[topw]
        psub = np.where(pareto_front(f1, f2, f3))[0]
        cand = sorted(set(topw[psub].tolist()) | set(topw[:n_refine].tolist()), key=lambda i: -WP[i])
        # Keep only the distinct electrode-pair configurations among the top candidates (by
        # WP, capped) and resample them on a dense ratio grid. Approximating a continuous
        # ratio this way makes the front denser and raises hypervolume.
        # This is the first point where name tuples are created — a few thousand of them,
        # not the full 35.75 M.
        cmi = np.asarray(cand, np.int64) % B                     # flat index → meta index
        pairs = list(dict.fromkeys(
            ((names[a], names[b]), (names[c], names[d])) for a, b, c, d in MET[cmi].tolist()
        ))[:max_pairs]
        del S1, S2, S3, WP, f1, f2, f3
        off_r = sub(off, off_refine); tgt_r = sub(tgt, tgt_refine)
        Ftr = {e: lf.elec_field(e, tgt_r) for e in names}
        For = {e: lf.elec_field(e, off_r) for e in names}
        fine = _ratio_grid(ratio_fine)
        E1t = np.stack([Ftr[p1[0]] - Ftr[p1[1]] for p1, p2 in pairs])   # (P, Ntr, 3) batch
        E2tu = np.stack([Ftr[p2[0]] - Ftr[p2[1]] for p1, p2 in pairs])
        E1o = np.stack([For[p1[0]] - For[p1[1]] for p1, p2 in pairs])
        E2ou = np.stack([For[p2[0]] - For[p2[1]] for p1, p2 in pairs])
        ref = []
        for r in fine:
            tt = ti.tmax(E1t, r * E2tu); to = ti.tmax(E1o, r * E2ou)
            thr = np.percentile(tt, pctl, axis=1)
            m1 = np.median(tt, axis=1) * _cnorm(r)
            rt = np.sqrt((tt ** 2).mean(1)); ro = np.sqrt((to ** 2).mean(1))
            m2 = (rt / np.maximum(ro, 1e-12)) ** 2
            m3 = 100.0 * (to > thr[:, None]).mean(1)
            for pi, (p1, p2) in enumerate(pairs):
                ref.append((p1, p2, float(r), float(m1[pi]), float(m2[pi]), float(m3[pi])))
        recs = ref
    else:
        # exact: every scan result is a final candidate. This is the one place that creates
        # B x R Python objects, so it stays heavy for large pools — `exact` is meant for small
        # pools only.
        pn = [((names[a], names[b]), (names[c], names[d])) for a, b, c, d in MET.tolist()]
        l1 = S1.tolist(); l2 = S2.tolist(); l3 = S3.tolist()
        del S1, S2, S3
        recs = [(pn[mi][0], pn[mi][1], float(ratios[ri]), l1[ri][mi], l2[ri][mi], l3[ri][mi])
                for ri in range(nr) for mi in range(B)]
        del pn, l1, l2, l3

    M1 = np.array([x[3] for x in recs]); M2 = np.array([x[4] for x in recs])
    M3 = np.array([x[5] for x in recs])
    WP = weighted_performance(M1, M2, M3, weights); par = pareto_front(M1, M2, M3)
    montages = [dict(ch1=x[0], ch2=x[1], ratio=x[2], M1=x[3], M2=x[4], M3=x[5],
                     WP=float(WP[i]), pareto=bool(par[i])) for i, x in enumerate(recs)]
    hv = hypervolume(M1[par], M2[par], M3[par]) if par.sum() >= 2 else 0.0
    bi = int(np.argmax(WP)); best = montages[bi]
    if not exact and len(recs) > 1:      # polish the ratio of the best montage continuously,
                                         # removing the grid residual so it matches a
                                         # continuous ratio
        from scipy.optimize import minimize_scalar
        p1, p2 = best["ch1"], best["ch2"]
        tp = sub(tgt, tgt_refine); op = sub(off, off_refine)
        E1t = lf.elec_field(p1[0], tp) - lf.elec_field(p1[1], tp); E2tu = lf.elec_field(p2[0], tp) - lf.elec_field(p2[1], tp)
        E1o = lf.elec_field(p1[0], op) - lf.elec_field(p1[1], op); E2ou = lf.elec_field(p2[0], op) - lf.elec_field(p2[1], op)
        mx1 = M1.max() or 1.0; mx2 = M2.max() or 1.0; mx3 = M3.max() or 1.0

        def _neg(r):
            mm = _metrics(E1t, r * E2tu, E1o, r * E2ou, pctl); s = _cnorm(r)
            return -(weights[0] * mm[0] * s / mx1 + weights[1] * mm[1] / mx2 - weights[2] * mm[2] / mx3)
        rr = float(minimize_scalar(_neg, bounds=(0.2, 5.0), method="bounded", options={"xatol": 1e-3}).x)
        mm = _metrics(E1t, rr * E2tu, E1o, rr * E2ou, pctl); s = _cnorm(rr)
        best = dict(best, ratio=round(rr, 4), M1=mm[0] * s, M2=mm[1], M3=mm[2])
    if verbose:
        print(f"[Classic 2ch] allowed {len(names)} | evaluated {n_eval} "
              f"| refined {len(recs)} | pareto {int(par.sum())} | HV {hv:.3f}"
              + ("  (exact)" if exact else ""))
    return dict(montages=montages, best=best,
                n_eval=n_eval, n_pareto=int(par.sum()), hypervolume=hv, allowed=names)


def format_montage(m):
    a, b = m["ch1"]; c, d = m["ch2"]; r = m["ratio"]
    i1, i2 = channel_currents(r)
    return (f"E1 {a}(+)/{b}(−)@{i1:.2f} × E2 {c}(+)/{d}(−)@{i2:.2f}  "
            f"| M1 {m['M1']:.3f}  M2 {m['M2']:.2f}  M3 {m['M3']:.1f}%  WP {m['WP']:+.3f}")
