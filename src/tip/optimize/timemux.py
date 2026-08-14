# -*- coding: utf-8 -*-
"""
timemux.py — time-multiplexed TI
==========================================
Switch between K two-pair montages with duty cycles w_k. Under fast switching a neuron
perceives the **time average**:
    drive(x) = sum_k w_k · env_k(x),  env_k = directional envelope 2·min(|n·E1_k|, |n·E2_k|)
Every slot injects the full I_TOTAL, so the time-averaged total current is also I_TOTAL —
which keeps the comparison fair.

**Key design decision (from the 2026-07-29 re-evaluation)**: the time-multiplexing gain only
appears **in the high-strength regime**. Solving for a weighted sum (WP) parks the solution in
the low-strength regime where the gain is zero — that was the cause of the earlier null result.
So this is solved as a **constrained problem: maximise M2 (focality) subject to
M1 >= floor · (best single-montage M1)**.

⚠ The iso-strength frontier measurements quoted here ("+18-47% M2 over the best single montage
at p80-95") were later **rejected** as search-sample overfitting; see the fairness re-measurement
in the project notes. The constrained formulation itself still stands, and the surviving gain is
on the directional axis rather than in raw strength.

Electrode budget: electrodes **may be reused across slots**, so K slots can be built from
fewer than 4K electrodes.
"""
import itertools
import numpy as np
from scipy.optimize import minimize
from .classic import _ratio_grid, channel_currents
from .selective import _gevd_pool


def _pool(lf, target, allowed, n, select_k, ratio_n, tgt_scan, off_scan, seed):
    """Exhaustive two-pair montage candidates plus their directional envelopes over the target
    and off-target pools. Returns els, quad (M,4), ratio (M), ET, EO."""
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    pool = [e for e in allowed if lf.has(e)]
    els = pool if len(pool) <= select_k else _gevd_pool(lf, target, pool, select_k)
    K = len(els)
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    tr = tgt if len(tgt) <= tgt_scan else np.sort(rng.choice(tgt, tgt_scan, replace=False))
    orr = off if len(off) <= off_scan else np.sort(rng.choice(off, off_scan, replace=False))
    Pt = np.stack([lf.elec_field(e, tr) @ n for e in els])          # (K,Nt)
    Po = np.stack([lf.elec_field(e, orr) @ n for e in els])
    metas = []
    for a, b, c, d in itertools.combinations(range(K), 4):
        metas += [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    p1a = np.array([m[0][0] for m in metas]); p1b = np.array([m[0][1] for m in metas])
    p2a = np.array([m[1][0] for m in metas]); p2b = np.array([m[1][1] for m in metas])
    C1t = Pt[p1a] - Pt[p1b]; C2t = Pt[p2a] - Pt[p2b]
    C1o = Po[p1a] - Po[p1b]; C2o = Po[p2a] - Po[p2b]
    ETs, EOs, QD, RA = [], [], [], []
    quad = np.stack([p1a, p1b, p2a, p2b], axis=1)                   # (M,4) electrode indices
    for r in _ratio_grid(ratio_n):
        i1, i2 = channel_currents(r)
        ETs.append(2.0 * np.minimum(np.abs(i1 * C1t), np.abs(i2 * C2t)))
        EOs.append(2.0 * np.minimum(np.abs(i1 * C1o), np.abs(i2 * C2o)))
        QD.append(quad); RA.append(np.full(len(metas), r))
    return els, np.concatenate(QD), np.concatenate(RA), np.concatenate(ETs), np.concatenate(EOs)


def _metrics(dt, do, pctl=50):
    M1 = float(np.median(dt))
    M2 = float((np.sqrt((dt ** 2).mean()) / max(np.sqrt((do ** 2).mean()), 1e-12)) ** 2)
    M3 = float(100.0 * (do > np.percentile(dt, pctl)).mean())
    return M1, M2, M3


def _opt_duty(ETs, EOs, floor, pctl=50):
    """Optimise the duty cycles w (on the simplex) for the chosen slots:
    maximise M2 subject to M1 >= floor."""
    S = len(ETs)
    if S == 1:
        return np.ones(1)

    def norm(w):
        w = np.abs(w); return w / (w.sum() + 1e-12)

    def negM2(w):
        w = norm(w); return -_metrics(w @ ETs, w @ EOs, pctl)[1]

    res = minimize(negM2, np.ones(S) / S, method="SLSQP", bounds=[(0.0, 1.0)] * S,
                   constraints=[{"type": "eq", "fun": lambda w: np.abs(w).sum() - 1.0},
                                {"type": "ineq", "fun": lambda w: float(np.median(norm(w) @ ETs)) - floor}],
                   options={"maxiter": 120, "ftol": 1e-9})
    w = norm(res.x)
    if float(np.median(w @ ETs)) < floor * 0.98:      # constraint violated → blend toward the
                                                      # strongest slot
        j = int(np.argmax([np.median(e) for e in ETs]))
        for b in np.linspace(0, 1, 21):
            w2 = (1 - b) * w; w2[j] += b; w2 /= w2.sum()
            if float(np.median(w2 @ ETs)) >= floor:
                return w2
        w = np.zeros(S); w[j] = 1.0
    return w


def optimize_timemux(lf, target, allowed, n, K_slots=4, max_electrodes=8, m1_floor=0.9,
                     select_k=12, ratio_n=5, tgt_scan=500, off_scan=1200, cand_top=200,
                     beam=6, pctl=50, seed=7, progress=None, verbose=False):
    """Optimise a time-multiplexing schedule (constrained: maximise focality subject to
    strength >= floor).

    K_slots        : maximum number of slots
    max_electrodes : total electrode budget (reuse across slots is allowed)
    m1_floor       : fraction of the best single-montage M1 to hold (0.9 = the high-strength
                     regime, which is where any gain lives)
    Returns dict(timemux=True, slots=[{ch1, ch2, ratio, duty}], ...)."""
    def prog(f, s):
        if progress:
            progress(f, s)

    prog(0.3, "시분할 후보 생성")
    els, quad, ratio, ET, EO = _pool(lf, target, allowed, n, select_k, ratio_n,
                                     tgt_scan, off_scan, seed)
    M1s = np.median(ET, axis=1)
    floor = float(m1_floor * M1s.max())
    # **Mixing pool** = everything in the top strength band, including montages that miss the
    # floor on their own.
    # This matters: the median of a mixture can exceed that of each component (the slots cover
    # the target complementarily) — that is where the gain comes from.
    ok = np.argsort(-M1s)[:cand_top]
    feas = ok[M1s[ok] >= floor]                          # clears the floor alone = a 1-slot solution
    if len(feas) == 0:
        feas = ok[:1]
    esets = [frozenset(quad[i].tolist()) for i in range(len(quad))]

    # Seeds: the best-M2 standalone solutions plus the best-M2 entries of the mixing pool
    # (which can fail alone yet survive in a mixture)
    s1 = sorted(feas, key=lambda i: -_metrics(ET[i], EO[i], pctl)[1])[:beam]
    s2 = sorted(ok, key=lambda i: -_metrics(ET[i], EO[i], pctl)[1])[:beam]
    states = []
    for i in dict.fromkeys(list(s1) + list(s2)):
        m1, m2, m3 = _metrics(ET[i], EO[i], pctl)
        states.append(dict(idx=[i], w=np.ones(1), M2=m2, M1=m1, eset=set(esets[i])))
    valid = [s for s in states if s["M1"] >= floor]
    best = max(valid, key=lambda s: s["M2"]) if valid else dict(idx=[int(feas[0])], w=np.ones(1),
                                                                M2=0.0, eset=set(esets[int(feas[0])]))

    prog(0.5, "슬롯 탐색")
    for step in range(K_slots - 1):
        nxt = []
        for st in states:
            dt = st["w"] @ ET[st["idx"]]; do = st["w"] @ EO[st["idx"]]
            tn = (dt ** 2).mean(); on = (do ** 2).mean()
            # only candidates that fit the electrode budget — reuse is allowed, so judge by
            # the size of the union
            cand = [j for j in ok if j not in st["idx"]
                    and len(st["eset"] | esets[j]) <= max_electrodes]
            if not cand:
                continue
            cj = np.array(cand)
            top = None
            for b in np.linspace(0.05, 0.6, 12):         # 1-D blend screening, **with the
                                                         # strength constraint applied**
                Tb = (1 - b) * dt[None, :] + b * ET[cj]      # (nc,Nt)
                m1b = np.median(Tb, axis=1)
                okm = m1b >= floor                           # keep only what clears the floor
                                                             # (named apart from the outer `feas`)
                if not okm.any():
                    continue
                Ob = (1 - b) * do[None, :] + b * EO[cj]
                sc = (Tb ** 2).mean(1) / np.maximum((Ob ** 2).mean(1), 1e-18)
                sc = np.where(okm, sc, -1.0)
                o = int(np.argmax(sc))
                if sc[o] > 0 and (top is None or sc[o] > top[0]):
                    top = (float(sc[o]), int(cj[o]))
            if top is None:
                continue
            j = top[1]
            idx2 = st["idx"] + [j]
            w2 = _opt_duty(ET[idx2], EO[idx2], floor, pctl)
            m1, m2, m3 = _metrics(w2 @ ET[idx2], w2 @ EO[idx2], pctl)
            if m1 >= floor:
                nxt.append(dict(idx=idx2, w=w2, M2=m2, M1=m1, eset=st["eset"] | esets[j]))
        if not nxt:
            break
        nxt.sort(key=lambda s: -s["M2"])
        states = nxt[:beam]
        if states[0]["M2"] > best["M2"]:
            best = states[0]
        prog(0.5 + 0.35 * (step + 1) / max(K_slots - 1, 1), f"슬롯 {len(best['idx'])}")

    idx = best["idx"]; w = best["w"]
    keep = w > 1e-3
    idx = [idx[i] for i in range(len(idx)) if keep[i]]; w = w[keep]; w = w / w.sum()
    dt = w @ ET[idx]; do = w @ EO[idx]
    M1, M2, M3 = _metrics(dt, do, pctl)
    # Gain over the best single montage — compared against the best M2 **among singles that
    # meet the same strength floor**, so the comparison is iso-strength
    solo_best = max((_metrics(ET[i], EO[i], pctl)[1] for i in feas), default=0.0)
    slots, used = [], set()
    for k, i in enumerate(idx):
        a, b, c, d = quad[i]
        slots.append(dict(ch1=(els[a], els[b]), ch2=(els[c], els[d]),
                          ratio=round(float(ratio[i]), 3), duty=round(float(w[k]), 3)))
        used |= {els[a], els[b], els[c], els[d]}
    out = dict(timemux=True, slots=slots, K=len(slots), n_electrodes=len(used),
               electrodes_used=sorted(used), max_electrodes=int(max_electrodes),
               m1_floor=float(m1_floor), M1_dir=round(M1, 4), M2_dir=round(M2, 3),
               M3_dir=round(M3, 1), solo_M2=round(solo_best, 3),
               focality_gain=round(M2 / max(solo_best, 1e-9), 3),
               direction=[float(x) for x in np.asarray(n, float) / (np.linalg.norm(n) + 1e-30)])
    if verbose:
        print(f"[timemux] {len(slots)} slots · {len(used)} electrodes (budget {max_electrodes}) · "
              f"M1={M1:.4f} M2={M2:.2f} (best single {solo_best:.2f}, x{out['focality_gain']:.2f})")
    return out


def to_benchmark_form(res, combine="avg"):
    """Reshape an `optimize_timemux` result into the form `benchmark` understands.

    This module returns slots as `slots=[{ch1, ch2, ratio, duty}]`, but
    `benchmark._envelope_arrays` and `total_current` expect
    `components=[classic-style best, ...]` plus `duties=[...]` (benchmark.py:37, 92). As-is it
    therefore cannot go into `METHODS`. This thin layer bridges that gap and keeps every
    original key, so time-mux-specific information such as M2_dir and focality_gain remains
    available to the report.
    """
    slots = res.get("slots") or []
    if not slots:
        raise ValueError("`slots` is empty - is this really an optimize_timemux result?")
    out = dict(res)
    out["components"] = [{"ch1": tuple(s["ch1"]), "ch2": tuple(s["ch2"]),
                          "ratio": float(s["ratio"])} for s in slots]
    out["duties"] = [float(s["duty"]) for s in slots]
    out["combine"] = combine
    return out


def make_benchmark_method(combine="avg", **tkw):
    """Wrapper matching the `benchmark.register_method()` contract
    `fn(lf, target, allowed, n, weights, pctl) -> best`.

        from tip.benchmark import register_method
        from tip.optimize.timemux import make_benchmark_method
        register_method("timemux_opt", make_benchmark_method(K_slots=3, max_electrodes=8))

    (`benchmark._run_timemux` is a **different** method: it greedily picks K diverse classic
     montages. This one optimises the duty cycles with SLSQP under a strength floor. Register
     both and compare them.)
    """
    def _run(lf, target, allowed, n, weights, pctl):
        r = optimize_timemux(lf, target, allowed, n, pctl=pctl, verbose=False, **tkw)
        return to_benchmark_form(r, combine=combine)
    return _run
