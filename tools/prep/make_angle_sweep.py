# -*- coding: utf-8 -*-
"""
make_angle_sweep.py — generate the angle-sweep dataset (per `ANGLE_SWEEP_REQUEST.md`)
==============================================================================
Produces the input for testing, in the NEURON harness, whether time multiplexing by rotating
the field direction **keeps the target entrained while reducing the off-target load**.

Design decisions relative to the request
-------------------
1. **The drive is the electric field E.** The target is a thalamic relay neuron (a soma), so
   the relevant quantity is local E, not the activating function (request §1-2). The
   optimisation objective is the directional envelope
   `Tdir(n) = 2·min(|n·E1|, |n·E2|)`.
2. **All three E components are exported** (strongly recommended in request §2-2). A point
   neuron feels a different E depending on its orientation, so the receiving side must be able
   to project onto an arbitrary orientation.
3. **Ve is not exported.** It is undefined at a point (it belongs to cable models). Request
   §2-3 allowed point locations and §1-2 fixed the drive as E, so everything is expressed as E.
4. **Angles are chosen by generating many candidates and filtering.** The mandatory conditions
   of request §2-1 (target M1 within ±20%, distinct off-target patterns, distinct electrode
   combinations) are met **constructively** rather than hoped for.
5. **The electrode pool is not reduced per angle.** The union of each candidate direction's top
   electrodes forms one shared pool — reducing the pool separately would make the comparison
   between angles meaningless, a failure mode this project has hit repeatedly.

★Search structure — two stages, run in parallel (v2)
-----------------------------
A single-stage exhaustive search took 286 s per direction (114 min in total). Three changes
brought that down:

  (a) **Precompute the electrode-pair differences.** There are only C(40,2) = 780 distinct
      `|Pt[a] - Pt[b]|`, yet the exhaustive search recomputed them 274,170 metas x 13 ratios
      times. They are now built once per pair and indexed.
  (b) **Two stages.** A coarse point set (200 target / 400 off) ranks everything, and only the
      top KEEP are re-evaluated precisely on the full point set (700 / 1400).
  (c) **The 24 directions run in parallel.** They are independent, and a worker needs no
      leadfield — only the projected `(K, N)` arrays — so the IPC stays light.

**The risk of a two-stage search, and how it is controlled.** Narrowing the search space has
overturned conclusions here before. So the winner's **rank in the coarse stage is always
printed**. If that rank approaches KEEP, the winner was near the cut and KEEP must be raised.

**Normalisation rule**: the global WP constants g1, g2, g3 are taken over the **entire coarse
stage (M x R)** and **the same g** is reused in the refine stage. g is a candidate-independent
constant, so any choice is legitimate as long as the objective applies identically to all
candidates. Re-deriving g on the top subset would underestimate the maximum of M3 and inflate
the weight of the leakage term — the same trap as normalising per chunk.

Output: `neuron_case_thalamusL_angles.npz` plus `_stats.json`
"""
import itertools
import json
import os
import sys
import time

import numpy as np


# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")

sys.path.insert(0, os.path.join(REPO, "src"))

# ── conditions from request §3 ──
F1, F2 = 9000.0, 9130.0           # Lamos 2025 clinical protocol (delta f = 130 Hz)
N_TARGET_PT, N_OFF_PT = 150, 700  # request §2-3
N_CAND, N_KEEP = 40, 12           # candidate directions -> final angle count (request: 8-12)
M1_TOL = 0.20                     # request §2-1, condition 1
SEED = 20260804
TOP_ELEC = 10                     # top electrodes per direction; their union is the shared pool
NT_C, NO_C = 200, 400             # points in the coarse stage
NT_F, NO_F = 700, 1400            # points in the refine stage
KEEP = 6000                       # (meta, ratio) pairs carried from coarse to refine
WEIGHTS = (0.5, 0.5, 0.5)
PCTL = 50


# ───────────────── search kernel (runs inside a worker) ─────────────────
def _pairs(K):
    """The list of electrode pairs plus an (a, b) -> pair-index table."""
    pl = list(itertools.combinations(range(K), 2))
    pid = {p: i for i, p in enumerate(pl)}
    return np.array(pl, np.int32), pid


def _metas_as_pairs(K):
    """The three channel splits of a 4-electrode combination, as (pair index, pair index)."""
    _, pid = _pairs(K)
    out = np.empty((3 * len(list(itertools.combinations(range(K), 4))), 2), np.int32)
    w = 0
    for a, b, c, d in itertools.combinations(range(K), 4):
        for (x, y), (z, u) in (((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))):
            out[w, 0] = pid[(x, y)]; out[w, 1] = pid[(z, u)]; w += 1
    return out


def _absdiff(P, pl):
    """(K,N) projections -> (P,N) absolute pair differences, in float32."""
    return np.abs(P[pl[:, 0]] - P[pl[:, 1]]).astype(np.float32)


def _eval(At, Ao, metas, cur, sel=None, chunk=20000):
    """M1, M2 and M3 over the (meta, ratio) grid.
    With sel=None this is exhaustive, shape (3,R,M); otherwise it is (3, len(sel)) for the
    listed (mi, ri) pairs."""
    R = len(cur)
    if sel is None:
        M = len(metas)
        RAW = np.empty((3, R, M), np.float32)
        for s in range(0, M, chunk):
            q = metas[s:s + chunk]
            a0, a1 = At[q[:, 0]], At[q[:, 1]]
            o0, o1 = Ao[q[:, 0]], Ao[q[:, 1]]
            for ri, (i1, i2) in enumerate(cur):
                et = np.minimum(a0 * (2.0 * i1), a1 * (2.0 * i2))
                eo = np.minimum(o0 * (2.0 * i1), o1 * (2.0 * i2))
                RAW[0, ri, s:s + len(q)] = np.median(et, axis=1)
                RAW[1, ri, s:s + len(q)] = (np.sqrt((et ** 2).mean(1)) /
                                            np.maximum(np.sqrt((eo ** 2).mean(1)), 1e-12)) ** 2
                thr = et if PCTL == 50 else None
                t = np.median(thr, axis=1) if thr is not None else np.percentile(et, PCTL, axis=1)
                RAW[2, ri, s:s + len(q)] = 100.0 * (eo > t[:, None]).mean(1)
        return RAW
    mi, ri = sel
    out = np.empty((3, len(mi)), np.float32)
    for s in range(0, len(mi), chunk):
        q = metas[mi[s:s + chunk]]
        i1 = np.array([cur[r][0] for r in ri[s:s + chunk]], np.float32)[:, None]
        i2 = np.array([cur[r][1] for r in ri[s:s + chunk]], np.float32)[:, None]
        et = np.minimum(At[q[:, 0]] * (2.0 * i1), At[q[:, 1]] * (2.0 * i2))
        eo = np.minimum(Ao[q[:, 0]] * (2.0 * i1), Ao[q[:, 1]] * (2.0 * i2))
        out[0, s:s + len(q)] = np.median(et, axis=1)
        out[1, s:s + len(q)] = (np.sqrt((et ** 2).mean(1)) /
                                np.maximum(np.sqrt((eo ** 2).mean(1)), 1e-12)) ** 2
        t = np.median(et, axis=1) if PCTL == 50 else np.percentile(et, PCTL, axis=1)
        out[2, s:s + len(q)] = 100.0 * (eo > t[:, None]).mean(1)
    return out


def search_direction(job):
    """Two-stage search for one direction. job = (k, n, K, Ptc, Poc, Ptf, Pof, ratios, cur)."""
    k, n, K, Ptc, Poc, Ptf, Pof, ratios, cur = job
    t0 = time.time()
    pl, _ = _pairs(K)
    metas = _metas_as_pairs(K)
    M, R = len(metas), len(cur)

    # ── stage 1: exhaustive over the coarse point set ──
    RAW = _eval(_absdiff(Ptc, pl), _absdiff(Poc, pl), metas, cur)
    g = [max(float(RAW[i].max()), 1e-30) for i in range(3)]          # ★global normalisation, fixed
    WPc = (WEIGHTS[0] * RAW[0] / g[0] + WEIGHTS[1] * RAW[1] / g[1]
           - WEIGHTS[2] * RAW[2] / g[2])
    flat = WPc.ravel()
    keep = min(KEEP, flat.size)
    top = np.argpartition(-flat, keep - 1)[:keep]
    top = top[np.argsort(-flat[top])]                                # by coarse rank
    ri_t, mi_t = np.divmod(top, M)
    del RAW, WPc, flat

    # ── stage 2: refine the survivors on the full point set ──
    F = _eval(_absdiff(Ptf, pl), _absdiff(Pof, pl), metas, cur, sel=(mi_t, ri_t))
    WPf = (WEIGHTS[0] * F[0] / g[0] + WEIGHTS[1] * F[1] / g[1]
           - WEIGHTS[2] * F[2] / g[2])                               # ★the same g as stage 1
    w = int(np.argmax(WPf))
    a, b = pl[metas[mi_t[w], 0]]; c, d = pl[metas[mi_t[w], 1]]
    return dict(k=k, ndir=[float(x) for x in n], ei=(int(a), int(b), int(c), int(d)),
                ratio=float(ratios[ri_t[w]]),
                M1=float(F[0, w]), M2=float(F[1, w]), M3=float(F[2, w]),
                coarse_rank=int(w), n_keep=keep, n_meta=M * R,
                secs=round(time.time() - t0, 1))


# ──────────────────────────── main ────────────────────────────
def carrier_E(lf, best, idx, channel_currents):
    """Montage -> the two carrier E vectors, each (N,3), normalised to a total current of
    ITOTAL."""
    a, b = best["ch1"]; c, d = best["ch2"]
    i1, i2 = channel_currents(best["ratio"])
    return (i1 * (lf.elec_field(a, idx) - lf.elec_field(b, idx)),
            i2 * (lf.elec_field(c, idx) - lf.elec_field(d, idx)))


def main():
    from concurrent.futures import ProcessPoolExecutor
    from tip import LeadField, Target
    from tip import config as C
    from tip.optimize.classic import _ratio_grid, channel_currents

    rng = np.random.default_rng(SEED)
    LF = LeadField(); dd = LF.data_dir
    coords = LF.coords()
    TH = np.load(os.path.join(dd, "thalamus_mask.npy"))
    tgt = Target.from_mask(LF, TH[0], name="Thalamus L", off_subsample=22000)
    names = [e for e in LF.names if LF.has(e)]

    print("=" * 74)
    print("angle sweep - left thalamus, Tdir objective (two stages, parallel)")
    print("=" * 74)
    tp = coords[tgt.target_idx]
    print(f"  target {len(tgt.target_idx)} voxels · centre {tp.mean(0).round(2)}"
          f" · electrodes {len(names)}", flush=True)

    # Point sets for the optimisation. The coarse set is a subset of the fine one, which keeps
    # the two stages unbiased relative to each other.
    tr = np.sort(rng.choice(tgt.target_idx, min(NT_F, len(tgt.target_idx)), replace=False))
    orr = np.sort(rng.choice(tgt.off_idx, min(NO_F, len(tgt.off_idx)), replace=False))
    trc = np.sort(rng.choice(tr, NT_C, replace=False))
    orc = np.sort(rng.choice(orr, NO_C, replace=False))

    # ── candidate directions: Fibonacci on a hemisphere (n and -n are equivalent) ──
    i = np.arange(N_CAND) + 0.5
    phi = np.arccos(1 - i / N_CAND)                # 0..π/2
    theta = np.pi * (1 + 5 ** 0.5) * i
    cand = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta),
                     np.cos(phi)], 1)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    print(f"  {N_CAND} candidate directions (hemisphere Fibonacci) -> selecting {N_KEEP} that "
          f"meet the conditions", flush=True)

    # ── read the electrode fields once and build the shared pool ──
    t0 = time.time()
    Ft = np.stack([LF.elec_field(e, tr) for e in names])       # (K,Nt,3)
    Fo = np.stack([LF.elec_field(e, orr) for e in names])      # (K,No,3)
    pool = set()
    for n in cand:
        pool |= set(np.array(names)[np.argsort(-np.abs(Ft @ n).mean(1))[:TOP_ELEC]])
    pool = [e for e in names if e in pool]
    sub = [names.index(e) for e in pool]; K = len(pool)
    nM = 3 * len(list(itertools.combinations(range(K), 4)))
    ratios = _ratio_grid(7); cur = [channel_currents(float(r)) for r in ratios]
    print(f"  shared electrode pool of {K} ({time.time()-t0:.1f}s): {sorted(pool)}")
    print(f"  search space {nM:,} metas x {len(ratios)} ratios = {nM*len(ratios):,}"
          f" -> exhaustive coarse, then refine the top {KEEP:,}\n", flush=True)

    # index the coarse points by their position within the fine set
    ci = np.searchsorted(tr, trc); oi = np.searchsorted(orr, orc)
    jobs = [(k, cand[k], K,
             (Ft[sub][:, ci] @ cand[k]).astype(np.float32),
             (Fo[sub][:, oi] @ cand[k]).astype(np.float32),
             (Ft[sub] @ cand[k]).astype(np.float32),
             (Fo[sub] @ cand[k]).astype(np.float32),
             ratios, cur) for k in range(N_CAND)]
    del Ft, Fo

    nw = max(1, min(6, (os.cpu_count() or 4) - 2))
    print(f"  searching {N_CAND} directions across {nw} workers ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        res = list(ex.map(search_direction, jobs))
    res.sort(key=lambda r: r["k"])
    for r in res:
        r["ch1"] = (pool[r["ei"][0]], pool[r["ei"][1]])
        r["ch2"] = (pool[r["ei"][2]], pool[r["ei"][3]])
        print(f"  [{r['k']+1:2}/{N_CAND}] n={np.round(r['ndir'],3)}  "
              f"{r['ch1']}x{r['ch2']} r={r['ratio']:.3g}  "
              f"M1={r['M1']:.4f} M2={r['M2']:.2f} M3={r['M3']:.1f}%  "
              f"coarse rank {r['coarse_rank']}/{r['n_keep']}  [{r['secs']}s]")
    print(f"  total {time.time()-t0:.0f}s", flush=True)

    # ★Sanity check on the two-stage search: a winner near the cut means KEEP is too small
    rk = np.array([r["coarse_rank"] for r in res])
    print(f"\n  coarse->refine rank movement: median {int(np.median(rk))} · "
          f"worst {int(rk.max())} / kept {KEEP:,}  → "
          f"{'ample margin' if rk.max() < KEEP * 0.5 else '★KEEP must be raised'}")

    # ── sample the point-neuron locations ──
    # ★Drawn **before** the selection step, because the selection and the report have to judge
    # on the same points and the same orientation sample. When they differed, the ±20% verdicts
    # disagreed with each other — which actually happened.
    tgt_pt = np.sort(rng.choice(tgt.target_idx, N_TARGET_PT, replace=False))
    blab = np.load(os.path.join(dd, "blabel1010.npy"))
    neural = np.where(np.isin(blab, (C.LABEL_GM, C.LABEL_WM)))[0]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(coords[tgt.target_idx]).query(coords[neural])
    far = neural[d > 10.0]                           # more than 10 mm from the target
    off_pt = np.sort(rng.choice(far, N_OFF_PT, replace=False))
    idx = np.concatenate([tgt_pt, off_pt])
    is_t = np.zeros(len(idx), bool); is_t[:len(tgt_pt)] = True
    print(f"\n  {len(idx)} locations = {len(tgt_pt)} target + {len(off_pt)} off (>10 mm away)")

    # ── select using the conditions of request §2-1 ──
    # ★The yardstick changes here. The search objective M1 is `Tdir(n)`, which is **a different
    # yardstick for every angle** — each angle scores its best along its own n. Comparing
    # strength across angles needs a common yardstick, and for a population of point neurons
    # that is **Tdir averaged over random orientations** (request §1-2: the drive is E).
    # The orientation seed is fixed to 7, the same as in `report_angle_sweep.py`, so the two
    # sides produce identical numbers.
    Uo = np.random.default_rng(7).normal(size=(512, 3))
    Uo /= np.linalg.norm(Uo, axis=1, keepdims=True)
    Fp = np.stack([LF.elec_field(e, idx) for e in pool])          # (K,850,3)
    M1s = np.empty(N_CAND)
    for k, r in enumerate(res):
        a, b, c, dd_ = r["ei"]; i1, i2 = channel_currents(r["ratio"])
        E1 = i1 * (Fp[a][is_t] - Fp[b][is_t]); E2 = i2 * (Fp[c][is_t] - Fp[dd_][is_t])
        M1s[k] = np.median(2.0 * np.minimum(np.abs(E1 @ Uo.T),
                                            np.abs(E2 @ Uo.T)).mean(1))
        r["M1_dir"] = r["M1"]; r["M1"] = float(M1s[k])
    del Fp
    print(f"\n  strength re-measured as orientation-averaged Tdir (the common yardstick) - "
          f"distinct from the search objective Tdir(n)")
    print(f"  candidate M1 range {M1s.min():.4f}-{M1s.max():.4f}")

    # The condition is that the selected angles lie within ±20% of **their own** median.
    # Filtering against the median of all candidates breaks the condition, because the median
    # shifts once the selection is made — which is what happened. Instead, sort and scan every
    # contiguous window to find the **largest subset** that satisfies the condition directly
    # (n = 24, so even O(n^3) is free).
    order = np.argsort(M1s)
    best_sel = []
    for a in range(N_CAND):
        for b in range(a, N_CAND):
            grp, seen = [], set()
            for j in order[a:b + 1]:                 # de-duplicate, closest to the median first
                key = frozenset(res[j]["ch1"]) | frozenset(res[j]["ch2"])
                if key not in seen:
                    seen.add(key); grp.append(int(j))
            if not grp:
                continue
            m = np.median(M1s[grp])
            if (np.abs(M1s[grp] - m) / m).max() <= M1_TOL and len(grp) > len(best_sel):
                best_sel = grp
    _m0 = np.median(M1s[best_sel]) if best_sel else 1.0
    sel = sorted(best_sel, key=lambda j: abs(M1s[j] - _m0))[:N_KEEP]
    if len(sel) > 1:                                 # re-check the condition after the N_KEEP cut
        m = np.median(M1s[sel])
        while len(sel) > 4 and (np.abs(M1s[sel] - m) / m).max() > M1_TOL:
            sel.pop(int(np.argmax(np.abs(M1s[sel] - m))))
            m = np.median(M1s[sel])
    m = np.median(M1s[sel])
    print(f"  largest subset satisfying ±{M1_TOL*100:.0f}% and distinct electrode combinations "
          f"-> **{len(sel)}** · median {m:.4f} · "
          f"max deviation {(np.abs(M1s[sel]-m)/m).max()*100:.1f}%")
    if len(sel) < 8:
        print(f"  ⚠️ the request asks for 8-12. Raising N_CAND ({N_CAND}) yields more that pass")
    if len(sel) < 4:
        print("  ★selection failed - raise M1_TOL or add more candidates"); return 1

    base = int(np.argmax(M1s))                       # baseline = the continuous single montage
                                                     # (highest M1)
    print(f"  baseline (highest M1) = candidate {base}: {res[base]['ch1']}x{res[base]['ch2']}"
          f" M1={res[base]['M1']:.4f}", flush=True)

    # ── export ──
    ax = np.linalg.eigh((tp - tp.mean(0)).T @ (tp - tp.mean(0)))[1][:, -1]
    out = {"coords": coords[idx], "target": is_t,
           "labels": np.array([f"{'thal' if t else 'off '}{i:04d}"
                               for i, t in enumerate(is_t)]),
           "target_center": tp.mean(0), "axis": ax,
           "f1": F1, "f2": F2, "itotal_mA": float(C.ITOTAL),
           "note": np.array("E vectors for point neurons. Ve is excluded because it is "
                            "undefined at a point (request §1-2: the drive is E). "
                            "Units V/m, normalised to ITOTAL = 2 mA.")}
    stats = []
    for r, j in enumerate([base] + sel):
        nm = "baseline" if r == 0 else f"ang{r-1:03d}"
        b = res[j]
        E1, E2 = carrier_E(LF, b, idx, channel_currents)

        # ★Fix the polarity gauge: each angle has one free bit, and it is aligned to the
        # target's phase.
        # Swapping the two electrodes of channel 1 maps E1 -> -E1, which flips the envelope
        # phase sign s = sign((u·E1)(u·E2)) everywhere. M1, M2 and M3 are built on absolute
        # values, so they are **blind** to this bit and the optimiser never fixes it. Left
        # unaligned, rotating the angle also breaks the target entrainment and produces the
        # spurious conclusion that "rotation ruins the target".
        # For a random orientation u, E[s] = 1 - 2*theta/pi (theta = angle between E1 and E2),
        # so the orientation-averaged phase sign equals sign(E1·E2) — which is what is used to
        # decide, evaluated on the target.
        ct = np.clip((E1[is_t] * E2[is_t]).sum(1) /
                     np.maximum(np.linalg.norm(E1[is_t], axis=1) *
                                np.linalg.norm(E2[is_t], axis=1), 1e-30), -1, 1)
        if (1.0 - 2.0 * np.arccos(ct) / np.pi).mean() < 0:
            E1 = -E1
            b = dict(b, ch1=(b["ch1"][1], b["ch1"][0]))
            b["gauge_flipped"] = True
        else:
            b["gauge_flipped"] = False

        out[f"{nm}__E1vec"] = E1.astype(np.float32)
        out[f"{nm}__E2vec"] = E2.astype(np.float32)
        out[f"{nm}__ndir"] = np.array(b["ndir"])
        out[f"{nm}__montage"] = np.array(f"{tuple(b['ch1'])} x {tuple(b['ch2'])}")
        out[f"{nm}__montage_json"] = np.array(json.dumps(
            {"ch1": list(b["ch1"]), "ch2": list(b["ch2"]), "ratio": b["ratio"],
             "ndir": b["ndir"], "M1": b["M1"], "M1_dir": b["M1_dir"],
             "M2": b["M2"], "M3": b["M3"], "gauge_flipped": b["gauge_flipped"],
             "itotal_mA": float(C.ITOTAL), "imax_mA": float(C.IMAX),
             "f1": F1, "f2": F2}, ensure_ascii=False))
        n = np.array(b["ndir"])
        same = (np.sign(E1 @ n) == np.sign(E2 @ n))
        stats.append(dict(name=nm, cand=j, ratio=b["ratio"], M1=b["M1"],
                          M1_dir=b["M1_dir"], M2=b["M2"],
                          M3=b["M3"], ch1=list(b["ch1"]), ch2=list(b["ch2"]),
                          ndir=b["ndir"], coarse_rank=b["coarse_rank"],
                          gauge_flipped=b["gauge_flipped"],
                          same_t=float(same[is_t].mean()),
                          same_o=float(same[~is_t].mean())))
        print(f"  {nm:<9} {b['ch1']}x{b['ch2']} r={b['ratio']:.3g}"
              f"{' [polarity flipped]' if b['gauge_flipped'] else '                  '}  "
              f"same sign: target {same[is_t].mean()*100:.1f}% / off {same[~is_t].mean()*100:.1f}%")

    op = os.path.join(dd, "neuron_case_thalamusL_angles.npz")
    np.savez_compressed(op, **out)
    print(f"\n  saved {op} ({os.path.getsize(op)/1e6:.1f} MB)", flush=True)
    json.dump(stats, open(op.replace(".npz", "_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
