# -*- coding: utf-8 -*-
"""
fiber.py — exhaustive search against a fibre-population objective (SETUP.md §8-3 Step 4)
================================================================
Exhaustively searches two-channel montages using a `FiberLeadField` drive (ve/field/af/gaf) as
the objective. The skeleton matches `optimize/selective.py` and `timemux.py`: 4-electrode
combinations x 3 pairings x a current-ratio grid.

The key difference: metrics are measured on **fibres**, not voxels.
  · the envelope is 2·min(|D1|, |D2|) computed per node,
  · the scalar for a fibre is the **maximum** along its trajectory (firing starts at the peak),
  · M1/M2/M3 are population statistics of those per-fibre values over the target and off pools.
  Reversing the reduction order (max first, min after) lets the two carriers peak at different
  places, which robs the min of its meaning.

Because every drive is linear in Ve, caching the per-electrode drives once
(`FiberLeadField.elec_drives`) reduces candidate evaluation to array arithmetic — which is what
makes an exhaustive search feasible at all.
"""
import itertools

import numpy as np

from .classic import _ratio_grid, channel_currents


def _screen_pool(fl, names, k, kind, kw):
    """Reduce the electrode pool to the k electrodes with the largest drive on the target fibres.

    ⚠ SETUP.md §6: narrowing the search space has overturned conclusions here before (the
      Huang select_k=16 case). The smaller k is, the riskier it gets — before drawing any
      conclusion, raise k and confirm the answer does not change. `select_k=None` disables the
      reduction entirely.

    ★**Never use this directly when comparing drives.** The score depends on `kind`, so each
    drive gets a different pool, and then a conclusion like "A_opt also wins on B's yardstick"
    can be an artefact of the differing search spaces. That happened: T10 was missing from the
    field pool, so gaf_opt was never even a candidate. For comparisons use `shared_pool()` and
    `compare_drives()`.
    """
    if k is None or len(names) <= k:
        return list(names)
    D = fl.elec_drives(kind, **kw)
    idx = [fl.idx[n] for n in names]
    score = np.abs(D[idx][:, fl.target]).max(-1).mean(-1)      # per-electrode target drive magnitude
    return [names[i] for i in np.argsort(-score)[:k]]


def shared_pool(fl, kinds=("field", "af", "gaf"), k=16, allowed=None, **kw):
    """The **union** of the top-k electrodes across several drives — a shared pool that makes
    cross-drive comparison fair.

    Because it contains whatever each drive prefers, no drive's optimum can be excluded from
    another drive's search space.
    """
    names = [n for n in (allowed or fl.names) if n in fl.idx]
    pool = set()
    for kd in kinds:
        pool |= set(_screen_pool(fl, names, k, kd, kw))
    return [n for n in names if n in pool]                     # preserve the original order


def _metrics_block(fd, tmask, pctl):
    """Per-fibre drive (M,F) → (M1, M2, M3), each (M,)."""
    et, eo = fd[:, tmask], fd[:, ~tmask]
    M1 = np.median(et, axis=1)
    rms_t = np.sqrt((et ** 2).mean(1)); rms_o = np.sqrt((eo ** 2).mean(1))
    M2 = (rms_t / np.maximum(rms_o, 1e-12)) ** 2
    M3 = 100.0 * (eo > np.percentile(et, pctl, axis=1)[:, None]).mean(1)
    return M1, M2, M3


def optimize_fiber(fl, allowed=None, kind="gaf", weights=(0.5, 0.5, 0.5), pctl=50,
                   select_k=16, ratio_n=7, chunk=120, verbose=False, progress=None,
                   **kw):
    """Find the two-channel montage with maximum WP over the fibre population.
    Returns a classic-style best dict.

    fl       : a FiberLeadField (its target_mask must be set)
    kind     : ve | field | af | gaf — which drive to optimise
    select_k : size of the reduced electrode pool (None = all). See the `_screen_pool` warning.
    The result also carries M1/M2/M3, `kind` and `pool`.
    """
    if not fl.target.any():
        raise ValueError("no target fibres - check the label_fibers result")
    names = [n for n in (allowed or fl.names) if n in fl.idx]
    names = _screen_pool(fl, names, select_k, kind, kw)
    K = len(names)
    if K < 4:
        raise ValueError(f"only {K} electrodes - at least 4 are required")

    D = fl.elec_drives(kind, **kw)[[fl.idx[n] for n in names]]      # (K,F,N)
    metas = []
    for a, b, c, d in itertools.combinations(range(K), 4):
        metas += [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    metas = np.array([[m[0][0], m[0][1], m[1][0], m[1][1]] for m in metas])
    ratios = _ratio_grid(ratio_n)
    if verbose:
        print(f"[fiber] electrodes {K} · candidates {len(metas)} x ratios {len(ratios)} "
              f"= {len(metas)*len(ratios)} montages · drive={kind}")

    best_wp, best = -np.inf, None
    allM1, allM2, allM3 = [], [], []
    for ri, r in enumerate(ratios):
        i1, i2 = channel_currents(r)
        fds = []
        for s in range(0, len(metas), chunk):                       # bound memory
            q = metas[s:s + chunk]
            C1 = i1 * (D[q[:, 0]] - D[q[:, 1]])                     # (m,F,N)
            C2 = i2 * (D[q[:, 2]] - D[q[:, 3]])
            env = 2.0 * np.minimum(np.abs(C1), np.abs(C2))          # per-node envelope
            fds.append(env.max(-1))                                 # per fibre = max along the trajectory
        fd = np.concatenate(fds)                                    # (M,F)
        M1, M2, M3 = _metrics_block(fd, fl.target, pctl)
        allM1.append(M1); allM2.append(M2); allM3.append(M3)
        if progress:
            progress((ri + 1) / len(ratios), f"drive={kind}")

    M1 = np.concatenate(allM1); M2 = np.concatenate(allM2); M3 = np.concatenate(allM3)
    w0, w1, w2 = weights
    WP = (w0 * M1 / max(M1.max(), 1e-30) + w1 * M2 / max(M2.max(), 1e-30)
          - w2 * M3 / max(M3.max(), 1e-30))
    bi = int(np.argmax(WP))
    ri, mi = divmod(bi, len(metas))
    a, b, c, d = metas[mi]
    best = {"ch1": (names[a], names[b]), "ch2": (names[c], names[d]),
            "ratio": float(ratios[ri]), "drive": kind,
            "M1": float(M1[bi]), "M2": float(M2[bi]), "M3": float(M3[bi]),
            "WP": float(WP[bi]), "pool": names}
    if verbose:
        print(f"[fiber] {kind}: {best['ch1']} x {best['ch2']} r={best['ratio']:.3g}"
              f"  M1={best['M1']:.4f} M2={best['M2']:.3f} M3={best['M3']:.1f}%")
    return best


def _all_metas(K):
    """4-electrode combinations x 3 pairings → (M,4) int32, with M = 3·C(K,4)."""
    c = np.array(list(itertools.combinations(range(K), 4)), np.int32)
    return np.concatenate([c[:, [0, 1, 2, 3]], c[:, [0, 2, 1, 3]], c[:, [0, 3, 1, 2]]])


def optimize_fiber_full(fl, allowed=None, kind="gaf", weights=(0.5, 0.5, 0.5), pctl=50,
                        coarse_fibers=150, coarse_ratios=(0.5, 1.0, 2.0),
                        n_refine=3000, ratio_n=7, mem_mb=400, seed=7,
                        verbose=True, progress=None, **kw):
    """Two-stage exhaustive search that **does not shrink the electrode pool** — the same
    strategy `classic.py` uses.

    `optimize_fiber(select_k=...)` shrinks the search space by discarding electrodes, which is
    exactly the failure mode this project keeps running into: raising the pool from 16 to 24
    changed the optimal montage for all three drives, i.e. it had not converged.
    `classic.py` instead keeps every electrode and subsamples the **evaluation points
    (voxels)**. This function does the same thing, subsampling **fibres**.

      1. coarse : **all** electrodes (70 → 3·C(70,4) ~ 2.75 M montages) x a few ratios, with
                  fibres reduced to `coarse_fibers`. This only produces a ranking.
      2. refine : re-evaluate the top `n_refine` candidates on **all fibres and the full ratio
                  grid**.

    Rough cost at 914 fibres and 70 electrodes: coarse ~20 min per drive, refine ~1 min.
    """
    if not fl.target.any():
        raise ValueError("no target fibres")
    names = [n for n in (allowed or fl.names) if n in fl.idx]
    K = len(names)
    D = fl.elec_drives(kind, **kw)[[fl.idx[n] for n in names]]        # (K,F,N)

    # Fibre subsample — the target/off ratio must be preserved or M2 and M3 become biased
    rng = np.random.default_rng(seed)
    ti_, oi_ = np.where(fl.target)[0], np.where(~fl.target)[0]
    frac = coarse_fibers / fl.n_fibers
    keep = np.sort(np.concatenate([
        rng.choice(ti_, max(2, min(len(ti_), int(round(len(ti_) * frac)))), replace=False),
        rng.choice(oi_, max(2, min(len(oi_), int(round(len(oi_) * frac)))), replace=False)]))
    Dc, tmask_c = D[:, keep], fl.target[keep]

    metas = _all_metas(K)
    M = len(metas)
    rc = np.asarray(coarse_ratios, float)
    if verbose:
        print(f"[fiber-full] electrodes {K} · montages {M:,} x ratios {len(rc)} "
              f"= {M*len(rc):,} · coarse fibres {len(keep)} "
              f"(target {int(tmask_c.sum())}) · drive={kind}")

    # ── ① coarse ──
    # ★Normalisation must be **global**. Dividing by m1.max() inside a chunk gives each chunk
    # its own reference, which makes the coarse ranking incomparable across chunks. That
    # happened here: the same optimum oscillated between rank 2474 and 11188, so it passed or
    # missed the refine cut at random.
    # → collect the raw metrics first, then normalise once against the global maxima.
    Fc, Nn = Dc.shape[1], Dc.shape[2]
    R = len(rc)
    chunk = max(64, int(mem_mb * 1e6 / (Fc * Nn * 8 * 6)))
    RAW = np.empty((3, R, M), np.float32)                # raw m1, m2, m3
    for s in range(0, M, chunk):
        q = metas[s:s + chunk]
        for ri, r in enumerate(rc):
            i1, i2 = channel_currents(float(r))
            env = 2.0 * np.minimum(np.abs(i1 * (Dc[q[:, 0]] - Dc[q[:, 1]])),
                                   np.abs(i2 * (Dc[q[:, 2]] - Dc[q[:, 3]])))
            a, b, c = _metrics_block(env.max(-1), tmask_c, pctl)
            RAW[0, ri, s:s + len(q)] = a
            RAW[1, ri, s:s + len(q)] = b
            RAW[2, ri, s:s + len(q)] = c
        if progress and (s // chunk) % 20 == 0:
            progress(0.85 * (s + len(q)) / M, f"coarse {kind}")
    g1 = max(float(RAW[0].max()), 1e-30)                 # global maximum
    g2 = max(float(RAW[1].max()), 1e-30)
    g3 = max(float(RAW[2].max()), 1e-30)
    wp_all = (weights[0] * RAW[0] / g1 + weights[1] * RAW[1] / g2
              - weights[2] * RAW[2] / g3)                # (R, M)
    bestWP = wp_all.max(0)                               # best over the ratios
    del RAW, wp_all
    top = np.argpartition(-bestWP, min(n_refine, M - 1))[:n_refine]
    if verbose:
        print(f"[fiber-full] coarse done → refining the top {len(top):,} candidates")

    # ── stage 2, refine: all fibres, full ratio grid ──
    ratios = _ratio_grid(ratio_n)
    qs = metas[top]
    rows = []
    rchunk = max(32, int(mem_mb * 1e6 / (fl.n_fibers * Nn * 8 * 6)))
    for s in range(0, len(qs), rchunk):
        q = qs[s:s + rchunk]
        for ri, r in enumerate(ratios):
            i1, i2 = channel_currents(float(r))
            env = 2.0 * np.minimum(np.abs(i1 * (D[q[:, 0]] - D[q[:, 1]])),
                                   np.abs(i2 * (D[q[:, 2]] - D[q[:, 3]])))
            a, b, c = _metrics_block(env.max(-1), fl.target, pctl)
            rows.append(np.stack([a, b, c, np.full(len(q), ri),
                                  np.arange(s, s + len(q))]))
        if progress:
            progress(0.85 + 0.15 * (s + len(q)) / len(qs), f"refine {kind}")
    R = np.concatenate(rows, axis=1)
    M1, M2, M3 = R[0], R[1], R[2]
    WP = (weights[0] * M1 / max(M1.max(), 1e-30) + weights[1] * M2 / max(M2.max(), 1e-30)
          - weights[2] * M3 / max(M3.max(), 1e-30))
    bi = int(np.argmax(WP))
    ri, qi = int(R[3, bi]), int(R[4, bi])
    a, b, c, d = qs[qi]
    best = {"ch1": (names[a], names[b]), "ch2": (names[c], names[d]),
            "ratio": float(ratios[ri]), "drive": kind,
            "M1": float(M1[bi]), "M2": float(M2[bi]), "M3": float(M3[bi]),
            "WP": float(WP[bi]), "n_electrodes": K, "n_montages": int(M * len(rc)),
            "coarse_rank": int(np.where(np.argsort(-bestWP) == top[qi])[0][0])
            if n_refine < M else 0}
    if verbose:
        print(f"[fiber-full] {kind}: {best['ch1']} x {best['ch2']} r={best['ratio']:.4g}"
              f"  M1={best['M1']:.4f} M2={best['M2']:.3f} M3={best['M3']:.1f}%"
              f"  (coarse rank {best['coarse_rank']})")
    return best


def compare_drives(fl, kinds=("field", "af", "gaf"), pool_k=16, allowed=None,
                   weights=(0.5, 0.5, 0.5), pctl=50, verbose=True, **kw):
    """Find each drive's optimal montage in **the same search space**, then cross-evaluate them.

    Returns {"opts": {kind: best}, "cross": {opt_kind: {eval_kind: metrics}},
          "wp": {eval_kind: {opt_kind: WP}}, "pool": [...]}

    Two things to be careful about:
      · Use the shared pool (`shared_pool`). A different pool per drive makes the comparison
        meaningless.
      · Normalise the cross WP **within the evaluating drive**. Comparing M1 and M2 separately
        decides the winner on an axis that is not the objective, which is misleading.
    """
    pool = shared_pool(fl, kinds, pool_k, allowed, **kw)
    if verbose:
        print(f"[compare] shared pool of {len(pool)}: {sorted(pool)}")
    opts = {k: optimize_fiber(fl, allowed=pool, kind=k, select_k=None,
                              weights=weights, pctl=pctl, verbose=verbose, **kw)
            for k in kinds}
    cross = {o: {e: fl.metrics(opts[o], kind=e, **kw) for e in kinds} for o in kinds}
    wp = {}
    for ev in kinds:
        m1 = np.array([cross[o][ev]["M1"] for o in kinds])
        m2 = np.array([cross[o][ev]["M2"] for o in kinds])
        m3 = np.array([cross[o][ev]["M3"] for o in kinds])
        w = (weights[0] * m1 / max(m1.max(), 1e-30)
             + weights[1] * m2 / max(m2.max(), 1e-30)
             - weights[2] * m3 / max(m3.max(), 1e-30))
        wp[ev] = {o: float(w[i]) for i, o in enumerate(kinds)}
    return {"opts": opts, "cross": cross, "wp": wp, "pool": pool}


def make_benchmark_method(fl, kind="gaf", precomputed=None, **fkw):
    """Build a wrapper matching the `benchmark.register_method()` contract.

    `benchmark` calls `fn(lf, target, allowed, n, weights, pctl) -> best`. The fibre objective
    uses a prebuilt `fl` instead of `lf`, `target` and `n`, so those are captured in a closure.
    The return is classic-shaped (ch1/ch2/ratio), which the existing evaluation and reporting
    machinery accepts unchanged.

        from tip.benchmark import register_method
        register_method("fiber_gaf", make_benchmark_method(fl, kind="gaf"))

    `precomputed` : pass an already-converged best dict and it is returned without searching.
    A 70-electrode exhaustive search takes ~25 min per drive, which cannot run inside the
    benchmark loop. This exists **to inject an offline exhaustive-search result**. Fairness is
    preserved: the benchmark scores "whatever montage each method proposes" on one yardstick,
    and a converged result is a better representative than a truncated search would be.
    """
    def _run(lf, target, allowed, n, weights, pctl):
        if precomputed is not None:
            return dict(precomputed)
        return optimize_fiber(fl, allowed=allowed, kind=kind,
                              weights=weights, pctl=pctl, **fkw)
    return _run
