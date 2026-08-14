# -*- coding: utf-8 -*-
"""
metrics.py — exposure quality metrics (matching the TIP / tip.lite definitions)
======================================================
As defined in migration §4-4 and the TIP manual (TI Analysis):

  M1 strength    = median(Tmax over the target)                        [V/m]
  M2 selectivity = (RMS_target / RMS_off)²        [dimensionless — invariant to current scale]
  M3 collateral  = fraction of off-target volume with Tmax above the target p-percentile
                                                                       [%] (lower is better)
  WP overall     = w1·M1hat + w2·M2hat - w3·M3hat (normalised within the candidate set)

Pure functions over Tmax arrays (target and off-target). Computing the fields is the job of
`leadfield` and `ti`.
"""
import numpy as np

from . import config as C


def M1(tmax_t):
    return float(np.median(tmax_t))


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def M2(tmax_t, tmax_o):
    ro = _rms(tmax_o)
    return float((_rms(tmax_t) / ro) ** 2) if ro > 0 else float("inf")


def voxel_volumes(lf, idx=None):
    """Voxel volume in mm³. The leadfield grid is **non-uniform**, so this differs per voxel.

    Measured: median 0.45 mm³, range 0.08-2.70, **max/min = 35.8x**.
    Same approach as `make_stn_mask.py` — the product of the axis-coordinate gradients.
    """
    import os
    g = np.load(C.inputs("gaxes1010.npz"))
    dx, dy, dz = (np.gradient(g[k]) for k in ("cx", "cy", "cz"))
    b = lf.bmask if idx is None else lf.bmask[idx]
    return dx[b[:, 0]] * dy[b[:, 1]] * dz[b[:, 2]]


def _wmedian(x, w):
    """Volume-weighted median (p = 50). See _wpercentile for the general p."""
    return _wpercentile(x, w, 50.0)


def _wpercentile(x, w, p):
    x = np.asarray(x, float); w = np.asarray(w, float)
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, p / 100.0)])


def M3(tmax_t, tmax_o, p=50, reference="target", wt=None, wo=None):
    """Fraction (%) of off-target **volume** with Tmax above the target p-percentile.
    Lower is better.

    ★2026-08-06 — passing `wt`/`wo` (voxel volumes) switches to **volume weighting**.
    Without them it stays a **voxel-count** fraction, as before.

    ⚠⚠ **2026-08-12 correction — do not use volume weighting. The default (count fraction)
    is the correct one.**

    For a while this said "volume weighting is right", based on the table below
    (`validate_tiplite_volw.py`, rank correlation 0.394 -> 0.745). **That comparison was
    invalid** — the reference was not tip.lite's published values but our own output. (The
    lesson: check where the comparison file came from before trusting it.) Comparing directly
    against tip.lite's published values with the rebuilt leadfield, **volume weighting is
    worse**: on a fully-solved left-thalamus montage the M3 ratio goes 1.770 -> 1.927. Across
    all 52 montages the count fraction sits at a median of 1.09, which is a good match.

    (the invalidated table, kept for the record)
    | variant | ratio | spread | rank corr. |
    |---|---:|---:|---:|
    | count fraction, count threshold  | 1.273 | 35.9% | 0.394 |
    | volume fraction, volume threshold | 1.056 | 22.1% | 0.745 |

    => the six places that use the count fraction (`optimize/classic.py` inner loop,
    `benchmark.py`, `fiberlead.py`, `activating.py`) are **correct, not a bug**. No unification
    work is needed. `wt`/`wo` stay for experiments only.
    """
    t = np.asarray(tmax_t); o = np.asarray(tmax_o)
    if reference == "target":
        thr = _wpercentile(t, wt, p) if wt is not None else np.percentile(t, p)
    else:  # 'brain' = target plus off-target (measured: not the tip.lite convention — off by 6x)
        if wt is not None and wo is not None:
            thr = _wpercentile(np.concatenate([t, o]), np.concatenate([wt, wo]), p)
        else:
            thr = np.percentile(np.concatenate([t, o]), p)
    if wo is None:
        return float(100.0 * np.mean(o > thr))
    wo = np.asarray(wo, float)
    return float(100.0 * wo[o > thr].sum() / wo.sum())


def all_metrics(tmax_t, tmax_o, p=50, reference="target", wt=None, wo=None):
    """M1, M2 and M3 together. `wt`/`wo` enable volume weighting for M3 — **normally leave
    them out** (see the M3 docstring).

    M2 should not be weighted either. Our M2 is `(RMS ratio)²` rather than the official
    "ratio of means", and against tip.lite's published values with the rebuilt leadfield it
    sits at a median of **0.96** over 52 montages — good enough to keep the current definition.
    """
    return dict(M1=M1(tmax_t), M2=M2(tmax_t, tmax_o),
                M3=M3(tmax_t, tmax_o, p=p, reference=reference, wt=wt, wo=wo))


def weighted_performance(rows, w=(0.5, 0.5, 0.5)):
    """rows: [{'M1':.., 'M2':.., 'M3':..}, ...] → WP list, normalised within the candidate set."""
    m1 = np.array([r["M1"] for r in rows]); m2 = np.array([r["M2"] for r in rows])
    m3 = np.array([r["M3"] for r in rows])
    w1, w2, w3 = w
    mx1, mx2, mx3 = m1.max() or 1, m2.max() or 1, m3.max() or 1
    return (w1 * m1 / mx1 + w2 * m2 / mx2 - w3 * m3 / mx3).tolist()
