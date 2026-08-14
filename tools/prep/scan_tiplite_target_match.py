# -*- coding: utf-8 -*-
"""scan_tiplite_target_match.py — search all 414 targets for the one that reproduces the CSV
============================================================================
For the left-thalamus CSV, no definition in the thalamus family makes M2 and M3 agree
(M2 0.45-0.64, M3 2.8-3.7). Deciding whether the target was misidentified or something else is
wrong requires looking at **every candidate rather than narrowing the field** — a lesson this
project has learned repeatedly. All 414 reference targets are already mapped onto our grid, so
the same montage set is run against each target in turn to see whether any of them brings all
three metric ratios close to 1 simultaneously.

Tmax per montage is computed once over the whole brain; each target is then just an index slice.

Usage:
    python scan_tiplite_target_match.py <csv> [--top 15] [--min-vox 300]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tip import LeadField, ti                          # noqa: E402
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")
from tip import config as C                            # noqa: E402


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
        ix = {n: i for i, n in enumerate(hdr)}
        for line in f:
            p = line.strip().split(",")
            if len(p) < len(hdr):
                continue
            rows.append(dict(e=[p[ix[k]] for k in ("elA1", "elA2", "elB1", "elB2")],
                             a1=float(p[ix["a1"]]), a2=float(p[ix["a2"]]),
                             M1=float(p[ix["TI_Strength_lin"]]),
                             M2=float(p[ix["TI_Selectivity_lin"]]),
                             M3=100.0 * float(p[ix["TI_Collateral_lin"]])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-vox", type=int, default=300)
    a = ap.parse_args()

    lf = LeadField(); dd = lf.data_dir
    low = {n.lower(): n for n in lf.names}
    rows = load_csv(a.csv)
    m1 = np.array([r["M1"] for r in rows]); m2 = np.array([r["M2"] for r in rows])
    m3 = np.array([r["M3"] for r in rows])

    bl = np.load(os.path.join(dd, "blabel1010.npy"))
    N = len(bl)
    allidx = np.arange(N)
    gm = np.isin(bl, C.NEURAL_LABELS)
    gsamp = np.sort(np.random.default_rng(42).choice(np.where(gm)[0], 100000, replace=False))

    # whole-brain Tmax per montage (with a small cache of electrode fields)
    cache = {}
    def fld(nm):
        if nm not in cache:
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            cache[nm] = lf.elec_field(nm, allidx).astype(np.float32)
        return cache[nm]

    T = np.empty((len(rows), N), np.float32)
    keep = np.ones(len(rows), bool)
    for n, r in enumerate(rows):
        e = [low.get(x.lower(), x) for x in r["e"]]
        if not all(lf.has(x) for x in e):
            keep[n] = False; continue
        A, B, Cc, D = e
        T[n] = ti.tmax(r["a1"] * (fld(A) - fld(B)), r["a2"] * (fld(Cc) - fld(D)))
    cache.clear()
    T = T[keep]; m1, m2, m3 = m1[keep], m2[keep], m3[keep]
    print(f"CSV {os.path.basename(a.csv)} · {len(T)} montages (whole-brain Tmax computed)")

    Tg = T[:, gsamp]                       # the off sample is 100k GM voxels minus the target
    ingm = np.zeros(N, bool); ingm[gsamp] = True
    pos = -np.ones(N, np.int64); pos[gsamp] = np.arange(len(gsamp))

    z = np.load(os.path.join(dd, "masks_tiplite.npz"))
    rk = lambda x, y: float(np.corrcoef(np.argsort(np.argsort(-x)),
                                        np.argsort(np.argsort(-y)))[0, 1])
    out = []
    for k in z.files:
        idx = z[k]
        if len(idx) < a.min_vox:
            continue
        Tt = T[:, idx]
        o1 = np.median(Tt, axis=1)
        drop = pos[idx[ingm[idx]]]
        m = np.ones(len(gsamp), bool); m[drop] = False
        To = Tg[:, m]
        o2 = (Tt ** 2).mean(1) / (To ** 2).mean(1)      # metrics.M2 = (RMS ratio)^2
        o3 = 100.0 * (To > o1[:, None]).mean(1)
        if not np.all(np.isfinite(o1)) or o1.min() <= 0:
            continue
        r1, r2, r3 = np.median(o1 / m1), np.median(o2 / m2), np.median(o3 / m3)
        score = abs(np.log(r1)) + abs(np.log(r2)) + abs(np.log(max(r3, 1e-9)))
        out.append((score, k, len(idx), r1, r2, r3, rk(o1, m1), rk(o2, m2), rk(o3, m3)))

    out.sort()
    print(f"\n{len(out)} targets scanned · score = |ln M1r| + |ln M2r| + |ln M3r| "
          f"(0 is a perfect match)\n")
    print(f"{'#':>3} {'target':40}{'voxels':>8}{'score':>7}{'M1r':>7}{'M2r':>7}{'M3r':>7}"
          f"{'  r1':>6}{'r2':>6}{'r3':>6}")
    for n, (s, k, nv, r1, r2, r3, k1, k2, k3) in enumerate(out[:a.top], 1):
        print(f"{n:>3} {k:40}{nv:>7}{s:>7.3f}{r1:>7.3f}{r2:>7.3f}{r3:>7.3f}"
              f"{k1:>6.2f}{k2:>6.2f}{k3:>6.2f}")

    # Ratios alone discriminate poorly — plenty of targets happen to land on the right factor.
    # The real discriminator is the **sum of the three rank correlations**: does the target
    # reproduce the same ordering of montages?
    byrk = sorted(out, key=lambda r: -(r[6] + r[7] + r[8]))
    print(f"\nre-sorted by the sum of rank correlations (r1+r2+r3):")
    print(f"{'#':>3} {'target':40}{'voxels':>8}{'r sum':>7}{'  r1':>6}{'r2':>6}{'r3':>6}"
          f"{'M1r':>7}{'M2r':>7}{'M3r':>7}")
    for n, (s, k, nv, r1, r2, r3, k1, k2, k3) in enumerate(byrk[:a.top], 1):
        print(f"{n:>3} {k:40}{nv:>7}{k1+k2+k3:>6.2f}{k1:>6.2f}{k2:>6.2f}{k3:>6.2f}"
              f"{r1:>7.3f}{r2:>7.3f}{r3:>7.3f}")

    ref = {k: i for i, (_, k, *_) in enumerate(out)}
    print("\nfor reference - where the thalamus family ranks:")
    for k in ("split__Thalamus_Left", "split__Thalamus_Right",
              "ICBM_152__Thalamus_left", "ICBM_152__Thalamus_right"):
        if k in ref:
            s, kk, nv, r1, r2, r3, k1, k2, k3 = out[ref[k]]
            print(f"  #{ref[k]+1:<4} {k:36}{s:>7.3f}{r1:>7.3f}{r2:>7.3f}{r3:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
