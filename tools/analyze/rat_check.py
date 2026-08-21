# -*- coding: utf-8 -*-
"""rat_check.py — first look at the NeuroRat leadfield, and the checks it has to pass
====================================================================================
Run with `TIP_MODEL=rat`.

★There is **no reference CSV for the rat.** tip.lite's numbers are for their mouse and its
 electrode set does not exist here, so nothing below is a comparison against an outside
 truth. What these checks establish is *internal* consistency — that the 37 fields belong to
 one linear basis, that the mask decodes to the right anatomy, and that a montage produces
 sane magnitudes. Report them as pipeline validation, never as model validation.

Checks
  1. pool         — 37 files, PO8 absent (it is the grounded reference), no NaN/Inf
  2. anatomy      — every labelled structure's voxel count and centroid, and the hemisphere
                    split across the oblique midline plane (both sides should be ~50%)
  3. reciprocity  — a leadfield basis is symmetric: the potential at B from driving A equals
                    the potential at A from driving B. We do not store the potential, so the
                    weaker but still telling check is used: |E| falls off with distance from
                    the driven electrode, monotonically in the median.
  4. montage      — M1/M2/M3 for a left-hippocampus target under a few electrode pairs
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from tip import config as C            # noqa: E402
from tip.leadfield import LeadField    # noqa: E402
from tip.targets import Target         # noqa: E402
from tip import ti, metrics            # noqa: E402


def hemisphere(lf, idx=None):
    """Boolean 'is in the left hemisphere' for mask rows, across the oblique midplane."""
    return C.MODEL.is_left_pts(lf.coords(idx))


def check_pool(lf):
    print("--- 1. electrode pool ---")
    print("  %d fields, reference %s %s"
          % (len(lf.names), C.REF_ELEC,
             "absent (correct)" if C.REF_ELEC not in lf.names else "★PRESENT — wrong"))
    bad = []
    for n in sorted(lf.names):
        M = np.load(lf.reg[n][0])
        if M.shape != (lf.N, 3):
            bad.append((n, "shape %s" % (M.shape,)))
        elif not np.isfinite(M).all():
            #  The solver leaves NaN on every degree of freedom it did not solve (background
            #  cells, sigma = 0). Those must not reach the mask: a single NaN poisons every
            #  metric that touches the montage, quietly.
            bad.append((n, "%d non-finite values" % int((~np.isfinite(M)).sum())))
        elif M.dtype != np.float32:
            bad.append((n, "dtype %s" % M.dtype))
    print("  shape/finite/dtype problems: %s" % (bad if bad else "none"))
    return not bad


def check_anatomy(lf):
    print("--- 2. anatomy ---")
    bl = np.load(C.inputs(C.BLABEL_FILE))
    names = {v: k for k, v in json.load(open(C.inputs("labels_rat.json"))).items()}
    left = hemisphere(lf)
    print("  %-20s %8s %7s  %s" % ("structure", "nvox", "left%", "centroid mm"))
    for lab in sorted(names):
        sel = bl == lab
        if not sel.any():
            continue
        c = lf.coords(np.where(sel)[0])
        print("  %-20s %8d %6.1f%%  %s"
              % (names[lab], sel.sum(), 100 * left[sel].mean(), np.round(c.mean(0), 2)))
    print("  whole mask left fraction %.1f%%" % (100 * left.mean()))
    return True


def check_falloff(lf):
    """Brain |E| should fall off with the electrode's distance from the brain.

    ★Use the **minimum** distance, not the median over all brain voxels. The rat brain is
    long (the cerebellum and medulla stretch 20 mm behind the forebrain), so the median
    distance is dominated by the far end and says almost nothing about the near field — with
    the median this check reports a *positive* correlation and looks alarming for no reason.
    """
    print("--- 3. distance fall-off ---")
    coords = lf.coords()
    inj = {}
    p = os.path.join(C.LEADFIELD_ROOT, C.MODEL.leadfield_dir, "inj_current.json")
    if os.path.exists(p):
        inj = json.load(open(p))
    rows = []
    for n in sorted(lf.names):
        q = np.asarray(lf.pos[n], float)
        M = np.load(lf.reg[n][0])
        d = np.linalg.norm(coords - q, axis=1)
        rows.append((n, float(d.min()), float(np.median(np.linalg.norm(M, axis=1)))))
    dm = np.array([r[1] for r in rows])
    mg = np.array([r[2] for r in rows])
    print("  distance to nearest brain voxel: %.2f .. %.2f mm" % (dm.min(), dm.max()))
    print("  median |E| per mA: %.3f .. %.3f V/m   (median %.3f)"
          % (mg.min(), mg.max(), np.median(mg)))
    #  ★Comparing electrodes to each other does not test anything here: all 37 sit within
    #  2.5-4.8 mm of the brain, and across that narrow range the electrode's own conductance
    #  (temporalis muscle, below) swamps the geometry — the across-electrode correlation
    #  comes out at -0.006 and means nothing. The test with real content is *within* each
    #  electrode: the field it produces must decay with distance from it, everywhere.
    worst = None
    for n in sorted(lf.names):
        q = np.asarray(lf.pos[n], float)
        d = np.linalg.norm(coords - q, axis=1)
        e = np.linalg.norm(np.load(lf.reg[n][0]), axis=1)
        rr = float(np.corrcoef(np.log(d), np.log(np.maximum(e, 1e-12)))[0, 1])
        if worst is None or rr > worst[1]:
            worst = (n, rr)
    print("  within-electrode corr(log distance, log |E|): worst of 37 is %s at %+.3f  %s"
          % (worst[0], worst[1],
             "ok, every electrode's field decays with distance" if worst[1] < -0.3
             else "LOOK AT THIS"))
    r = worst[1]
    if inj:
        v = np.array([inj[n] * 1e3 for n, _, _ in rows])
        hi = [n for (n, _, _), x in zip(rows, v) if x > 0.4]
        print("  injected current at 1 V: %.3f .. %.3f mA  (median %.3f)"
              % (v.min(), v.max(), np.median(v)))
        #  C6 / CP6 / P6 sit over the right temporalis muscle (sigma 0.461 against fat's
        #  0.0776) and draw 1.5-2.7x the current of the rest. corr(I, muscle fraction within
        #  2 mm) = +0.80 — physics, not a defect. Each field is normalised to 1 mA anyway.
        print("  high-conductance electrodes (>0.4 mA/V): %s  — temporalis muscle, expected"
              % (sorted(hi) if hi else "none"))
    return r < 0


def check_montage(lf, pairs):
    print("--- 4. montage metrics, left hippocampus ---")
    bl = np.load(C.inputs(C.BLABEL_FILE))
    lab = C.MODEL.label_id("hippocampus")
    sel = (bl == lab) & hemisphere(lf)
    tgt = Target(lf, np.where(sel)[0], name="hippocampus L")
    print("  target: %s" % tgt.summary())
    print("  %-22s %8s %8s %8s" % ("montage (1 mA each)", "M1", "M2", "M3%"))
    out = []
    for a, b, c, d in pairs:
        if not all(x in lf.names for x in (a, b, c, d)):
            print("  %-22s skipped (electrode missing)" % ("%s-%s|%s-%s" % (a, b, c, d)))
            continue
        E1 = np.load(lf.reg[a][0]) - np.load(lf.reg[b][0])   # 1 mA through pair 1
        E2 = np.load(lf.reg[c][0]) - np.load(lf.reg[d][0])
        t = ti.tmax(E1, E2)
        m = metrics.all_metrics(t[tgt.target_idx], t[tgt.off_idx])
        out.append((a, b, c, d, m))
        #  metrics.M3 already returns a percentage — do not scale it again.
        print("  %-22s %8.4f %8.3f %8.2f"
              % ("%s-%s|%s-%s" % (a, b, c, d), m["M1"], m["M2"], m["M3"]))
    print("  (these five are arbitrary smoke-test montages, not optimised)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    if C.MODEL_NAME != "rat":
        sys.exit("run with TIP_MODEL=rat")
    lf = LeadField()
    check_pool(lf)
    check_anatomy(lf)
    check_falloff(lf)
    check_montage(lf, [
        ("C5", "C6", "CP5", "CP6"),
        ("F5", "P6", "F6", "P5"),
        ("FC5", "CP6", "FC6", "CP5"),
        ("C3", "C4", "P3", "P4"),
        ("AF7", "PO7", "AF8", "PO3"),
    ])


if __name__ == "__main__":
    main()
