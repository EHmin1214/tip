# -*- coding: utf-8 -*-
"""rat_montage_check.py — does the rat leadfield predict a real montage?
=======================================================================
The rat has no reference dataset, so this is the one validation available: solve a montage
**directly** in Sim4Life and compare it against what the leadfield says.

★Why it is not a formality here. The two heads were solved under different conventions:

    human   basis k = electrode k at 1 V, Cz at 0 V, **every other electrode absent**
    rat     basis k = electrode k at 1 V, **all 37 others at 0 V**  (EM LF port mode)

For the human, driving +i at A and -i at B puts zero net current through Cz, so the
reference drops out and `i·(LF[A] - LF[B])` is exact. For the rat it is not: in
`LF[A] - LF[B]` the other 36 electrodes carry whatever current the two basis states leave
behind, whereas a real montage has them **open**. They are only Ø0.25 mm, but they are PEC
and they sit on the scalp, so 36 of them shorted together is a low-impedance path across
the head that no experiment has.

This script measures the size of that error rather than arguing about it:

    direct   = Sim4Life with A at 1 V, B at 0 V, the other 36 floating, normalised to 1 mA
    leadfield= LF[A] - LF[B]                                          (already per mA)

and reports the magnitude ratio, the per-voxel direction agreement, and what it does to
M1/M2/M3 on a real target — because a 10% field error that leaves the ranking alone means
something very different from one that does not.

    python tools/s4l/rat_montage_check.py --a AF4 --b C3
"""
import os
import sys
import glob
import argparse

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("TIP_MODEL", "rat")

from tip import config as C            # noqa: E402
from tip.leadfield import LeadField    # noqa: E402
from tip.targets import Target         # noqa: E402
from tip import ti, metrics            # noqa: E402
import rat_extract as RX               # noqa: E402

DESK = os.path.dirname(REPO)
DEFAULT_RES = os.path.join(DESK, "s4l_projects", "rat_montage_test.smash_Results")


def direct_field(res_dir, smash):
    """Brain-voxel E per 1 mA from a directly solved montage, plus the injected current."""
    outs = [p for p in glob.glob(os.path.join(res_dir, "*_Output.h5"))
            if os.path.getsize(p) > 100e6]
    if not outs:
        sys.exit("no solved output in " + res_dir)
    out = max(outs, key=os.path.getsize)
    names = RX.component_names(smash)
    geo = RX.geometry(out, names)
    E, I = RX.extract_port(out, geo)
    return E * 1e-3 / I, I, out, geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="AF4", help="electrode held at 1 V")
    ap.add_argument("--b", default="C3", help="electrode held at 0 V")
    ap.add_argument("--res", default=DEFAULT_RES)
    ap.add_argument("--smash", default=os.path.join(DESK, "s4l_projects",
                                                    "rat_montage_test.smash"))
    a = ap.parse_args()

    if C.MODEL_NAME != "rat":
        sys.exit("run with TIP_MODEL=rat")
    lf = LeadField()
    for e in (a.a, a.b):
        if e not in lf.names:
            sys.exit(f"{e} is not in the rat leadfield")

    D, I, out, geo = direct_field(a.res, a.smash)
    P = (np.load(lf.reg[a.a][0]).astype(np.float64)
         - np.load(lf.reg[a.b][0]).astype(np.float64))
    print("direct solve : %s" % os.path.basename(out))
    print("  %s(1 V) -> %s(0 V), 36 electrodes floating" % (a.a, a.b))
    print("  injected current %.4f mA per volt" % (I * 1e3))
    print("  brain voxels %d  (leadfield %d)" % (len(D), len(P)))
    if len(D) != len(P):
        sys.exit("★ voxel count differs — the grids are not the same, stop here")

    nd, np_ = np.linalg.norm(D, axis=1), np.linalg.norm(P, axis=1)
    ok = (nd > 0) & (np_ > 0)
    cos = (D[ok] * P[ok]).sum(1) / (nd[ok] * np_[ok])
    ratio = np_[ok] / nd[ok]
    print("\n--- leadfield  vs  direct ---")
    print("  |E| direct    median %.4f V/m per mA   (max %.3f)" % (np.median(nd), nd.max()))
    print("  |E| leadfield median %.4f V/m per mA   (max %.3f)" % (np.median(np_), np_.max()))
    print("  magnitude ratio  median %.4f   p05 %.4f   p95 %.4f"
          % (np.median(ratio), *np.percentile(ratio, [5, 95])))
    print("  direction cosine median %.6f   p05 %.6f" % (np.median(cos), np.percentile(cos, 5)))
    print("  vector correlation %.6f"
          % np.corrcoef(D.ravel(), P.ravel())[0, 1])

    #  What it does to the numbers anyone would actually report. A single pair has no
    #  envelope, so use the pair against itself: Tmax of (E, E) is just |E|.
    bl = np.load(C.inputs(C.BLABEL_FILE))
    lab = C.MODEL.label_id("hippocampus")
    sel = (bl == lab) & C.MODEL.is_left_pts(lf.coords())
    tgt = Target(lf, np.where(sel)[0], name="hippocampus L")
    print("\n--- metrics on the left hippocampus (single pair, |E|) ---")
    print("  %-10s %8s %8s %8s" % ("", "M1", "M2", "M3%"))
    for nm, F in (("direct", D), ("leadfield", P)):
        t = np.linalg.norm(F, axis=1)
        m = metrics.all_metrics(t[tgt.target_idx], t[tgt.off_idx])
        print("  %-10s %8.4f %8.3f %8.2f" % (nm, m["M1"], m["M2"], m["M3"]))


if __name__ == "__main__":
    main()
