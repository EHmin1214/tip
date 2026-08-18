# -*- coding: utf-8 -*-
"""fit_rat_midplane.py — can the rat midsagittal plane be improved?  Measured answer: no.
=========================================================================================
Run with `TIP_MODEL=rat`. **This tool changes nothing.** It exists so nobody spends another
evening re-deriving what is written here.

The plane in `models.py` came out of the electrode-placement run: normal = the mean of 50
left/right tissue-pair differences over the whole body, orthogonalised against the AP axis;
offset fitted from nine midline structures, whose spread about it is 0.506 mm.

Scored against the brain mask it does not halve the structures — the left fraction runs from
41% (midbrain) to 57% (hippocampus), 53.3% over the whole mask. That looks like a plane
error, so three independent refits were tried. Each optimises a different, defensible
objective; all three land somewhere else, and **none of them reduces the imbalance**:

    objective                       moves the plane      whole-mask left      verdict
    mirror-overlap of the mask      0.44 deg, +0.006 mm   53.3% -> 54.1%      worse
    per-structure volume balance    1.24 deg, +0.338 mm   53.3% -> 51.7%      max dev 7.2 -> 7.1 pt
    label-mirror agreement          2.01 deg, +0.691 mm   53.3% -> 54.3%      worse

The mirror-overlap objective is the one to distrust: for a solid blob nearly every *interior*
voxel maps back inside whatever the plane does, so it sits at 0.90 and carries almost no
signal. The label-mirror version fixes that — a mirrored hippocampus voxel landing in cortex
now counts as a miss — and it is the informative one: **its best achievable agreement is only
0.81.** A mirror-symmetric segmentation would score well above 0.9.

So the asymmetry is in the phantom, not in our plane. NeuroRat V4.0 is one animal's
segmentation and is genuinely lopsided; the cerebral cortex holds 401k voxels left of the
plane against 344k right of it, and the interhemispheric fissure is plainly visible as a
density minimum at s = 0, which is where the plane already is. Also telling: every structure
has |centroid_right| > |centroid_left| by a similar 0.25-0.55 mm, an offset error would move
that gap, and moving the plane by 0.3-0.7 mm does not.

**Consequence to state wherever a lateralised rat target is reported**: "left hippocampus"
carries roughly 7% of the contralateral structure. That is a property of the model. Do not
paper over it by tuning the plane — three objectives already say there is nothing to tune.
"""
import os
import sys
import json

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from tip import config as C            # noqa: E402
from tip.leadfield import LeadField    # noqa: E402


def main():
    if C.MODEL_NAME != "rat":
        sys.exit("run with TIP_MODEL=rat")
    lf = LeadField()
    P = lf.coords()
    bl = np.load(C.inputs(C.BLABEL_FILE))
    names = {v: k for k, v in json.load(open(C.inputs("labels_rat.json"))).items()}
    n0 = np.asarray(C.MODEL.midline_normal, float); n0 /= np.linalg.norm(n0)
    d0 = float(C.MODEL.midline_offset)

    #  label volume, for the mirror test that actually has signal
    LV = np.zeros((len(lf.cx), len(lf.cy), len(lf.cz)), np.uint8)
    LV[lf.bmask[:, 0], lf.bmask[:, 1], lf.bmask[:, 2]] = bl

    def label_at(Q):
        i = np.clip(np.searchsorted(lf.cx, Q[:, 0]) - 1, 0, len(lf.cx) - 1)
        j = np.clip(np.searchsorted(lf.cy, Q[:, 1]) - 1, 0, len(lf.cy) - 1)
        k = np.clip(np.searchsorted(lf.cz, Q[:, 2]) - 1, 0, len(lf.cz) - 1)
        return LV[i, j, k]

    rng = np.random.default_rng(0)
    sel = rng.choice(len(P), min(250000, len(P)), replace=False)
    S, SL = P[sel], bl[sel]

    a = np.array([1.0, 0, 0])
    u = np.cross(n0, a); u /= np.linalg.norm(u)
    v = np.cross(n0, u)

    def unpack(x):
        n = n0 + x[0] * u + x[1] * v
        return n / np.linalg.norm(n), d0 + x[2]

    def neg_agree(x):
        n, d = unpack(x)
        Q = S - 2.0 * ((S @ n) - d)[:, None] * n
        return -float((label_at(Q) == SL).mean())

    print("label-mirror agreement at the plane in models.py: %.4f" % (-neg_agree(np.zeros(3))))
    best = None
    for x0 in ([0, 0, 0], [0, 0, .3], [0, 0, -.3], [.02, 0, 0], [-.02, 0, 0],
               [0, .02, 0], [0, -.02, 0]):
        r = minimize(neg_agree, np.array(x0, float), method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-7, maxiter=2000, maxfev=2000))
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    n, d = unpack(best[1])
    print("best achievable anywhere:                        %.4f  "
          "(rot %.2f deg, offset %+.3f mm)"
          % (-best[0], np.degrees(np.arccos(np.clip(float(n0 @ n), -1, 1))), d - d0))
    print("★ a mirror-symmetric segmentation would exceed 0.90. This one cannot. The "
          "asymmetry is the phantom's, not the plane's.\n")

    s0, s1 = P @ n0, P @ n
    print("%-20s %10s %10s %12s %12s" % ("structure", "left% now", "left% best",
                                         "vox left", "vox right"))
    for lab in sorted(names):
        g = bl == lab
        if not g.any():
            continue
        L = s0[g] > d0
        print("%-20s %9.1f%% %9.1f%% %12d %12d"
              % (names[lab], 100 * L.mean(), 100 * (s1[g] > d).mean(),
                 int(L.sum()), int((~L).sum())))
    print("%-20s %9.1f%% %9.1f%%" % ("WHOLE MASK", 100 * (s0 > d0).mean(),
                                     100 * (s1 > d).mean()))
    print("\nNothing was written. See the module docstring before changing models.py.")


if __name__ == "__main__":
    main()
