# -*- coding: utf-8 -*-
"""identify_electrode.py — identify which electrode an unlabelled leadfield column belongs to

Sometimes the batch parent dies and only `iSolve` survives to finish (an orphaned solve). The
file then carries no record of which electrode was solved, so the extracted field is matched by
direction against **a set whose names are known**.
TP8 was identified this way during the rebuild — the first and second place (an adjacent
electrode) separate clearly.

    python identify_electrode.py <unknown.npy> [candidate-set dir] [candidate names...]

The default candidate set is `tiplitepos_lf` — our own solve at the tip.lite electrode
coordinates. Solved at the same coordinates, the same electrode reaches cos ~0.99.
"""
import os
import sys

import numpy as np

#  Working directory for large intermediates. Override with TIP_SCRATCH.
SP = os.environ.get("TIP_SCRATCH") or os.path.join(REPO, "outputs", "scratch")


def cos_median(A, B):
    """Median per-voxel directional cosine. Insensitive to magnitude differences, which is what
    a model difference mostly is."""
    na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    return float(np.median((A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])))


def main(argv):
    if not argv:
        print(__doc__); return 2
    unknown = argv[0]
    ref_dir = os.path.join(SP, argv[1] if len(argv) > 1 else "tiplitepos_lf")
    cands = argv[2:] or [os.path.splitext(f)[0] for f in os.listdir(ref_dir)
                         if f.endswith(".npy")]
    U = np.load(unknown).astype(np.float64)
    print(f"unknown: {os.path.basename(unknown)}  {U.shape}  "
          f"median |E| {np.median(np.linalg.norm(U, axis=1)):.4f}")
    print(f"candidate set: {ref_dir}  ({len(cands)})\n")
    out = []
    for c in cands:
        p = os.path.join(ref_dir, f"{c}.npy")
        if not os.path.exists(p):
            continue
        R = np.load(p).astype(np.float64)
        if R.shape != U.shape:
            continue
        out.append((cos_median(U, R), c,
                    float(np.median(np.linalg.norm(U, axis=1))
                          / max(np.median(np.linalg.norm(R, axis=1)), 1e-30))))
    out.sort(reverse=True)
    print(f"{'#':>3} {'electrode':10}{'median cos':>12}{'mag ratio':>11}")
    for i, (c, nm, rr) in enumerate(out[:8], 1):
        print(f"{i:>3} {nm:8}{c:>10.4f}{rr:>9.3f}")
    if len(out) >= 2:
        gap = out[0][0] - out[1][0]
        verdict = "clear" if (out[0][0] > 0.95 and gap > 0.02) else "ambiguous - re-solve advised"
        print(f"\nverdict: **{out[0][1]}**  (cos {out[0][0]:.4f}, gap to 2nd {gap:.4f}) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
