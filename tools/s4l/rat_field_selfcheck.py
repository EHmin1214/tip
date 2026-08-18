# -*- coding: utf-8 -*-
"""rat_field_selfcheck.py — is the field we decode the field the solver wrote?
=============================================================================
The rat has no reference dataset, so every check has to be internal. This is the sharpest one
available without burning another solve: the solver stores **both** the staggered edge E and
the nodal potential, and they must satisfy

    E_x(edge i) = -(phi[i+1] - phi[i]) / (x[i+1] - x[i])

component by component. Agreement means our reading of `comp0/1/2` — which axis each one
runs along, which nodes an edge connects, the metres-vs-millimetres of the axes — is right.
That is exactly the part that is easy to get wrong and impossible to notice later: a swapped
component or an off-by-one in the staggering produces a perfectly plausible leadfield.

It also checks the 4-edge cell-centring used by `rat_extract`, by comparing the cell-centred
field against a central difference of the potential at the same cell centres.

    python tools/s4l/rat_field_selfcheck.py [port_output.h5]
"""
import os
import sys
import glob

import h5py
import numpy as np

DESK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DESK = os.path.dirname(DESK)


def main():
    if len(sys.argv) > 1:
        p = sys.argv[1]
    else:
        d = glob.glob(os.path.join(DESK, "s4l_projects", "rat_lf.smash_Results",
                                   "*_emlf_ports"))[0]
        outs = sorted(glob.glob(os.path.join(d, "*_Output.h5")),
                      key=os.path.getsize, reverse=True)
        p = outs[0]
    print("file:", os.path.basename(p))

    with h5py.File(p, "r") as f:
        mg = f["Meshes"][[m for m in f["Meshes"] if "voxels" in f["Meshes"][m]][0]]
        ax = [np.asarray(mg["axis_" + c], float) for c in "xyz"]      # metres, nodes
        fg = f["FieldGroups"]
        key = [x for x in fg if "EM E(x,y,z,f0)" in fg[x]["AllFields"]][0]
        E = fg[key]["AllFields"]["EM E(x,y,z,f0)"]["_Object"]["Snapshots"]["0"]
        e = [E["comp%d" % i][..., 0].astype(np.float64) for i in range(3)]
        pk = [x for x in fg if "EM Potential(x,y,z,f0)" in fg[x]["AllFields"]][0]
        phi = (fg[pk]["AllFields"]["EM Potential(x,y,z,f0)"]["_Object"]
               ["Snapshots"]["0"]["comp0"][..., 0].astype(np.float64))

    print("nodes %s   phi %s   comps %s"
          % ([len(a) for a in ax], phi.shape, [c.shape for c in e]))

    ok = True
    for axis in range(3):
        d = np.diff(ax[axis])
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1); sl1[axis] = slice(1, None)
        shape = [1, 1, 1]; shape[axis] = -1
        pred = -(phi[tuple(sl1)] - phi[tuple(sl0)]) / d.reshape(shape)
        if pred.shape != e[axis].shape:
            print("  comp%d SHAPE MISMATCH pred %s vs stored %s"
                  % (axis, pred.shape, e[axis].shape))
            ok = False
            continue
        #  10.0 M of the 21.5 M cells are background (sigma = 0) and carry no degree of
        #  freedom, so their nodal potential is NaN. Compare only where both sides exist.
        m = np.isfinite(pred) & np.isfinite(e[axis])
        den = np.abs(e[axis][m]).max()
        rel = float(np.abs(pred[m] - e[axis][m]).max() / den)
        med = float(np.median(np.abs(pred[m] - e[axis][m])) / np.median(np.abs(e[axis][m])))
        print("  comp%d  %.1f%% of edges solved   max rel err %.3e   median rel err %.3e   %s"
              % (axis, 100 * m.mean(), rel, med, "ok" if rel < 1e-4 else "LOOK AT THIS"))
        ok &= rel < 1e-4

    #  the cell-centring rat_extract uses, against a central difference at the same centres
    Ex = .25 * (e[0][:, :-1, :-1] + e[0][:, 1:, :-1] + e[0][:, :-1, 1:] + e[0][:, 1:, 1:])
    c = [(a[:-1] + a[1:]) * 0.5 for a in ax]
    #  potential averaged onto cell centres, then differenced along x
    pc = .125 * (phi[:-1, :-1, :-1] + phi[1:, :-1, :-1] + phi[:-1, 1:, :-1] + phi[:-1, :-1, 1:]
                 + phi[1:, 1:, :-1] + phi[1:, :-1, 1:] + phi[:-1, 1:, 1:] + phi[1:, 1:, 1:])
    g = -np.gradient(pc, c[0], axis=0)
    fin = np.isfinite(g) & np.isfinite(Ex)
    m = fin & (np.abs(Ex) > 0.05 * np.abs(np.where(fin, Ex, 0)).max())   # where there is field
    r = float(np.median(np.abs(g[m] - Ex[m]) / np.abs(Ex[m])))
    print("  cell-centred Ex vs central difference of phi: median rel err %.3e  %s"
          % (r, "ok (a per-cent-level gap is expected between the two schemes)"
             if r < 0.05 else "LOOK AT THIS"))
    print("VERDICT:", "decoding is consistent with the solver's own potential" if ok
          else "mismatch - do not trust the extracted leadfield")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
