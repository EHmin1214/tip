# -*- coding: utf-8 -*-
"""rat_lf_float.py — re-solve the rat per-electrode leadfield under the MONTAGE convention.
============================================================================================
The rat leadfield was solved in EM LF **port mode**: basis k = electrode k at 1 V with all 36
others held at **0 V**. A real montage drives one pair and leaves the rest **floating**. The
36 shorted PEC pins are a low-impedance path across the scalp that no experiment has, and it
is not a small effect — measured on `O1-C5 | PO3-AF3` (2026-08-20):

    injected at 1 V   port basis 0.261-0.304 mA/V (3288-3835 ohm)
                      montage    0.129-0.130 mA/V (7690-7769 ohm)   => 2.19x more current
    consequence       M1 leadfield 0.6807 vs Sim4Life 1.2494        => x1.835

This script solves each electrode the way the montage is solved — electrode k at 1 V, the
reference at 0 V, **every other electrode floating** — so superposition reproduces a montage
exactly instead of approximately.

★Why superposition still holds. Each basis is normalised to 1 mA. In `LF[A] - LF[B]`:
  · a floating electrode carries zero net current in both bases, so zero in the difference,
    and stays an equipotential — the floating condition is satisfied;
  · the reference is at 0 V in both and carries -1 mA in both, so **0 mA** in the difference —
    also consistent with floating.
The combination therefore satisfies every condition of the montage problem, and by uniqueness
it *is* its solution. This is the same argument that makes the human leadfield exact.

Usage (needs Sim4Life + a QS_SOLVER seat; about 274 s per electrode):

    TIP_MODEL=rat .venv-s4l/Scripts/python.exe tools/s4l/rat_lf_float.py OUTDIR AF3 C5 O1 PO3

`OUTDIR` is relative to `inputs/leadfield/` unless absolute. Existing `.npy` files are kept
unless `--force`; `inj_current.json` is merged, never overwritten wholesale.

⚠ Works on a **copy** of the leadfield project. Touching `rat_lf.smash` would change its
fingerprint and silently invalidate every cached montage analysis.
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
os.environ.setdefault("TIP_MODEL", "rat")

import numpy as np                                    # noqa: E402
from tip import config as C                           # noqa: E402
import rebuild_solve_batch as R                       # noqa: E402
import s4l_montage as M                               # noqa: E402


def work_project(base, tag="rat_lf_float"):
    """A private copy of the leadfield project. Never solve in the original."""
    dst = os.path.join(os.path.dirname(base), tag + ".smash")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(base):
        print("[rat_lf_float] copying %s -> %s (%.1f GB)"
              % (os.path.basename(base), os.path.basename(dst),
                 os.path.getsize(base) / 1e9), flush=True)
        for p in (dst, dst + "_Results"):
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)
        shutil.copy2(base, dst)
    return dst


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("elec", nargs="+")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ref", default=None, help="reference electrode (default: the model's)")
    a = ap.parse_args(argv)

    import s4l_v1.document as doc
    import XCoreModeling as xm

    out_dir = a.out if os.path.isabs(a.out) else \
        os.path.join(REPO, "inputs", "leadfield", a.out)
    os.makedirs(out_dir, exist_ok=True)
    ref = a.ref or C.REF_ELEC

    #  The same brain-voxel index set every other rat product uses. Taking it from the
    #  stored geometry rather than re-deriving it keeps the new leadfield aligned with the
    #  masks and targets that already exist.
    bm = np.load(C.inputs(C.BMASK_FILE)).astype(np.int64)
    bi, bj, bk = bm[:, 0], bm[:, 1], bm[:, 2]

    smash = work_project(M.base_smash())
    doc.Open(smash)
    sims = list(doc.AllSimulations)
    if not sims:
        sys.exit("no simulation in " + smash)
    sim = sims[0]

    inj_path = os.path.join(out_dir, "inj_current.json")
    inj = json.load(open(inj_path)) if os.path.exists(inj_path) else {}

    for n, el in enumerate(a.elec, 1):
        dst = os.path.join(out_dir, el + ".npy")
        if os.path.exists(dst) and not a.force:
            print("[rat_lf_float] = %-5s reused" % el, flush=True)
            continue
        if el == ref:
            print("[rat_lf_float] ! %-5s is the reference — skipped" % el, flush=True)
            continue
        t0 = time.time()
        M.set_pair(sim, xm, el, ref)          # el at 1 V, ref at 0 V, the other 35 floating
        sim.CreateVoxels()
        sim.WriteInputFile()
        inp = sim.InputFilename
        r = subprocess.run([R.ISOLVE, inp], cwd=os.path.dirname(inp),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("%s iSolve failed with %d: %s"
                               % (el, r.returncode, r.stdout[-400:]))
        E, _sig, _ax, I = M.extract_full(inp.replace("_Input.h5", "_Output.h5"))
        #  Same normalisation as `rat_extract.py`: store the field per 1 mA.
        Mn = (E[bi, bj, bk].astype(np.float64) * 1e-3 / I).astype(np.float32)
        np.save(dst, Mn)
        inj[el] = I
        json.dump(inj, open(inj_path, "w"), indent=1)
        print("[rat_lf_float] + %-5s %3d/%d · %.0fs · I = %.4f mA per V · "
              "median |E| = %.4f V/m per mA"
              % (el, n, len(a.elec), time.time() - t0, I * 1e3,
                 float(np.median(np.linalg.norm(Mn, axis=1)))), flush=True)
        del E, Mn

    print("[rat_lf_float] === END · %d electrode(s) in %s" % (len(a.elec), out_dir))
    os._exit(0)                    # a clean DLL detach takes minutes


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
