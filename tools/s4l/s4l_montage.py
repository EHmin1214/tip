# -*- coding: utf-8 -*-
"""s4l_montage.py — export a TIP montage as a Sim4Life project

Why this exists
---------------
E inside the brain is already **exact** from leadfield superposition (superposition check
6.9e-15); re-solving gives the same numbers. The value of solving again in Sim4Life is
elsewhere:

- **The whole head.** Our leadfield stores only the 1,907,678 brain voxels (18% of the grid).
  Scalp, skull, CSF, eye and neck are absent, so **safety numbers such as scalp current
  density are simply not available.**
- **J, potential and loss density** — we only store E.
- **Sim4Life's own TI post-processing** — an independent implementation, so it cross-checks
  our metrics. (Convention mismatches have produced metric bugs here three times.)

Structure
---------
TI has two carriers. The problem is quasi-static, so the channels are **solved separately**
and the envelope is combined in post-processing. Hence two simulations, `ch1` and `ch2`.
The rebuilt project is copied, so materials, grid and voxel settings are inherited exactly
(grid 185 x 254 x 228 = 10.7 MCells).

**Current convention**: the simulation is solved at 1 V Dirichlet. The real current in mA is
applied afterwards as `E · (i_k / I_inj)` — the same convention as the leadfield. The intended
currents are recorded in a sidecar JSON.

Usage (from the Sim4Life Python):
    python s4l_montage.py export <out.smash> A B C D --ratio 1.0 --itotal 2.0
"""
import json
import os
import shutil
import sys

os.environ.setdefault("S4L_API_AUTO_INIT", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
from tip.config import inputs as IN                # input-file resolver
OUTPUTS = os.path.join(REPO, "outputs")

import rebuild_solve_batch as R  # noqa: E402


def _grp(xm):
    m = xm.GetActiveModel()
    for e in m.RootGroup.Entities:
        if e.Name == "Cloned Templates":
            return e
    raise KeyError("no 'Cloned Templates' group")


def _elec(xm, name):
    for e in _grp(xm).Entities:
        if e.Name == f"{name}_ElectrodeTemplate":
            return e
    raise KeyError(f"{name}_ElectrodeTemplate not found")


def set_pair(sim, xm, plus, minus):
    """Move `src` (1 V) to the `plus` electrode and `ref` (0 V) to `minus`.

    `rebuild_solve_batch.set_src` hard-codes Cz as the reference, so it cannot drive a montage.
    This follows the same component-level convention — `sim.Add` / `sim.Remove` are blocked in
    Sim4Life 9.6.
    ⚠ `RemoveSettingsFromComponent` returns False even on success.
      **Attach before detaching**, otherwise there is a moment with nothing assigned.
    """
    import XSimulator                                     # noqa: N813

    def _nm(c):
        me = c.ModelEntity
        return None if me is None else str(me.Name)

    for role, who in (("src", plus), ("ref", minus)):
        bs = [c for c in sim.AllSettings
              if type(c).__name__ == "BoundarySettings" and c.Name == role]
        if not bs:
            raise RuntimeError(f"no boundary setting named '{role}'")
        bs = bs[0]
        tgt = f"{who}_ElectrodeTemplate"
        old = [c for c in (bs.raw.AssignedComponent(i)
                           for i in range(bs.raw.SizeAssignedComponents()))
               if _nm(c) != tgt]
        if not sim.raw.AcquireComponent(_elec(xm, who)).AssignSettings(bs.raw):
            raise RuntimeError(f"failed to assign {who} to {role}")
        for o in old:
            XSimulator.RemoveSettingsFromComponent(bs.raw, o)
        got = [_nm(bs.raw.AssignedComponent(i))
               for i in range(bs.raw.SizeAssignedComponents())]
        if got != [tgt]:
            raise RuntimeError(f"{role} ended up as {got}")
    return True


def export(out_smash, ch1, ch2, ratio=1.0, itotal=2.0, verbose=True):
    """Create a new project containing the montage. **Does not solve.**

    ch1 · ch2 : (anode, cathode) electrode-name pairs
    returns   : {"smash": path, "meta": path, "sims": ["ch1", "ch2"]}
    """
    import s4l_v1 as s4l
    import s4l_v1.document as doc
    import XCoreModeling as xm
    from tip.optimize.classic import channel_currents

    i1, i2 = channel_currents(float(ratio), budget=float(itotal))
    os.makedirs(os.path.dirname(out_smash), exist_ok=True)

    #  ★Copy the rebuilt project **at the file level first**. Using doc.SaveAs alone leaves
    #    `_Results` behind, which forces the voxels to be rebuilt (87 s).
    if os.path.exists(out_smash):
        os.remove(out_smash)
    shutil.copy2(R.SMASH, out_smash)
    src_res = R.SMASH + "_Results"
    dst_res = out_smash + "_Results"
    if os.path.isdir(src_res):
        shutil.rmtree(dst_res, ignore_errors=True)
        shutil.copytree(src_res, dst_res)

    doc.Open(out_smash)
    sims = list(doc.AllSimulations)
    base = sims[0]

    #  Channel 1 reuses the original simulation; channel 2 is a clone.
    base.Name = "ch1"
    set_pair(base, xm, ch1[0], ch1[1])
    two = base.Clone()
    two.Name = "ch2"
    if two not in list(doc.AllSimulations):
        doc.AllSimulations.Add(two)
    set_pair(two, xm, ch2[0], ch2[1])

    doc.Save()

    meta = {
        "montage": {"ch1": list(ch1), "ch2": list(ch2), "ratio": float(ratio)},
        "currents_mA": {"ch1": float(i1), "ch2": float(i2), "itotal": float(itotal)},
        "convention": ("Solved at 1 V Dirichlet. Real field = E_1V * (i_k / I_inj), with "
                       "I_inj = integral of sigma|E|^2 dV. Same convention as the leadfield."),
        "grid": "185 x 254 x 228 = 10.714 MCells (inherited from the rebuilt project)",
        "base_project": R.SMASH,
    }
    mpath = out_smash.replace(".smash", "_montage.json")
    json.dump(meta, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if verbose:
        print(f"[s4l_montage] project {out_smash}")
        print(f"  ch1 {ch1[0]}(+) → {ch1[1]}(−)  {i1:.4f} mA")
        print(f"  ch2 {ch2[0]}(+) → {ch2[1]}(−)  {i2:.4f} mA")
        print(f"  simulations: {[s.Name for s in doc.AllSimulations]}")
        print(f"  metadata {mpath}")
    return {"smash": out_smash, "meta": mpath,
            "sims": [s.Name for s in doc.AllSimulations]}


def extract_full(out_h5):
    """Extract E over the **full grid**. `R.extract` returns brain voxels only, and the whole
    head is the point here, so this extends it.

    Returns: E (NX,NY,NZ,3) float32 in V/m at 1 V drive, sigma (NX,NY,NZ) float32,
    the axes in metres, and I in amperes.
    """
    import h5py
    import numpy as np
    with h5py.File(out_h5, "r") as f:
        lut, vox, ax = R._sigma_and_axes(f)
        fg = f["FieldGroups"]
        key = [x for x in fg if "EM E(x,y,z,f0)" in fg[x]["AllFields"]][0]
        sn = fg[key]["AllFields"]["EM E(x,y,z,f0)"]["_Object"]["Snapshots"]["0"]
        e0 = sn["comp0"][..., 0].astype(np.float32)
        e1 = sn["comp1"][..., 0].astype(np.float32)
        e2 = sn["comp2"][..., 0].astype(np.float32)
    # Staggered-grid edges -> cell centres, averaging the four parallel edges. Same convention
    # as `R.extract`.
    Ex = .25 * (e0[:, :-1, :-1] + e0[:, 1:, :-1] + e0[:, :-1, 1:] + e0[:, 1:, 1:]); del e0
    Ey = .25 * (e1[:-1, :, :-1] + e1[1:, :, :-1] + e1[:-1, :, 1:] + e1[1:, :, 1:]); del e1
    Ez = .25 * (e2[:-1, :-1, :] + e2[1:, :-1, :] + e2[:-1, 1:, :] + e2[1:, 1:, :]); del e2
    sig = lut[vox].astype(np.float32)
    d = [np.diff(a) for a in ax]
    E2 = (Ex.astype(np.float64) ** 2 + Ey.astype(np.float64) ** 2
          + Ez.astype(np.float64) ** 2)
    E2 = np.where(np.isfinite(E2), E2, 0.0)
    I = float(np.sum(lut[vox] * E2 *
                     (d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :])))
    del E2
    return np.stack([Ex, Ey, Ez], -1), sig, ax, I


def solve_project(smash, out_dir, verbose=True):
    """Solve the project's simulations (ch1, ch2) in turn and save E over the **full grid**.

    ⚠ `CreateVoxels` deletes `_Results`, so extraction must happen **before** the next channel
      is solved. Hence the sequential loop that extracts immediately into `out_dir`.
    """
    import subprocess
    import time
    import numpy as np
    import s4l_v1.document as doc

    os.makedirs(out_dir, exist_ok=True)
    doc.Open(smash)
    inj = {}
    for sim in list(doc.AllSimulations):
        t0 = time.time()
        sim.CreateVoxels()
        sim.WriteInputFile()
        inp = sim.InputFilename
        r = subprocess.run([R.ISOLVE, inp], cwd=os.path.dirname(inp),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{sim.Name} iSolve failed with {r.returncode}: {r.stdout[-400:]}")
        E, sig, ax, I = extract_full(inp.replace("_Input.h5", "_Output.h5"))
        np.save(os.path.join(out_dir, f"{sim.Name}_E1V.npy"), E)
        #  ★Also save the brain-voxel subset (22.9 MB per channel).
        #    The metrics (M1/M2/M3) need only this, and the full grid (128 MB) is deleted after
        #    the analysis. This file is what makes re-deriving metrics for a different target
        #    possible **without re-solving**.
        bm = np.load(IN("bmask1010.npy")).astype(np.int64)
        np.save(os.path.join(out_dir, f"{sim.Name}_Ebrain.npy"),
                E[bm[:, 0], bm[:, 1], bm[:, 2]].astype(np.float32))
        if not os.path.exists(os.path.join(out_dir, "sigma.npy")):
            np.save(os.path.join(out_dir, "sigma.npy"), sig)
            np.savez(os.path.join(out_dir, "axes.npz"),
                     **{f"axis_{c}": a for c, a in zip("xyz", ax)})
        inj[sim.Name] = I
        if verbose:
            print(f"[s4l_montage] {sim.Name} done · {time.time()-t0:.0f}s · "
                  f"I={I*1e3:.4f} mA · E {E.shape}")
        del E, sig
    json.dump(inj, open(os.path.join(out_dir, "inj.json"), "w"), indent=1)
    return inj


def cleanup_shadows(smash):
    """Delete Sim4Life autosave shadow files — about 250 MB accumulates per montage."""
    d, base = os.path.dirname(smash), os.path.basename(smash)
    n = 0
    for f in os.listdir(d):
        if f != base and f.startswith((base + ".", "." + base + ".")):
            try:
                os.remove(os.path.join(d, f)); n += 1
            except OSError:
                pass
    return n


def main(argv):
    if not argv:
        print(__doc__); return 1
    if argv[0] == "export":
        out = argv[1]
        a, b, c, d = argv[2:6]
        ratio = float(argv[argv.index("--ratio") + 1]) if "--ratio" in argv else 1.0
        itot = float(argv[argv.index("--itotal") + 1]) if "--itotal" in argv else 2.0
        export(out, (a, b), (c, d), ratio, itot)
        if "--solve" in argv:
            odir = argv[argv.index("--solve") + 1]
            solve_project(out, odir)
        print(f"cleaned up {cleanup_shadows(out)} shadow file(s)")
        print("=== END · export complete")    # orch completion marker
    elif argv[0] == "solve":
        solve_project(argv[1], argv[2])
        print(f"cleaned up {cleanup_shadows(argv[1])} shadow file(s)")
        print("=== END · solve complete")
    else:
        print(__doc__); return 1
    os._exit(0)                            # a clean shutdown takes minutes in DLL detach


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
