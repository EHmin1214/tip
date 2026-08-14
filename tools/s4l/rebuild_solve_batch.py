import sys
# -*- coding: utf-8 -*-
"""rebuild_solve_batch.py — re-solve several electrodes in the rebuilt project to form a leadfield
==========================================================================================
The purpose is a **control**. The M2 ratio against tip.lite computed from the original
`leadfieldF` (0.793 right thalamus, 0.878 left thalamus, over 15 montages without the lower
ring) — does it **reproduce in a re-solve of the same physical setup**?

  · If it reproduces → the original leadfield is sound and the difference lies in the model
                       (anisotropy, conductivity, grid).
  · If it does not   → **the original leadfield itself** is the problem. The rebuild matches
                       the original at cos 0.9916, but component correlations of 0.905-0.959
                       mean the spatial structure differs subtly.

This runs inside a Sim4Life worker via `exec` (`s4l_run_python`) and needs the Sim4Life API.
About 2 minutes per electrode (writing the input file plus iSolve).

Produces `<OUT>/<electrode>.npy` — (N,3) float32 in bmask1010 order, **normalised to 1 mA**
      (the same convention as the original leadfieldF: E * 1e-3 / I_inj with
       I_inj = integral of sigma|E|^2 dV)
"""
import os
import subprocess
import time

import h5py
import numpy as np


#  Sim4Life solver. Override with TIP_ISOLVE if installed elsewhere.
ISOLVE = os.environ.get("TIP_ISOLVE") or \
    r"C:\Program Files\Sim4Life_9.6\Solvers\iSolve.exe"
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")
from tip.config import inputs as IN, LEADFIELD_DIR as _LFDIR   # input-file resolver

DD = INPUTS
#  Working directory for large intermediates. Override with TIP_SCRATCH.
SP = os.environ.get("TIP_SCRATCH") or os.path.join(REPO, "outputs", "scratch")
#  ★2026-08-12: the project file moved out of the session scratchpad into the repository area.
#  The scratchpad can be wiped, and without this file there is no way to extend the rebuilt set.
#  (The original and six Sim4Life autosave shadows remain in SP above.)
PROJ = os.environ.get("TIP_S4L_PROJECTS") or \
    os.path.join(os.path.dirname(REPO), "s4l_projects")
#  Overridable via environment variables, so the same driver can solve a different electrode
#  layout (tip.lite coordinates, say) into a separate folder. Unset means the existing
#  rebuilt set.
OUT = os.environ.get("TIP_REBUILD_OUT") or os.path.join(SP, "rebuild_lf")
SMASH = os.environ.get("TIP_REBUILD_SMASH") or os.path.join(PROJ, "mida1010_rebuild.smash")

_BM = None


def _bmask():
    global _BM
    if _BM is None:
        _BM = np.load(IN("bmask1010.npy")).astype(np.int64)
    return _BM


def _sigma_and_axes(f):
    """Extract the per-voxel conductivity lookup and the grid axes (in metres) from the
    output h5."""
    mats = f["AllMaterialMaps"][list(f["AllMaterialMaps"])[0]]
    sig = {}
    for k in mats:
        for pk in mats[k]:
            if b"ElectricConductivity" in mats[k][pk]["_ClassInfo"].attrs.get("_TypeName", b""):
                sig[k.replace("-", "").lower()] = float(
                    mats[k][pk]["_Object"].attrs["uniform_scalar"])
                break
    mesh = [m for m in f["Meshes"] if "voxels" in f["Meshes"][m]][0]
    mg = f["Meshes"][mesh]
    vox = mg["voxels"][...]
    lut = np.zeros(int(vox.max()) + 1)
    for row, idx in zip(mg["id_map"][...], mg["index_map"][...]):
        u = row.tobytes().hex()
        if u in sig:
            lut[idx] = sig[u]
    ax = [np.asarray(mg[f"axis_{c}"], float) for c in "xyz"]
    return lut, vox, ax


def extract(out_h5):
    """Return the brain-voxel E in V/m and the total injected current I in amperes."""
    bm = _bmask(); i, j, k = bm[:, 0], bm[:, 1], bm[:, 2]
    with h5py.File(out_h5, "r") as f:
        lut, vox, ax = _sigma_and_axes(f)
        fg = f["FieldGroups"]
        key = [x for x in fg if "EM E(x,y,z,f0)" in fg[x]["AllFields"]][0]
        sn = fg[key]["AllFields"]["EM E(x,y,z,f0)"]["_Object"]["Snapshots"]["0"]
        e0 = sn["comp0"][..., 0].astype(np.float32)
        e1 = sn["comp1"][..., 0].astype(np.float32)
        e2 = sn["comp2"][..., 0].astype(np.float32)
    Ex = .25 * (e0[:, :-1, :-1] + e0[:, 1:, :-1] + e0[:, :-1, 1:] + e0[:, 1:, 1:]); del e0
    Ey = .25 * (e1[:-1, :, :-1] + e1[1:, :, :-1] + e1[:-1, :, 1:] + e1[1:, :, 1:]); del e1
    Ez = .25 * (e2[:-1, :-1, :] + e2[1:, :-1, :] + e2[:-1, 1:, :] + e2[1:, 1:, :]); del e2
    E = np.stack([Ex[i, j, k], Ey[i, j, k], Ez[i, j, k]], 1).astype(np.float64)
    E2 = Ex.astype(np.float64) ** 2 + Ey.astype(np.float64) ** 2 + Ez.astype(np.float64) ** 2
    del Ex, Ey, Ez
    E2 = np.where(np.isfinite(E2), E2, 0.0)
    d = [np.diff(a) for a in ax]
    I = float(np.sum(lut[vox] * E2 *
                     (d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :])))
    return E, I


def set_src(sim, xm, name):
    """Move `src` (1 V Dirichlet) onto the single electrode `name`.

    Dead ends in Sim4Life 9.6: `sim.AssignEntities` was removed; `sim.Add(src, ent)` fails
    because the setting is already registered, so `AddSettings` returns False and it dies with
    "incompatible"; `sim.Remove(src)` does not work either. What does work is the
    **component-level** API.

        c = sim.raw.AcquireComponent(entity)   # creates it if absent, reuses it if present
        c.AssignSettings(src.raw)              # attach
        XSimulator.RemoveSettingsFromComponent(src.raw, old_component)   # detach

    ⚠ `RemoveSettingsFromComponent` returns False even on success — verify with
      `SizeAssignedComponents`, not the return value. Attach before detaching so there is
      never a moment with nothing assigned.
    """
    import XSimulator                                     # noqa: N813
    m = xm.GetActiveModel()
    grp = [e for e in m.RootGroup.Entities if e.Name == "Cloned Templates"][0]
    ent = [e for e in grp.Entities if e.Name == f"{name}_ElectrodeTemplate"]
    if not ent:
        raise KeyError(f"{name}_ElectrodeTemplate not found")
    src = [c for c in sim.AllSettings
           if type(c).__name__ == "BoundarySettings" and c.Name == "src"][0]
    tgt = f"{name}_ElectrodeTemplate"
    # ⚠ Re-assigning an electrode that is already attached would make the detach step remove
    #    what was just attached, so the same entity is excluded from `old`.
    # ⚠ In a project where an electrode body was deleted (a copy with moved coordinates, say)
    #    a **husk component** remains whose `ModelEntity` is None — touching `.Name` on it
    #    crashes. A None entity is never the target, so it goes into `old` and is detached too.
    def _nm(c):
        me = c.ModelEntity
        return None if me is None else str(me.Name)
    old = [c for c in (src.raw.AssignedComponent(i)
                       for i in range(src.raw.SizeAssignedComponents()))
           if _nm(c) != tgt]
    if not sim.raw.AcquireComponent(ent[0]).AssignSettings(src.raw):
        raise RuntimeError(f"failed to assign src to {name}")
    for o in old:
        XSimulator.RemoveSettingsFromComponent(src.raw, o)
    got = [_nm(src.raw.AssignedComponent(i))
           for i in range(src.raw.SizeAssignedComponents())]
    if got != [tgt]:
        raise RuntimeError(f"src ended up as {got}")

    # The reference electrode (ref, Cz) can also be a husk — reattach it to a fresh body.
    ref = [c for c in sim.AllSettings
           if type(c).__name__ == "BoundarySettings" and c.Name == "ref"]
    if ref:
        ref = ref[0]
        cur = [_nm(ref.raw.AssignedComponent(i))
               for i in range(ref.raw.SizeAssignedComponents())]
        if cur != ["Cz_ElectrodeTemplate"]:
            cz = [e for e in grp.Entities if e.Name == "Cz_ElectrodeTemplate"]
            if not cz:
                raise RuntimeError("Cz_ElectrodeTemplate not found - cannot attach the reference")
            dead = [ref.raw.AssignedComponent(i)
                    for i in range(ref.raw.SizeAssignedComponents())]
            if not sim.raw.AcquireComponent(cz[0]).AssignSettings(ref.raw):
                raise RuntimeError("failed to assign Cz to ref")
            for o in dead:
                XSimulator.RemoveSettingsFromComponent(ref.raw, o)
    return got


def solve_one(sim, xm, name, force=False):
    """Drive electrode `name` as src, solve, and save the 1 mA-normalised leadfield."""
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, f"{name}.npy")
    if os.path.exists(dst) and not force:
        return dst, None, "reused"
    set_src(sim, xm, name)
    t0 = time.time()
    # ★WriteInputFile fails unless CreateVoxels is called first. (The message
    #   `Failure type: network` is just the default exception text, not the real cause.)
    #   ⚠ CreateVoxels deletes the `_Results` folder — move any earlier output out first.
    sim.CreateVoxels()
    sim.WriteInputFile()
    inp = sim.InputFilename
    r = subprocess.run([ISOLVE, inp], cwd=os.path.dirname(inp),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"iSolve failed with {r.returncode}: {r.stdout[-400:]}")
    out_h5 = inp.replace("_Input.h5", "_Output.h5")
    E, I = extract(out_h5)
    np.save(dst, (E * 1e-3 / I).astype(np.float32))       # normalised to 1 mA
    # Each output is 700 MB; 28 of them would be 20 GB. Delete as soon as it is extracted.
    # (The next CreateVoxels would wipe `_Results` anyway.)
    try:
        os.remove(out_h5)
    except OSError:
        pass
    return dst, I, f"{time.time()-t0:.0f}s"


def compare_original(name, enames):
    """Compare against the same electrode in the original leadfieldF (direction and magnitude)."""
    if name not in enames:
        return None
    M = np.load(os.path.join(_LFDIR, f"M{enames.index(name)}.npy")).astype(np.float64)
    un = __import__("json").load(open(IN("unitnorm.json")))
    key = str(enames.index(name))
    if key not in un:
        return None
    Mo = M * float(un[key]) * 0.5                          # config.LEADFIELD_AMP_FIX
    Er = np.load(os.path.join(OUT, f"{name}.npy")).astype(np.float64)
    a, b = np.linalg.norm(Mo, axis=1), np.linalg.norm(Er, axis=1)
    cos = np.median((Mo * Er).sum(1) / np.maximum(a * b, 1e-30))
    return float(np.median(b / np.maximum(a, 1e-30))), float(cos)
