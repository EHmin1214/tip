# -*- coding: utf-8 -*-
"""electrode_add.py — **place a new electrode** in the rebuilt project, solve it, add it to the leadfield
================================================================================
`rebuild_solve_batch.py` can only drive electrodes that **already exist**. The single thing
this file adds is **creating the electrode body**; everything else — swapping the driven
electrode, voxelling, solving, extracting, normalising — reuses the validated functions in
`rebuild_solve_batch`, so the conventions cannot drift apart.

    ViP.Create1010System(..., add_outer_ring=True) → 85 vertices
    71 already solved (the standard 61 plus 10 lower-ring) → **the remaining 14** are this
    file's job

Runs **inside a Sim4Life worker** (needs `s4l_v1`). For batch execution see `add_electrodes.py`.

Conventions (established during the leadfield rebuild; breaking them makes results
incompatible with the existing set)
--------------------------------------------------------------------
- electrode body = a **cylinder of r = 4 mm, h = 2 mm centred at the origin (axis Z, -1..+1)**,
  rotated onto the scalp normal and translated into place
- name = `{electrode}_ElectrodeTemplate`, in the `"Cloned Templates"` group
- **added to the voxeler but assigned no material** (a transparent electrode — undriven
  electrodes then have no effect on the result)
- saving is done by `rebuild_solve_batch.solve_one`: `E * 1e-3 / I` (**normalised to 1 mA**).
  `unitnorm` and `LEADFIELD_AMP_FIX` must **not** be applied

⚠ The grid assertion must pass. Adding an entity can perturb the automatic grid, and a
  difference of a single cell invalidates the entire `bmask` mapping
  (`flat = i + 185j + 185·254k`).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rebuild_solve_batch as R  # noqa: E402

SKIN = "Epidermis_Dermis"
GROUP = "Cloned Templates"
R_ELEC, H_ELEC = 4.0, 2.0          # mm — identical to the original electrodes
LANDMARKS = os.path.join(R.DD, "landmarks.npy")   # Nz, Iz, RPA, LPA


# ───────────────────────── vertices ─────────────────────────
def _entity(name):
    import XCoreModeling as xm
    for e in xm.GetActiveModel().GetEntities():
        if e.Name == name:
            return e
    raise KeyError(f"entity not found: {name}")


def vertices_1010(add_outer_ring=True, verbose=True):
    """The vertices from `ViP.Create1010System`, as {name: (x, y, z)}.

    ⚠ The return type varies between versions (entity group / list / (name, point) pairs), so
      it is **detected rather than assumed**. An unrecognised shape is reported and stops the
      run — better than silently using wrong coordinates.
    """
    import ViP
    import XCoreModeling as xm
    skin = _entity(SKIN)
    lm = np.load(LANDMARKS).astype(float)          # (4,3) = Nz, Iz, RPA, LPA
    import s4l_v1
    v3 = [s4l_v1.Vec3(*map(float, p)) for p in lm]
    out = ViP.Create1010System(skin, v3[0], v3[1], v3[2], v3[3],
                               add_outer_ring=add_outer_ring)

    def _xyz(o):
        for attr in ("Position", "Point", "Center"):
            p = getattr(o, attr, None)
            if p is not None:
                try:
                    return [float(p[0]), float(p[1]), float(p[2])]
                except Exception:
                    pass
        bb = xm.GetBoundingBox([o])
        return [float(0.5 * (bb[0][i] + bb[1][i])) for i in range(3)]

    ents = list(getattr(out, "Entities", None) or out)
    pos = {}
    for e in ents:
        nm = getattr(e, "Name", None)
        if nm is None:
            raise TypeError(f"unrecognised vertex return type: {type(out).__name__} / "
                            f"element {type(e).__name__} · dir={dir(e)[:25]}")
        pos[str(nm)] = _xyz(e)
    if verbose:
        print(f"[electrode_add] Create1010System → {len(pos)} vertices")
    return pos


def missing_names(pos, out_dir=None):
    """Names of electrodes not yet solved (case-insensitive). `Cz` is excluded — it is the
    reference."""
    out_dir = out_dir or R.OUT
    have = {os.path.splitext(f)[0].lower()
            for f in os.listdir(out_dir)} if os.path.isdir(out_dir) else set()
    return [n for n in sorted(pos) if n.lower() not in have and n != "Cz"]


def min_gap(pos, name, others):
    """Minimum spacing to the existing electrodes in mm. The original set's minimum was
    21.5 mm; much closer than that means they overlap."""
    p = np.asarray(pos[name], float)
    d = [float(np.linalg.norm(p - np.asarray(pos[o], float))) for o in others if o != name]
    return min(d) if d else float("inf")


# ───────────────────────── electrode body ─────────────────────────
def _group():
    import XCoreModeling as xm
    m = xm.GetActiveModel()
    for e in m.RootGroup.Entities:
        if e.Name == GROUP:
            return e
    raise KeyError(f"no '{GROUP}' group - check this really is the rebuilt project")


def place_electrode(name, p, verbose=True):
    """Create an electrode body at the scalp point `p` and register it as
    `{name}_ElectrodeTemplate`.

    `ViP.PlaceElectrodes` cannot be used: a vertex from `xm.CreatePoint` carries no surface
    parameterisation, so it returns an **empty group**. The established route is a rotation
    into the normal frame followed by a translation (validated: centre error 0 mm, 0.31 mm
    against the predicted AABB).
    """
    import QTech
    import s4l_v1
    import ViP
    import XCoreModeling as xm
    from s4l_v1.model import Transform, Vec3

    tgt = f"{name}_ElectrodeTemplate"
    grp = _group()
    for e in grp.Entities:
        if e.Name == tgt:
            if verbose:
                print(f"[electrode_add] {tgt} already exists - reusing")
            return e

    skin = _entity(SKIN)
    p = [float(x) for x in p]
    n = ViP.ComputeSurfaceNormal(skin, Vec3(*p))
    n = np.array([float(n[0]), float(n[1]), float(n[2])], float)
    n /= np.linalg.norm(n)
    # Build u from whichever axis is least parallel to n (numerical stability)
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(a, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # Cylinder r=4, h=2 centred at the origin along Z — the same spec as the original electrodes
    body = xm.CreateSolidCylinder(Vec3(0.0, 0.0, -H_ELEC / 2),
                                  Vec3(0.0, 0.0, H_ELEC / 2), R_ELEC)
    #  Mat3 takes **column** vectors: u, v, n are the columns of the new basis
    body.ApplyTransform(Transform(QTech.Mat3(Vec3(*u), Vec3(*v), Vec3(*n))))
    body.ApplyTransform(Transform(Vec3(1, 1, 1), Vec3(0, 0, 0), Vec3(*p)))
    body.Name = tgt
    if hasattr(grp, "Add"):
        grp.Add(body)
    else:
        body.ParentEntity = grp

    bb = xm.GetBoundingBox([body])
    c = np.array([0.5 * (bb[0][i] + bb[1][i]) for i in range(3)])
    err = float(np.linalg.norm(c - np.array(p)))
    if err > 0.5:
        raise RuntimeError(f"{name} electrode centre off by {err:.3f} mm - the transform is wrong")
    if verbose:
        print(f"[electrode_add] {tgt} created · centre error {err:.4f} mm · normal {n.round(3)}")
    return body


def attach_voxeler(sim, ent, verbose=True):
    """Attach an entity to the voxeler settings. **No material is assigned** (transparent
    electrode).

    ⚠ `sim.Add(vox, [ent])` **succeeds only for the first call in a process**; after that it
      dies with `ValueError: ... is incompatible with this type of simulation` (observed).
      That message is misleading — all that happened is `raw.AddSettings` returning False.

    What works is the **component-level API**, the same one `rebuild_solve_batch.set_src` uses:
        sim.raw.AcquireComponent(entity).AssignSettings(vox.raw)
    """
    vox = [s for s in sim.AllSettings
           if type(s).__name__ in ("AutomaticVoxelerSettings", "ManualVoxelerSettings")]
    if not vox:
        raise RuntimeError("no voxeler settings found")
    v = vox[0]
    name = str(ent.Name)

    def _nm(c):
        me = c.ModelEntity
        return None if me is None else str(me.Name)

    try:
        n = v.raw.SizeAssignedComponents()
        if any(_nm(v.raw.AssignedComponent(i)) == name for i in range(n)):
            if verbose:
                print(f"[electrode_add] {name} already attached to the voxeler")
            return v
    except Exception:
        pass                                  # if the query fails, just try attaching

    if not sim.raw.AcquireComponent(ent).AssignSettings(v.raw):
        raise RuntimeError(f"failed to attach {name} to voxeler '{v.Name}'")
    if verbose:
        print(f"[electrode_add] attached {name} to voxeler '{v.Name}'")
    return v


# ───────────────────────── grid assertion ─────────────────────────
def assert_grid_h5(out_h5, tol=1e-6):
    """Check that the grid in the **output h5** matches `gaxes1010`. A difference of one cell
    invalidates the bmask mapping.

    ⚠ There is no `sim.Grid` attribute (observed). The axes of the voxelled result are only
      reliably available in the output h5, and that is the grid `extract()` actually indexes —
      so **measuring it here is the correct place.**
    """
    import h5py
    g = np.load(os.path.join(R.DD, "gaxes1010.npz"))
    ref = [g["cx"], g["cy"], g["cz"]]          # cell centres (mm), reference 185x254x228
    with h5py.File(out_h5, "r") as f:
        mesh = [m for m in f["Meshes"] if "voxels" in f["Meshes"][m]][0]
        ax = [np.asarray(f["Meshes"][mesh][f"axis_{c}"], float) for c in "xyz"]
    cc = [0.5 * (a[1:] + a[:-1]) for a in ax]        # grid points -> cell centres
    if max(abs(c).max() for c in cc) < 1.0:          # metres -> mm
        cc = [c * 1e3 for c in cc]
    for k, (a, b) in enumerate(zip(cc, ref)):
        if len(a) != len(b):
            raise RuntimeError(f"grid axis {k}: {len(a)} cells != reference {len(b)} - mapping invalid")
        e = float(np.abs(a - b).max())
        if e > tol:
            raise RuntimeError(f"grid axis {k}: max error {e:.3e} mm > {tol} - mapping invalid")
    return tuple(len(c) for c in cc)


# ───────────────────────── end to end ─────────────────────────
def add_and_solve(sim, xm, name, p, force=False, verbose=True):
    """Place the electrode if needed, solve it, and save to `R.OUT/{name}.npy`.

    Why `R.solve_one` cannot be used directly: it **deletes the output h5 after saving**. The
    grid assertion can only be made from that h5, so it has to run **before** the save.
    This reproduces the same sequence while still delegating the risky parts (`set_src`,
    `extract`) to R.

    Returns: (path, I in amperes or None, status string)
    """
    import subprocess
    import time
    dst = os.path.join(R.OUT, f"{name}.npy")
    if os.path.exists(dst) and not force:
        return dst, None, "reused"
    os.makedirs(R.OUT, exist_ok=True)

    ent = place_electrode(name, p, verbose=verbose)
    attach_voxeler(sim, ent, verbose=verbose)

    R.set_src(sim, xm, name)
    t0 = time.time()
    # ★WriteInputFile fails unless CreateVoxels is called first. (The message
    #  `Failure type: network` is just the default exception text, not the real cause.)
    #  ⚠ CreateVoxels deletes `_Results` — move any earlier output out first.
    sim.CreateVoxels()
    sim.WriteInputFile()
    inp = sim.InputFilename
    r = subprocess.run([R.ISOLVE, inp], cwd=os.path.dirname(inp),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"iSolve failed with {r.returncode}: {r.stdout[-400:]}")
    out_h5 = inp.replace("_Input.h5", "_Output.h5")

    dims = assert_grid_h5(out_h5)                 # ★judged before saving
    if verbose:
        print(f"[electrode_add] grid assertion passed {dims}")

    E, I = R.extract(out_h5)
    np.save(dst, (E * 1e-3 / I).astype(np.float32))   # 1 mA normalisation (rebuild convention)
    try:
        os.remove(out_h5)                         # each output is 700 MB
    except OSError:
        pass
    return dst, I, f"{time.time()-t0:.0f}s"


def update_positions(pos, names, out_dir=None):
    """Add the new electrode coordinates to `positions.json`.
    Skip this and `app.py` **silently drops** electrodes missing from `LF.pos` out of the GUI
    list."""
    out_dir = out_dir or R.OUT
    pth = os.path.join(out_dir, "positions.json")
    cur = json.load(open(pth, encoding="utf-8")) if os.path.exists(pth) else {}
    for n in names:
        if n in pos:
            cur[n] = [round(float(x), 4) for x in pos[n]]
    json.dump(cur, open(pth, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return len(cur)
