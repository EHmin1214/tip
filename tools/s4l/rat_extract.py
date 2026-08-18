# -*- coding: utf-8 -*-
"""rat_extract.py — turn the NeuroRat port solve into a TIP leadfield
====================================================================
Reads the 37 `*_Output.h5` produced by `iSolve` on `rat_lf.smash` and writes

    inputs/leadfield/leadfield_rat/{electrode}.npy   (N,3) float32, V/m per 1 mA
    inputs/bmask_rat.npy    (N,3) int32   grid indices of the brain voxels
    inputs/gaxes_rat.npz    cx, cy, cz    cell-centre coordinates, mm
    inputs/blabel_rat.npy   (N,)  uint8   tissue label per brain voxel
    inputs/pos_rat.json     {electrode: [x,y,z]}  mm
    inputs/labels_rat.json  {tissue name: label}

★ Normalisation. Every port output is the *raw 1 V basis*: the driven electrode sits at
  1 V, all other electrodes at 0 V, the six outer faces are flux 0. No port current is
  stored, so it is computed here as `I = integral of sigma|E|^2 dV / 1V`.
  **`LEADFIELD_AMP_FIX` must NOT be applied.** That 0.5 exists because the human path
  integrated `El. Loss Density` (= sigma|E|^2 / 2); here E itself is on disk.

★ The electrode a port belongs to is its *file name*: the port h5 basename is the
  electrode's component UUID, and `<smash>/Data/_Object/Simulations/.../Components/<uuid>`
  carries `_Description`. No Sim4Life session is needed for any of this.

★ Tissue labels are assigned here from the fixed `RAT_LABELS` table below, **not** from the
  solver's voxel ids. Those ids are `index_map` positions and would shift if the component
  list ever changed; a stale `blabel_rat.npy` would then silently mean something else.
"""
import os
import sys
import json
import glob
import argparse

import h5py
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUTS = os.path.join(REPO, "inputs")
DESK = os.path.dirname(REPO)

SMASH = os.environ.get("TIP_RAT_SMASH") or os.path.join(DESK, "s4l_projects", "rat_lf.smash")
_PD = glob.glob(os.path.join(DESK, "s4l_projects", "rat_lf.smash_Results", "*_emlf_ports"))
PORTS = os.environ.get("TIP_RAT_PORTS") or (_PD[0] if _PD else "")
OUT = os.path.join(INPUTS, "leadfield", "leadfield_rat")

#  The ten NeuroRat brain structures. Everything else (skull, skin, muscle, CSF, ...) carries
#  current but is not a place we ever score a target, so it stays out of the mask.
RAT_LABELS = {
    "Cerebral_cortex": 1, "Rest_of_brain": 2, "Hippocampus": 3, "Thalamus": 4,
    "Caudo_putamen": 5, "Midbrain": 6, "Pons": 7, "Medulla_oblongata": 8,
    "Cerebellum": 9, "Olfactory_bulb": 10,
}
#  Mesh volumes measured in Sim4Life (mm^3) — the self-check the grid has to pass.
MESH_VOL = {
    "Hippocampus": 31.88, "Thalamus": 39.73, "Cerebral_cortex": 310.83, "Cerebellum": 197.06,
    "Rest_of_brain": 179.15, "Pons": 95.24, "Midbrain": 62.92, "Medulla_oblongata": 42.71,
    "Caudo_putamen": 31.39, "Olfactory_bulb": 26.09,
}


def component_names(smash=SMASH):
    """component UUID (hex, no dashes) -> entity name, straight out of the project file."""
    out = {}
    with h5py.File(smash, "r") as f:
        C = f["Data/_Object/Simulations/_Object/_Group/0/_Object/Components"]
        for k in C:
            d = C[k].attrs.get("_Description", b"")
            d = d.decode("utf-8", "replace") if isinstance(d, bytes) else str(d)
            out[k.replace("-", "").lower()] = d.split("  (")[0].strip()
    return out


def _mesh(f):
    return f["Meshes"][[m for m in f["Meshes"] if "voxels" in f["Meshes"][m]][0]]


def sigma_lut(f):
    """voxel id -> sigma (S/m), plus voxel id -> component uuid.

    Electrodes are PEC: they have no ElectricConductivity property and stay at 0.
    """
    amm = f["AllMaterialMaps"][list(f["AllMaterialMaps"])[0]]
    sig = {}
    for k in amm:
        for pk in amm[k]:
            tn = amm[k][pk]["_ClassInfo"].attrs.get("_TypeName", b"")
            if b"ElectricConductivity" in tn:
                sig[k.replace("-", "").lower()] = float(
                    amm[k][pk]["_Object"].attrs.get("uniform_scalar", 0.0))
                break
    mg = _mesh(f)
    n = int(np.asarray(mg["index_map"]).max()) + 1
    lut = np.zeros(n)
    uuid_of = {}
    for row, idx in zip(mg["id_map"][...], mg["index_map"][...]):
        u = row.tobytes().hex()
        uuid_of[int(idx)] = u
        lut[int(idx)] = sig.get(u, 0.0)
    return lut, uuid_of


def geometry(out_h5, names):
    """Brain mask, labels, cell-centre axes and cell volumes, from any one port output."""
    with h5py.File(out_h5, "r") as f:
        mg = _mesh(f)
        vox = mg["voxels"][...]
        ax = [np.asarray(mg["axis_" + c], float) for c in "xyz"]   # metres, node positions
        lut, uuid_of = sigma_lut(f)
    d = [np.diff(a) for a in ax]
    cellvol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]      # m^3
    centres = [((a[:-1] + a[1:]) * 0.5) * 1e3 for a in ax]                          # mm

    labmap = np.zeros(int(vox.max()) + 1, np.uint8)
    for idx, u in uuid_of.items():
        lab = RAT_LABELS.get(names.get(u, ""))
        if lab is not None and idx < len(labmap):
            labmap[idx] = lab
    lab_vol = labmap[vox]
    sel = lab_vol > 0
    ijk = np.argwhere(sel).astype(np.int32)
    blabel = lab_vol[sel]

    report = {}
    for nm, lab in RAT_LABELS.items():
        v = float(cellvol[lab_vol == lab].sum()) * 1e9
        report[nm] = (int((blabel == lab).sum()), v, MESH_VOL.get(nm))
    return dict(vox=vox, lut=lut, cellvol=cellvol, centres=centres,
                bmask=ijk, blabel=blabel, report=report, uuid_of=uuid_of)


def cell_centre_E(f):
    """Staggered edge components -> cell-centred (nx,ny,nz) each, by 4-edge averaging."""
    fg = f["FieldGroups"]
    key = [x for x in fg if "EM E(x,y,z,f0)" in fg[x]["AllFields"]][0]
    sn = fg[key]["AllFields"]["EM E(x,y,z,f0)"]["_Object"]["Snapshots"]["0"]
    e0 = sn["comp0"][..., 0].astype(np.float32)
    e1 = sn["comp1"][..., 0].astype(np.float32)
    e2 = sn["comp2"][..., 0].astype(np.float32)
    Ex = .25 * (e0[:, :-1, :-1] + e0[:, 1:, :-1] + e0[:, :-1, 1:] + e0[:, 1:, 1:])
    del e0
    Ey = .25 * (e1[:-1, :, :-1] + e1[1:, :, :-1] + e1[:-1, :, 1:] + e1[1:, :, 1:])
    del e1
    Ez = .25 * (e2[:-1, :-1, :] + e2[1:, :-1, :] + e2[:-1, 1:, :] + e2[1:, 1:, :])
    del e2
    return Ex, Ey, Ez


def extract_port(out_h5, geo):
    """(N,3) brain-voxel E in V/m, and the injected current I in amperes."""
    i, j, k = geo["bmask"][:, 0], geo["bmask"][:, 1], geo["bmask"][:, 2]
    with h5py.File(out_h5, "r") as f:
        Ex, Ey, Ez = cell_centre_E(f)
    E = np.stack([Ex[i, j, k], Ey[i, j, k], Ez[i, j, k]], 1).astype(np.float64)
    E2 = (Ex.astype(np.float64) ** 2 + Ey.astype(np.float64) ** 2
          + Ez.astype(np.float64) ** 2)
    del Ex, Ey, Ez
    E2 = np.where(np.isfinite(E2), E2, 0.0)
    #  I = integral sigma |E|^2 dV over the whole domain, with the 1 V drive.
    #  No LEADFIELD_AMP_FIX here — see the module docstring.
    I = float(np.sum(geo["lut"][geo["vox"]] * E2 * geo["cellvol"]))
    return E, I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default=PORTS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    names = component_names()
    outs = sorted(glob.glob(os.path.join(a.ports, "*_Output.h5")))
    if not outs:
        sys.exit("no port outputs under " + a.ports)
    print("%d port outputs" % len(outs))

    geo = geometry(outs[0], names)
    print("\n%-22s%9s%10s%10s%8s" % ("tissue", "nvox", "vox mm3", "mesh mm3", "ratio"))
    worst = 0.0
    for nm, (n, v, t) in geo["report"].items():
        r = v / t if t else float("nan")
        worst = max(worst, abs(r - 1))
        print("%-22s%9d%10.2f%10.2f%8.3f" % (nm, n, v, t or 0, r))
    print("brain voxels N = %d   worst volume error %.1f%%" % (len(geo["bmask"]), worst * 100))
    if worst > 0.10:
        sys.exit("★ voxel volumes disagree with the meshes by more than 10% — stop and look")

    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(INPUTS, "bmask_rat.npy"), geo["bmask"])
    np.savez(os.path.join(INPUTS, "gaxes_rat.npz"),
             cx=geo["centres"][0], cy=geo["centres"][1], cz=geo["centres"][2])
    np.save(os.path.join(INPUTS, "blabel_rat.npy"), geo["blabel"])
    json.dump(RAT_LABELS, open(os.path.join(INPUTS, "labels_rat.json"), "w"), indent=1)
    print("geometry written")
    if a.geometry_only:
        return

    inj = {}
    for p in outs:
        uid = os.path.basename(p).replace("_Output.h5", "").replace("-", "").lower()
        el = names.get(uid, "").replace("Elec_0.25mm", "").strip()
        if not el:
            print("  ! %s -> unknown component, skipped" % os.path.basename(p))
            continue
        dst = os.path.join(a.out, el + ".npy")
        if os.path.exists(dst) and not a.force:
            print("  = %-5s reused" % el)
            continue
        E, I = extract_port(p, geo)
        Mn = (E * 1e-3 / I).astype(np.float32)
        np.save(dst, Mn)
        inj[el] = I
        print("  + %-5s I = %.4f mA per V   median |E| = %.4f V/m per mA"
              % (el, I * 1e3, float(np.median(np.linalg.norm(Mn, axis=1)))))
    if inj:
        p = os.path.join(a.out, "inj_current.json")
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.update(inj)
        json.dump(old, open(p, "w"), indent=1)

    #  Electrode coordinates, straight from the placement run.
    ep = os.path.join(DESK, "rat_electrodes.json")
    if os.path.exists(ep):
        pos = json.load(open(ep))["electrodes"]
        json.dump(pos, open(os.path.join(INPUTS, "pos_rat.json"), "w"), indent=1)
        json.dump(pos, open(os.path.join(a.out, "positions.json"), "w"), indent=1)
        print("positions written (%d)" % len(pos))


if __name__ == "__main__":
    main()
