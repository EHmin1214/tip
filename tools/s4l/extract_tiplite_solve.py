# -*- coding: utf-8 -*-
"""extract_tiplite_solve.py — bring a tip.lite reference solve onto our brain voxels
==================================================================================
Solving the reference project (a copy of `MIDA_Anisotropic.smash`) under our conventions
(TP8 at 1 V, Cz at 0 V) produces **a reference field independent of ours**. Resampled onto our
`bmask1010` voxels, it decides directly which of the original `leadfieldF` and the rebuild sits
closer to that reference.

Coordinates: the two projects are registered by `frame_ours2tip.npz` (Kabsch on 114 pairs of
tissue centroids, residual 0.119 mm).

    tip = R @ ours + t        →  where to sample
    E_ours = R.T @ E_tip      →  the vectors have to be rotated back as well

The reference grid is a uniform 0.4 mm and ours is non-uniform 0.4-8.7 mm, so this is
**downsampling**. The trilinear interpolation error is far smaller than the effects being
chased here (a factor of 0.45, an 11% depth difference).

Produces `<OUT>/<electrode>.npy` — (N,3) float32, **normalised to 1 mA**
      (the same convention as `lf.elec_field`, so the two are directly interchangeable)
"""
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")
from tip.config import inputs as IN, LEADFIELD_DIR as _LFDIR   # input-file resolver

DD = INPUTS
#  Working directory for large intermediates. Override with TIP_SCRATCH.
SP = os.environ.get("TIP_SCRATCH") or os.path.join(REPO, "outputs", "scratch")
OUT = os.path.join(SP, "tiplite_lf")


def sigma_lut(f):
    """Voxel material index -> conductivity (S/m) lookup, plus the grid axes in metres."""
    mats = f["AllMaterialMaps"][list(f["AllMaterialMaps"])[0]]
    sig = {}
    for k in mats:
        for pk in mats[k]:
            tn = mats[k][pk]["_ClassInfo"].attrs.get("_TypeName", b"")
            if b"ElectricConductivity" in tn:
                sig[k.replace("-", "").lower()] = float(
                    mats[k][pk]["_Object"].attrs["uniform_scalar"])
                break
    mesh = [m for m in f["Meshes"] if "voxels" in f["Meshes"][m]][0]
    mg = f["Meshes"][mesh]
    vox = mg["voxels"]
    lut = np.zeros(int(np.asarray(vox[:1]).max()) + 1)   # resized below
    ids = np.array([row.tobytes().hex() for row in mg["id_map"][...]])
    idx = mg["index_map"][...]
    lut = np.zeros(int(idx.max()) + 1)
    for u, i in zip(ids, idx):
        if u in sig:
            lut[i] = sig[u]
    ax = [np.asarray(mg[f"axis_{c}"], float) for c in "xyz"]
    return lut, mg, ax


def cellcenter(sn, c, shape):
    """Staggered-grid edge components -> cell centres, averaging the four parallel edges.
    Stays in float32."""
    e = sn[f"comp{c}"][..., 0].astype(np.float32)
    if c == 0:
        r = .25 * (e[:, :-1, :-1] + e[:, 1:, :-1] + e[:, :-1, 1:] + e[:, 1:, 1:])
    elif c == 1:
        r = .25 * (e[:-1, :, :-1] + e[1:, :, :-1] + e[:-1, :, 1:] + e[1:, :, 1:])
    else:
        r = .25 * (e[:-1, :-1, :] + e[1:, :-1, :] + e[:-1, 1:, :] + e[1:, 1:, :])
    del e
    assert r.shape == shape, (r.shape, shape)
    return np.nan_to_num(r, copy=False)


def trilinear(vol, ax, P):
    """Trilinear interpolation of the cell-centred volume `vol` at points `P` (M,3, same axis
    units)."""
    out = np.zeros(len(P), np.float64)
    w0 = []
    for d in range(3):
        c = 0.5 * (ax[d][:-1] + ax[d][1:])              # cell centres
        i = np.clip(np.searchsorted(c, P[:, d]) - 1, 0, len(c) - 2)
        t = (P[:, d] - c[i]) / (c[i + 1] - c[i])
        w0.append((i, np.clip(t, 0.0, 1.0)))
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                w = ((1 - w0[0][1]) if bx == 0 else w0[0][1])
                w = w * ((1 - w0[1][1]) if by == 0 else w0[1][1])
                w = w * ((1 - w0[2][1]) if bz == 0 else w0[2][1])
                out += w * vol[w0[0][0] + bx, w0[1][0] + by, w0[2][0] + bz]
    return out


def main(out_h5, name):
    os.makedirs(OUT, exist_ok=True)
    fr = np.load(os.path.join(SP, "frame_ours2tip.npz"))
    R, t = fr["R"], fr["t"]
    bm = np.load(IN("bmask1010.npy"))
    g = np.load(IN("gaxes1010.npz"))
    Pours = np.stack([g["cx"][bm[:, 0]], g["cy"][bm[:, 1]], g["cz"][bm[:, 2]]], 1)
    Ptip = Pours @ R.T + t                               # mm, in the tip.lite frame
    print(f"brain voxels {len(Ptip)} · extent in the tip frame "
          f"x{Ptip[:,0].min():.1f}~{Ptip[:,0].max():.1f} "
          f"y{Ptip[:,1].min():.1f}~{Ptip[:,1].max():.1f} "
          f"z{Ptip[:,2].min():.1f}~{Ptip[:,2].max():.1f}")

    with h5py.File(out_h5, "r") as f:
        lut, mg, ax = sigma_lut(f)
        vox = mg["voxels"]
        shape = tuple(len(a) - 1 for a in ax)
        print("grid:", shape, "= %.1f MCells" % (np.prod(shape) / 1e6))
        axm = [a * 1e3 for a in ax]                      # m → mm
        for d in range(3):
            lo, hi = axm[d].min(), axm[d].max()
            assert Ptip[:, d].min() > lo and Ptip[:, d].max() < hi, \
                f"axis {d} out of range: {Ptip[:,d].min():.1f}~{Ptip[:,d].max():.1f} vs {lo:.1f}~{hi:.1f}"
        fg = f["FieldGroups"]
        key = [x for x in fg if "EM E(x,y,z,f0)" in fg[x]["AllFields"]][0]
        sn = fg[key]["AllFields"]["EM E(x,y,z,f0)"]["_Object"]["Snapshots"]["0"]

        d = [np.diff(a) for a in ax]
        dV = (d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :])
        sig = lut[np.asarray(vox)]
        I = 0.0
        E = np.empty((len(Ptip), 3))
        for c in range(3):
            V = cellcenter(sn, c, shape)
            print(f"  component {c} centred, {V.nbytes/1e9:.2f} GB", flush=True)
            I += float(np.sum(sig * (V.astype(np.float64) ** 2) * dV))
            E[:, c] = trilinear(V, axm, Ptip)
            del V
    print(f"injected current I = {I*1e3:.4f} mA")
    Eours = E @ R                                        # E_ours = R.T @ E_tip
    np.save(os.path.join(OUT, f"{name}.npy"), (Eours * 1e-3 / I).astype(np.float32))
    print("saved:", os.path.join(OUT, f"{name}.npy"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "TP8")
