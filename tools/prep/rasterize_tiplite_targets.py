# -*- coding: utf-8 -*-
"""rasterize_tiplite_targets.py — transfer the tip.lite target meshes onto our leadfield grid
===============================================================================
The targets in the tip.lite reference project (`MIDA_Anisotropic.smash`) are triangle meshes in
a different coordinate frame from our MIDA source. Since both projects use **the same MIDA
v1.0 tissue meshes**, a rigid transform was fitted from the volume centroids of the 115
identically-named tissues (`fit_frames.py`, median residual 0.119 mm). The target meshes are
mapped back into our frame with it, and a mask is built by testing whether each of our grid
cell centres lies inside the mesh.

The inside test is **scanline ray casting along X**. For each (y, z) grid line, every
intersection x with the mesh is collected and sorted, and the spans are filled by the
even-odd rule. A closed mesh must give an even number of intersections, so the count of
odd-numbered columns doubles as a self-check.

Output is `<masks>/masks_tiplite/<name>.npy` — a bool array in `bmask1010` order
(N = 1,907,678).

Usage:
    python rasterize_tiplite_targets.py                 # the default target set
    python rasterize_tiplite_targets.py --all           # all 414
    python rasterize_tiplite_targets.py "Targets_bn/*tha*"
"""
import argparse
import fnmatch
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")
from tip.config import inputs as IN, LEADFIELD_DIR as _LFDIR   # input-file resolver

DD = INPUTS
#  Working directory for large intermediates. Override with TIP_SCRATCH.
SP = os.environ.get("TIP_SCRATCH") or os.path.join(REPO, "outputs", "scratch")
OUT = os.path.join(INPUTS, "masks", "masks_tiplite")

# The default set — the targets that correspond directly to our existing masks
DEFAULT = [
    "Targets_split/Thalamus Left", "Targets_split/Thalamus Right",
    "Targets_split/Hippocampus Left", "Targets_split/Hippocampus Right",
    "Targets_combined/Thalamus", "Targets_combined/Hippocampus",
]


def load_grid():
    g = np.load(IN("gaxes1010.npz"))
    return g["cx"].astype(np.float64), g["cy"].astype(np.float64), g["cz"].astype(np.float64)


def voxelize(Q, T, cx, cy, cz, chunk=20000):
    """Fill the grid from vertices Q (in mm, our frame) and triangles T, returning flat
    (i, j, k) indices.

    Returns: (array of flat indices, number of columns with an odd intersection count)
    """
    ny, nz = len(cy), len(cz)
    v0, v1, v2 = Q[T[:, 0]], Q[T[:, 1]], Q[T[:, 2]]
    # Nudge the ray origin slightly off the grid centre to avoid hitting vertices and edges
    # exactly
    ey, ez = 1.0e-4 * np.sqrt(2.0), 1.0e-4 * np.sqrt(3.0)
    sy, sz = cy + ey, cz + ez

    cols, xs = [], []
    for s in range(0, len(T), chunk):
        a0, a1, a2 = v0[s:s + chunk], v1[s:s + chunk], v2[s:s + chunk]
        ylo = np.minimum(np.minimum(a0[:, 1], a1[:, 1]), a2[:, 1])
        yhi = np.maximum(np.maximum(a0[:, 1], a1[:, 1]), a2[:, 1])
        zlo = np.minimum(np.minimum(a0[:, 2], a1[:, 2]), a2[:, 2])
        zhi = np.maximum(np.maximum(a0[:, 2], a1[:, 2]), a2[:, 2])
        j0 = np.searchsorted(sy, ylo, "left"); j1 = np.searchsorted(sy, yhi, "right")
        k0 = np.searchsorted(sz, zlo, "left"); k1 = np.searchsorted(sz, zhi, "right")
        nj, nk = np.maximum(j1 - j0, 0), np.maximum(k1 - k0, 0)
        n = nj * nk
        if n.sum() == 0:
            continue
        ti = np.repeat(np.arange(len(n)), n)                   # triangle indices
        off = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        jj = j0[ti] + off // nk[ti]
        kk = k0[ti] + off % nk[ti]
        py, pz = sy[jj], sz[kk]
        b0, b1, b2 = a0[ti], a1[ti], a2[ti]
        e1y, e1z = b1[:, 1] - b0[:, 1], b1[:, 2] - b0[:, 2]
        e2y, e2z = b2[:, 1] - b0[:, 1], b2[:, 2] - b0[:, 2]
        det = e1y * e2z - e2y * e1z
        ok = np.abs(det) > 1e-14
        qy, qz = py - b0[:, 1], pz - b0[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (qy * e2z - e2y * qz) / det
            v = (e1y * qz - qy * e1z) / det
        ok &= (u > 0) & (v > 0) & (u + v < 1)
        if not ok.any():
            continue
        u, v, ti2 = u[ok], v[ok], ti[ok]
        x = b0[ok, 0] + u * (b1[ok, 0] - b0[ok, 0]) + v * (b2[ok, 0] - b0[ok, 0])
        cols.append(jj[ok].astype(np.int64) * nz + kk[ok].astype(np.int64))
        xs.append(x)
    if not cols:
        return np.zeros(0, np.int64), 0
    col = np.concatenate(cols); x = np.concatenate(xs)

    order = np.lexsort((x, col))
    col, x = col[order], x[order]
    ucol, start, cnt = np.unique(col, return_index=True, return_counts=True)
    odd = int((cnt % 2).sum())

    flats = []
    nyz = nz
    for c, st, ct in zip(ucol, start, cnt):
        if ct % 2:
            continue                                   # discard open columns (hygiene)
        j, k = divmod(int(c), nyz)
        xv = x[st:st + ct]
        for p in range(0, ct, 2):
            i0 = np.searchsorted(cx, xv[p], "left")
            i1 = np.searchsorted(cx, xv[p + 1], "right")
            if i1 > i0:
                flats.append(np.arange(i0, i1) * (len(cy) * nz) + j * nyz + k)
    if not flats:
        return np.zeros(0, np.int64), odd
    return np.unique(np.concatenate(flats)), odd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    fr = np.load(os.path.join(SP, "frame_ours2tip.npz"))
    R, t = fr["R"], fr["t"]                       # tip = R @ ours + t
    Z = np.load(os.path.join(SP, "tiplite_targets.npz"))
    meta = json.load(open(os.path.join(SP, "tiplite_targets_meta.json")))
    keys = sorted(k[:-2] for k in Z.files if k.endswith("|P"))
    if a.all:
        sel = keys
    elif a.patterns:
        sel = [k for k in keys if any(fnmatch.fnmatch(k, p) for p in a.patterns)]
    else:
        sel = [k for k in DEFAULT if k in keys]
    print(f"{len(sel)}/{len(keys)} targets")

    cx, cy, cz = load_grid()
    dx, dy, dz = (np.gradient(c) for c in (cx, cy, cz))
    bm = np.load(IN("bmask1010.npy")).astype(np.int64)
    bflat = bm[:, 0] * (len(cy) * len(cz)) + bm[:, 1] * len(cz) + bm[:, 2]
    bsort = np.argsort(bflat); bs = bflat[bsort]
    os.makedirs(OUT, exist_ok=True)

    print(f"{'target':38}{'voxels':>8}{'in brain':>10}{'vol mm3':>11}{'mesh mm3':>11}"
          f"{'ratio':>7}{'odd col':>9}{'s':>6}")
    rep = {}
    for k in sel:
        t0 = time.time()
        P = Z[k + "|P"].astype(np.float64)
        Q = (P - t) @ R                              # tip frame -> our frame
        flat, odd = voxelize(Q, Z[k + "|T"].astype(np.int64), cx, cy, cz)
        pos = np.searchsorted(bs, flat)
        pos = np.clip(pos, 0, len(bs) - 1)
        hit = bs[pos] == flat
        bidx = np.sort(bsort[pos[hit]])
        i, j, kk = np.divmod(flat, len(cy) * len(cz))[0], 0, 0
        ii = flat // (len(cy) * len(cz)); rest = flat % (len(cy) * len(cz))
        jj = rest // len(cz); kk = rest % len(cz)
        vol = float((dx[ii] * dy[jj] * dz[kk]).sum())
        mv = meta[k]["vol"]
        mask = np.zeros(len(bm), bool); mask[bidx] = True
        # Windows file names are case-insensitive, so ICBM's `Thalamus_left` collides with the
        # MIDA split `Thalamus Left`. The group prefix is therefore mandatory.
        grp = k.split("/", 1)[0].replace("Targets_", "")
        np.save(os.path.join(OUT, f"{grp}__{k.split('/',1)[1].replace(' ','_')}.npy"), mask)
        rep[k] = dict(vox=int(len(flat)), inbrain=int(len(bidx)), vol=vol, mesh_vol=mv,
                      ratio=vol / mv, odd=odd)
        print(f"{k.split('/',1)[1]:38}{len(flat):>8}{len(bidx):>8}{vol:>11.1f}{mv:>11.1f}"
              f"{vol/mv:>7.3f}{odd:>7}{time.time()-t0:>6.1f}")
    json.dump(rep, open(os.path.join(OUT, "_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
