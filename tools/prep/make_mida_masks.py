# -*- coding: utf-8 -*-
"""
make_mida_masks.py — transfer the MIDA voxel labels onto the leadfield grid, producing every target mask at once
=====================================================================================
`MIDA_v1_voxels/MIDA_v1.raw` is a **headerless uint8 label volume** (480x480x350, isotropic
0.5 mm — stated at the end of `MIDA_v1.txt`).
Transferring it onto the leadfield grid (`bmask1010` + `gaxes1010`) yields **all 126 labels in
one pass**. The alternative, `ViP.WhichPointsAreInSurface`, requires exporting a mesh to file
and repeating per label; this route finishes in a single step.

★The coordinate transform is **derived and then verified**, never guessed
------------------------------------------------
It is already known that `.nii` sits in a different frame from `.sab` (an axis permutation plus
a Y offset). So rather than assuming a transform, the axis correspondence, signs and offset are
solved for by matching the centroids of **seven already-validated masks** (built from the MIDA
meshes) against the centroids in the label volume. The result is then **verified by voxel
overlap**. A poor overlap means the transform is wrong, and nothing is written.

Usage: python make_mida_masks.py [--write]
"""
import io
import json
import os
import sys
from itertools import permutations, product

import numpy as np

# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")

sys.path.insert(0, os.path.join(REPO, "src"))
from tip import LeadField


#  MIDA voxel volume. Not in the repository (licensed data) - point TIP_MIDA_DIR at
#  the extracted "MIDA (Static) - 1.0" folder.
MIDA_DIR = os.environ.get("TIP_MIDA_DIR") or \
    os.path.join(os.path.dirname(REPO), "MIDA (Static) - 1.0")
RAW = os.path.join(MIDA_DIR, "MIDA_v1_voxels", "MIDA_v1.raw")
TXT = os.path.join(MIDA_DIR, "MIDA_v1_voxels", "MIDA_v1.txt")
SHAPE = (480, 480, 350)
STEP = 0.5                      # mm

# Already-validated masks (file name -> MIDA label). These anchor both the derivation and the
# verification of the transform.
ANCHOR = {"amygdala": 4, "hippocampus": 5, "caudate_nucleus": 7,
          "putamen": 8, "nucleus_accumbens": 16, "hypothalamus": 21,
          "thalamus": 116}

# Neural structures worth transferring (peripheral tissue, bone and muscle excluded).
# The id becomes the file name.
WANT = {
    3: "pineal_body", 4: "amygdala", 5: "hippocampus", 6: "csf_ventricles",
    7: "caudate_nucleus", 8: "putamen", 9: "cerebellum_wm", 2: "cerebellum_gm",
    11: "brainstem_midbrain", 13: "spinal_cord", 14: "brainstem_pons",
    15: "brainstem_medulla", 16: "nucleus_accumbens", 17: "globus_pallidus",
    18: "optic_tract", 20: "mammillary_body", 21: "hypothalamus",
    22: "commissura_anterior", 23: "commissura_posterior",
    99: "substantia_nigra", 100: "cerebral_peduncles", 101: "optic_chiasm",
    116: "thalamus",
}


def load_names():
    names = {}
    with io.open(TXT, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5 and p[0].strip().isdigit():
                names[int(p[0])] = p[4].strip()
    return names


def main(write=False):
    lf = LeadField()
    dd = lf.data_dir
    C = lf.coords()                                   # (N,3) mm, the leadfield brain voxels
    names = load_names()

    print("loading the MIDA label volume ...", flush=True)
    vol = np.fromfile(RAW, dtype=np.uint8)
    assert vol.size == SHAPE[0] * SHAPE[1] * SHAPE[2], f"size mismatch {vol.size}"
    vol = vol.reshape(SHAPE, order="F")               # assumes X varies fastest - verified below
    print(f"  {vol.shape} · {len(np.unique(vol))} labels")

    # ── 1. centroids of the anchor structures (label volume in index space, existing masks
    #       in mm) ──
    src, dst = [], []
    for mid, lab in ANCHOR.items():
        p = os.path.join(dd, "masks", f"{mid}.npy")
        if not os.path.exists(p):
            continue
        m = np.load(p)
        idx = np.where(m.any(axis=0) if m.ndim == 2 else m)[0]   # merge left and right
        ii = np.argwhere(vol == lab)
        if len(ii) == 0 or len(idx) == 0:
            print(f"  ⚠ {mid}: label {lab} has {len(ii)} voxels, mask has {len(idx)} - skipping")
            continue
        src.append(ii.mean(0) * STEP)                  # index -> mm (origin still unknown)
        dst.append(C[idx].mean(0))
        print(f"  anchor {mid:18} label {lab:4} voxels {len(ii):7} <-> mask {len(idx):6}")
    src = np.array(src); dst = np.array(dst)
    if len(src) < 3:
        print("not enough anchors - aborting"); return 1

    # ── 2. solve for the axis correspondence, signs and offset, assuming an axis-aligned
    #       transform: permutation x sign x translation ──
    best = None
    for perm in permutations(range(3)):
        for sgn in product((1, -1), repeat=3):
            s = src[:, list(perm)] * np.array(sgn)
            off = (dst - s).mean(0)
            err = np.abs(dst - (s + off)).max()
            if best is None or err < best[0]:
                best = (err, perm, sgn, off)
    err, perm, sgn, off = best
    print(f"\nderived: axis permutation {perm} · signs {sgn} · offset {np.round(off,2)}"
          f" · worst centroid error **{err:.2f} mm**")
    if err > 5.0:
        print("★ the error is large - either the axis-aligned assumption fails or the reshape "
              "order differs. Nothing will be written.")
        return 1

    # ── 3. leadfield voxels -> label-volume indices (the inverse transform) ──
    inv = np.empty_like(C)
    q = (C - off) / np.array(sgn)                      # -> permuted mm
    for a, pa in enumerate(perm):
        inv[:, pa] = q[:, a]
    ijk = np.round(inv / STEP).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(SHAPE)), axis=1)
    lab_at = np.zeros(len(C), np.uint8)
    lab_at[ok] = vol[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    print(f"  {ok.sum()} of {len(C)} leadfield voxels fall inside the volume ({ok.mean()*100:.1f}%)")

    # ── 4. verification: voxel overlap (Dice) against the existing masks ──
    print("\nverification - Dice against the existing masks")
    dices = []
    for mid, lab in ANCHOR.items():
        p = os.path.join(dd, "masks", f"{mid}.npy")
        if not os.path.exists(p):
            continue
        m = np.load(p); ref = (m.any(axis=0) if m.ndim == 2 else m)
        new = (lab_at == lab)
        d = 2 * (ref & new).sum() / max(ref.sum() + new.sum(), 1)
        dices.append(d)
        print(f"  {mid:20} Dice {d:.3f}  (existing {ref.sum():6} · new {new.sum():6})")
    md = float(np.median(dices))
    print(f"  → median Dice {md:.3f}")
    if md < 0.7:
        print("★ the overlap is poor - the transform cannot be trusted. Nothing will be written.")
        return 1

    # ── 5. map everything, splitting left and right at the midline x = MIDLINE ──
    from tip import config as Cfg
    mid_x = Cfg.MIDLINE_X
    out_dir = os.path.join(dd, "masks")
    print(f"\nmapping (midline x={mid_x})")
    made = []
    for lab, mid in sorted(WANT.items()):
        sel = (lab_at == lab)
        if sel.sum() == 0:
            print(f"  {mid:22} 0 voxels - skipping"); continue
        L = sel & (C[:, 0] <= mid_x)
        R = sel & (C[:, 0] > mid_x)
        bilat = min(L.sum(), R.sum()) > 0.15 * sel.sum()
        arr = np.stack([L, R]) if bilat else sel[None, :]
        made.append(dict(id=mid, en=names.get(lab, ""), file=f"{mid}.npy",
                         mida_label=int(lab), bilateral=bool(bilat),
                         vox_L=int(L.sum()), vox_R=int(R.sum()),
                         vox_total=int(sel.sum()),
                         vol_cm3=round(float(sel.sum()) * 8 / 1000, 2),
                         source="MIDA_v1.raw label volume -> leadfield grid "
                                "(derived transform, Dice-verified)"))
        print(f"  {mid:22} label {lab:4} total {sel.sum():6}  L{L.sum():6}/R{R.sum():6}"
              f"  {'bilateral' if bilat else 'single'}")
        if write:
            np.save(os.path.join(out_dir, f"{mid}.npy"), arr)
    if write:
        with io.open(os.path.join(out_dir, "manifest_mida.json"), "w", encoding="utf-8") as f:
            json.dump(made, f, ensure_ascii=False, indent=1)
        print(f"\nwritten - {len(made)} masks plus manifest_mida.json")
    else:
        print("\n(nothing written - pass --write to save)")
    return 0


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv))
