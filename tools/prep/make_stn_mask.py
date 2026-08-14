# -*- coding: utf-8 -*-
"""
make_stn_mask.py — build the STN (subthalamic nucleus) target mask
==================================================
**MIDA v1.0 has no subthalamic nucleus.** Its deep grey matter is limited to amygdala,
hippocampus, caudate, putamen, accumbens, pallidum, hypothalamus, substantia nigra and
thalamus (checked across all 126 labels). So this cannot come from a segmentation.

Instead it is built in the **AC-PC stereotactic frame**, the standard approach in clinical DBS
targeting. MIDA provides the anterior and posterior commissure meshes, and `.sab` is in **the
same coordinate system** as the leadfield (MIGRATION_STATUS §2-2c), so no transform error
enters.

    AC  = [-28.4, 271.0, 11.8]     ← Sim4Life mesh centre, used to derive MCP
    PC  = [-28.2, 273.2, 38.8]
    MCP = (AC+PC)/2, AC-PC distance 27.1 mm (normal human range 23-28 mm — verified)

    STN = MCP + 12·e_LR (lateral) + 3·e_AP (posterior) - INF·e_SI (inferior)
          ← the clinical standard is INF = 4; the default here is 2, for the reason below

**Verified anatomical relations**: 5.0 mm dorsal to substantia nigra, 10.5 mm ventral to
thalamus, 12 mm lateral of the midline.

⚠️ **This is an approximation on two levels. Know both before using it.**

**(1) A coordinate approximation, not a segmentation** — a sphere placed at standard
stereotactic coordinates. The real STN is lens-shaped (~150 mm³, with large inter-subject
variation) and is not a sphere. Precise DBS work needs registration to the DISTAL / Ewert atlas.

**(2) ★The leadfield does not fully contain the STN — the more fundamental limit**

Checking `blabel1010`, the brain mask contains only **GM, WM and seven deep structures** — it
has **no midbrain, brainstem or cerebellum**. The STN sits on the diencephalon-midbrain
boundary, exactly where that coverage stops. Coverage by inferior offset from MCP (at a radius
of 3.3 mm):

    inferior 0 mm → 174 voxels · 2 mm → 78 · 3 mm → 38 ·
    **4 mm (the standard STN centre) → 9** · 5 mm → 0

In other words, **there is no E-field data at all at the anatomical STN centre.** The default
here is therefore an inferior offset of **2 mm** — the **dorsolateral STN**, which is the DBS
sweet spot used clinically and which the mask does cover at that height. **The ventral STN
cannot be addressed with this leadfield.**

> **The real fix**: include the midbrain in the brain mask and regenerate the leadfield
> (SETUP.md §9). About 2.2 minutes per electrode across 70 electrodes, and the solver needs
> the GUI (headless submission was not possible at the time this was written).

Usage: python make_stn_mask.py [radius_mm] [inferior_offset_mm]     (defaults 4.0, 2.0)
"""
import json
import os
import sys

import numpy as np

# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")

sys.path.insert(0, os.path.join(REPO, "src"))
from tip import LeadField
from tip import config as C


# Centres of the MIDA commissure meshes, read from Sim4Life (leadfield coordinates, mm)
AC = np.array([-28.4, 271.0, 11.8])
PC = np.array([-28.2, 273.2, 38.8])
# Standard stereotactic offsets from MCP. The Benabid convention is lateral 12, posterior 3,
# inferior 4.
# The inferior offset defaults to **2.0** here: at 4 (the anatomical STN centre) the sphere
# falls outside the brain mask and only 9 voxels remain (see the module docstring). An offset
# of 2 is the dorsolateral STN, i.e. the clinical DBS sweet spot.
LAT, POST = 12.0, 3.0
INF_DEFAULT = 2.0
RAD_DEFAULT = 4.0


def acpc_frame():
    """The AC-PC stereotactic basis (e_LR right, e_AP posterior, e_SI superior) plus MCP."""
    mcp = (AC + PC) / 2.0
    e_ap = (PC - AC); e_ap /= np.linalg.norm(e_ap)
    y = np.array([0.0, 1.0, 0.0])                      # leadfield +y is superior
    e_si = y - np.dot(y, e_ap) * e_ap; e_si /= np.linalg.norm(e_si)
    e_lr = np.cross(e_si, e_ap); e_lr /= np.linalg.norm(e_lr)
    if e_lr[0] < 0:                                    # -x is left, so orient +e_LR to the right
        e_lr = -e_lr
    return mcp, e_lr, e_ap, e_si


def stn_centers(inf=INF_DEFAULT):
    mcp, e_lr, e_ap, e_si = acpc_frame()
    base = mcp + POST * e_ap - inf * e_si
    return base - LAT * e_lr, base + LAT * e_lr        # (left, right)


def main():
    r = float(sys.argv[1]) if len(sys.argv) > 1 else RAD_DEFAULT
    INF = float(sys.argv[2]) if len(sys.argv) > 2 else INF_DEFAULT
    LF = LeadField(); dd = LF.data_dir
    coords = LF.coords()
    blab = np.load(os.path.join(dd, "blabel1010.npy"))
    neural = np.isin(blab, (C.LABEL_GM, C.LABEL_WM))   # brain tissue only (excludes CSF etc.)

    cl, cr = stn_centers(INF)
    mcp, e_lr, e_ap, e_si = acpc_frame()
    print("=" * 66)
    print("building the STN mask - AC-PC stereotactic coordinates")
    print("=" * 66)
    print(f"  AC {AC} · PC {PC}")
    print(f"  MCP {np.round(mcp,2)} · AC-PC {np.linalg.norm(PC-AC):.2f} mm")
    print(f"  offsets lateral {LAT} · posterior {POST} · inferior {INF} mm · radius {r} mm")
    if INF >= 3.5:
        print("  ⚠️ beyond 3.5 mm inferior the sphere leaves the brain mask and almost no "
              "voxels remain (see the module docstring)")
    print(f"\n  STN_L {np.round(cl,2)}\n  STN_R {np.round(cr,2)}")

    mask = np.zeros((2, LF.N), bool)
    for k, c in enumerate((cl, cr)):
        d2 = ((coords - c) ** 2).sum(1)
        mask[k] = (d2 <= r * r) & neural

    # Voxel volumes — the grid is graded, so convert to real volume
    g = np.load(os.path.join(dd, "gaxes1010.npz"))
    dx = np.gradient(g["cx"]); dy = np.gradient(g["cy"]); dz = np.gradient(g["cz"])
    b = LF.bmask
    vvol = dx[b[:, 0]] * dy[b[:, 1]] * dz[b[:, 2]]     # mm³ per voxel

    print(f"\n  left  {int(mask[0].sum()):>5} voxels · {vvol[mask[0]].sum()/1000:.3f} cm3")
    print(f"  right {int(mask[1].sum()):>5} voxels · {vvol[mask[1]].sum()/1000:.3f} cm3")
    sph = 4/3*np.pi*r**3/1000
    cov = (vvol[mask[0]].sum()+vvol[mask[1]].sum())/2000 / sph
    print(f"  the brain mask covers {cov*100:.0f}% of the {sph:.3f} cm3 sphere")
    print(f"  (the real STN is ~0.15 cm3; the uncovered part is midbrain, absent from the "
          f"leadfield)")
    if mask[0].sum() == 0 or mask[1].sum() == 0:
        print("\n★failed: zero brain-tissue voxels. Check the coordinates or the radius.")
        return 1

    # Check the tissue composition
    for k, nm in ((0, "left"), (1, "right")):
        u, c_ = np.unique(blab[mask[k]], return_counts=True)
        comp = " ".join(f"{int(a)}:{int(b_)}" for a, b_ in zip(u, c_))
        print(f"  {nm} tissue composition (blabel:voxels) {comp}   [75=GM 131=WM]")

    out = os.path.join(dd, "masks", "stn.npy")
    np.save(out, mask)
    print(f"\n  saved {out}")

    # update the manifest
    mf = os.path.join(dd, "masks", "manifest.json")
    man = json.load(open(mf, encoding="utf-8"))
    man = [e for e in man if e.get("id") != "stn"]
    man.append({
        "id": "stn", "ko": "시상하핵 STN (배측·근사)",
        "en": "Subthalamic Nucleus (dorsal, approx.)",
        "file": "stn.npy", "bilateral": True,
        "vox_L": int(mask[0].sum()), "vox_R": int(mask[1].sum()),
        "center_L": [round(float(x), 1) for x in cl],
        "center_R": [round(float(x), 1) for x in cr],
        "vol_L_cm3": round(float(vvol[mask[0]].sum() / 1000), 3),
        "vol_R_cm3": round(float(vvol[mask[1]].sum() / 1000), 3),
        "source": "AC-PC stereotactic approximation (MIDA has no STN segmentation). "
                  f"MCP + lateral {LAT} / posterior {POST} / inferior {INF} mm, "
                  f"sphere of radius {r} mm. AC and PC are the centres of the MIDA Commissura "
                  "meshes (AC-PC distance 27.1 mm, verified).",
        "approx": True,
        "blabel": None,
    })
    man.sort(key=lambda e: e["id"])
    json.dump(man, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  manifest updated - this appears in the GUI target list as STN (L/R)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
