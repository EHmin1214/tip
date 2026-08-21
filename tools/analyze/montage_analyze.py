# -*- coding: utf-8 -*-
"""montage_analyze.py — turn a Sim4Life montage solve into numbers and images

Reads the **full-grid** result left by `s4l_montage.solve_project` and produces the TI
envelope, per-tissue statistics and slice images. It needs no h5py, so it runs in the `tip`
environment (h5py exists only in the Sim4Life Python and matplotlib only in `tip` — that is
why the pipeline is split in two).

What is new here
----------------
Our leadfield holds only the **1,907,678 brain voxels, 18% of the grid**. This is the first
place scalp, skull, CSF, eye and neck appear — and **safety numbers such as scalp current
density exist nowhere else.**

Current convention
------------------
The simulation was solved at 1 V Dirichlet. The real field of channel k is
`E_1V * (i_k / I_inj,k)`, the same convention as the leadfield. `i_k` comes from
`<smash>_montage.json` and `I_inj` from `inj.json`.

⚠ **Do not apply `LEADFIELD_AMP_FIX` again here.** That constant corrects an older path which
used `El. Loss Density` as the current. The `I_inj` used here integrates sigma|E|^2 dV directly,
so it is already correct (see the confirmed factor of 2.0 in the scale audit).

Usage: python montage_analyze.py <result-dir> <montage.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")
from tip.config import inputs as IN, LEADFIELD_DIR as _LFDIR   # input-file resolver
from tip import config as C          # read C.<name> late — the head can change

DD = INPUTS

#  IT'IS LF 4.x conductivity -> tissue name, restricted to what the safety numbers need.
#  Both languages are emitted so the UI can pick one; shipping a single language leaves this
#  table in the wrong language whenever the UI is switched.
SIGMA_TISSUE = [
    (0.1483, "Skin (epidermis/dermis)", "피부(Epidermis/Dermis)"),
    (0.0776, "Subcutaneous fat", "피하지방"),
    (0.0063, "Skull, compact bone", "두개골 치밀골"),
    (0.0805, "Skull, diploe", "두개골 판사이층"),
    (1.8790, "CSF", "CSF"),
    (0.4191, "Grey matter group", "회백질 계열"),
    (0.3480, "White matter group", "백질 계열"),
    (0.4610, "Muscle", "근육"),
    (2.1650, "Vitreous humour", "안구 유리체"),
]


def ti_envelope(E1, E2):
    """Isotropic TI envelope (Grossman closed form) — the orientation-independent upper bound.
    Processed in slices along an axis to bound memory: the full grid is 10.7 M cells."""
    out = np.empty(E1.shape[:3], np.float32)
    for i in range(E1.shape[0]):
        a, b = E1[i].astype(np.float64), E2[i].astype(np.float64)
        na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
        sw = nb > na
        A = np.where(sw[..., None], b, a); B = np.where(sw[..., None], a, b)
        nA = np.where(sw, nb, na); nB = np.where(sw, na, nb)
        dot = (A * B).sum(-1)
        B = np.where((dot < 0)[..., None], -B, B)
        cosa = np.abs(dot) / np.maximum(nA * nB, 1e-30)
        cross = np.linalg.norm(np.cross(A, B), axis=-1)
        diff = np.linalg.norm(A - B, axis=-1)
        out[i] = (2.0 * np.where(nB <= nA * cosa, nB,
                                 cross / np.maximum(diff, 1e-30))).astype(np.float32)
    return out


def channels(meta):
    """Channel names in order, and how they combine into envelopes.

    A montage is N electrode pairs, one Sim4Life simulation each. `combine` groups them:
    classic TI is one group of two, dual TI is two groups of two, and the total envelope is
    the **sum over groups** of each group's Tmax — the rule `optimize_dual_ti` scores with
    (`ct = etA + etB`).
    `compose` says how the groups are put together:
      "sum"     — classic and dual TI: the systems run at once, so their envelopes add.
      "timeavg" — time multiplexing: only one slot is on at a time, so the neuron sees the
                  duty-weighted average **per direction**, maximised over directions
                  afterwards. ⚠ Not the weighted sum of each slot's Tmax: that adds maxima
                  taken along a different direction in every slot and overestimates.
    ⚠ Older jobs and cache entries have none of these keys, so fall back to the two-channel
    classic shape rather than failing on them.
    """
    m = meta.get("montage", {})
    cur = meta.get("currents_mA", {})
    names = [k for k in cur if k != "itotal"]
    names.sort(key=lambda k: int(k[2:]) if k[2:].isdigit() else 0)
    if not names:
        names = ["ch1", "ch2"]
    comb = [list(g) for g in (m.get("combine") or [names[:2]])]
    compose = m.get("compose") or "sum"
    duties = [float(w) for w in (m.get("duties") or [1.0 / len(comb)] * len(comb))]
    return names, comb, compose, duties


#  Directions sampled for the time-multiplexed envelope. It has no closed form, and the
#  sampling is **one-sided**: too few directions understate the envelope. Measured against a
#  dense m=2048 reference on a real rat montage: m=64 puts M1 3.43% low, m=256 1.39% low,
#  m=1024 0.26% low. 256 is what `gui/app.py` uses for the leadfield side, and both sides of
#  the comparison must sample identically or the ratio would carry the difference.
TIMEAVG_M = 256


def compose_env(E, meta):
    """The montage's envelope from its per-channel fields, following `compose`."""
    from tip import ti as _ti
    _n, comb, compose, duties = channels(meta)
    if compose == "timeavg":
        return _ti.tmax_timeavg([(E[g[0]], E[g[1]]) for g in comb], duties, m=TIMEAVG_M)
    if compose != "sum":
        raise ValueError("unknown compose rule %r" % compose)
    out = None
    for g in comb:
        e = ti_envelope(E[g[0]], E[g[1]])
        out = e if out is None else out + e
    return out


def load(out_dir, meta_path):
    meta = json.load(open(meta_path, encoding="utf-8"))
    inj = json.load(open(os.path.join(out_dir, "inj.json"), encoding="utf-8"))
    cur = meta["currents_mA"]
    names = channels(meta)[0]
    E = {}
    for ch in names:
        e = np.load(os.path.join(out_dir, f"{ch}_E1V.npy"), mmap_mode="r")
        scale = (cur[ch] * 1e-3) / inj[ch]          # 1 V solution -> the real current
        E[ch] = (np.asarray(e) * np.float32(scale))
    sig = np.load(os.path.join(out_dir, "sigma.npy"), mmap_mode="r")
    ax = np.load(os.path.join(out_dir, "axes.npz"))
    return E, np.asarray(sig), ax, meta, inj


def analyze(out_dir, meta_path, target_mask=None, verbose=True):
    E, sig, ax, meta, inj = load(out_dir, meta_path)
    env = compose_env(E, meta)
    #  ⚠ Current density stays **per channel**, and the safety number below is the maximum
    #  over channels. That is right for every mode including time multiplexing: only one slot
    #  conducts at a time, so the skin sees each slot's full current, not the duty-weighted
    #  average. Time-averaging here would understate the peak by the duty factor.
    Jmag = {ch: np.linalg.norm(E[ch], axis=-1) * sig for ch in E}   # A/m²

    res = {"montage": meta["montage"], "currents_mA": meta["currents_mA"],
           "inj_mA_at_1V": {k: v * 1e3 for k, v in inj.items()},
           "head_resistance_ohm": {k: 1.0 / v for k, v in inj.items()},
           "grid": list(env.shape), "tissues": [], "safety": {}}

    # ── per tissue, identified by conductivity ──
    for s, name, name_ko in SIGMA_TISSUE:
        m = np.isclose(sig, s, rtol=1e-3)
        n = int(m.sum())
        if n < 100:
            continue
        e = env[m]
        res["tissues"].append({
            "tissue": name, "tissue_ko": name_ko, "sigma": s, "cells": n,
            "env_median": float(np.median(e)), "env_p95": float(np.percentile(e, 95)),
            "env_max": float(e.max()),
            "J_max": float(max(Jmag[c][m].max() for c in Jmag)),
        })

    # ── safety numbers — the part that exists only here ──
    skin = np.isclose(sig, 0.1483, rtol=1e-3)
    if skin.any():
        res["safety"]["skin_J_max_A_per_m2"] = float(max(Jmag[c][skin].max() for c in Jmag))
        res["safety"]["skin_E_max_V_per_m"] = float(max(
            (np.linalg.norm(E[c], axis=-1)[skin]).max() for c in E))
    res["safety"]["note"] = ("Scalp current density is invisible to the leadfield, which holds "
                             "brain voxels only. It comes from the whole-head grid.")

    # ── target, if one was given ──
    if target_mask is not None:
        bm = np.load(IN(C.BMASK_FILE)).astype(np.int64)
        idx = np.where(target_mask)[0]
        ev = env[bm[idx, 0], bm[idx, 1], bm[idx, 2]]
        res["target"] = {"cells": int(len(idx)), "env_median": float(np.median(ev)),
                         "env_p95": float(np.percentile(ev, 95)), "env_max": float(ev.max())}

    # ── slice images ──
    res["slices"] = render_slices(env, sig, ax, out_dir)

    json.dump(res, open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if verbose:
        print(f"[analyze] grid {env.shape} · {len(res['tissues'])} tissues")
        for t in res["tissues"]:
            print(f"  {t['tissue']:<26} sigma={t['sigma']:<7} cells {t['cells']:>9,} · "
                  f"env median {t['env_median']:.4f} p95 {t['env_p95']:.4f} · "
                  f"J max {t['J_max']:.3f} A/m2")
        if "target" in res:
            print(f"  target: median {res['target']['env_median']:.4f} V/m")
        print(f"  skin J max {res['safety'].get('skin_J_max_A_per_m2', float('nan')):.3f} A/m2")
    return res


def render_slices(env, sig, ax, out_dir, pct=99.0):
    """Three slices — sagittal, coronal, axial. Envelope as colour, tissue boundaries as
    contours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny, nz = env.shape
    vmax = float(np.percentile(env[env > 0], pct)) if (env > 0).any() else 1.0
    cuts = [("sagittal", env[nx // 2], sig[nx // 2]),
            ("coronal", env[:, ny // 2], sig[:, ny // 2]),
            ("axial", env[:, :, nz // 2], sig[:, :, nz // 2])]
    out = []
    for name, sl, ss in cuts:
        fig, a = plt.subplots(figsize=(4.6, 4.6), dpi=110)
        a.imshow(sl.T, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                 interpolation="nearest")
        a.contour(ss.T > 0.02, levels=[0.5], colors="w", linewidths=0.4, alpha=0.5)
        a.set_title(f"{name}  TI envelope [V/m]", fontsize=9)   # ASCII: the default font has
                                                                # no Korean glyphs
        a.axis("off")
        fig.colorbar(a.images[0], ax=a, fraction=0.045)
        p = os.path.join(out_dir, f"slice_{name}.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        out.append({"plane": name, "file": os.path.basename(p), "vmax": vmax})
    return out


# ─────────────── target, metrics, and where the stimulation lands ───────────────
#  Everything below **depends on the target**, so it uses brain-voxel E rather than the full
#  grid (`*_Ebrain.npy`, 22.9 MB per channel). That is what lets a different target be
#  evaluated **without re-solving**.

#  brain label -> (English, Korean), **per head model**. Both languages, for the same reason
#  as SIGMA_TISSUE above.
#
#  ⚠ These numbers are the phantom's, not a standard: 75 means cortex in MIDA and nothing at
#  all in NeuroRat. Hard-coding the MIDA table made the whole "where does it land" breakdown
#  come out empty for another head — empty, not wrong, which is the kind of failure nobody
#  notices in a report.
BLABEL_MIDA = {131: ("White matter", "백질"), 75: ("Grey matter (cortex)", "회백질(피질)"),
               156: ("Thalamus", "시상"), 17: ("Putamen", "조가비핵"),
               90: ("Caudate", "미상핵"), 81: ("Hippocampus", "해마"),
               21: ("Amygdala", "편도체"), 48: ("Accumbens", "측좌핵"),
               110: ("Hypothalamus", "시상하부")}
#  NeuroRat: the ten structures tools/s4l/rat_extract.py writes, keyed by its RAT_LABELS.
BLABEL_RAT = {1: ("Cerebral cortex", "대뇌피질"), 2: ("Rest of brain", "기타 뇌"),
              3: ("Hippocampus", "해마"), 4: ("Thalamus", "시상"),
              5: ("Caudoputamen", "선조체"), 6: ("Midbrain", "중뇌"),
              7: ("Pons", "교뇌"), 8: ("Medulla", "연수"),
              9: ("Cerebellum", "소뇌"), 10: ("Olfactory bulb", "후각구")}


def blabel_table():
    """The structure table for the head currently selected by `TIP_MODEL`.

    Falls back to whatever `labels_<model>.json` the extractor wrote, so a head added later
    still produces a breakdown instead of a silent blank.
    """
    if C.MODEL_NAME == "human":
        return BLABEL_MIDA
    if C.MODEL_NAME == "rat":
        return BLABEL_RAT
    p = IN("labels_%s.json" % C.MODEL_NAME)
    if os.path.exists(p):
        return {int(v): (k.replace("_", " "), k.replace("_", " "))
                for k, v in json.load(open(p, encoding="utf-8")).items()}
    return {}


def _brain_env(out_dir):
    """Brain-voxel TI envelope (N,) — **the same voxel set and the same definition** the
    leadfield metrics use."""
    meta = json.load(open(os.path.join(out_dir, "montage.json"), encoding="utf-8"))
    inj = json.load(open(os.path.join(out_dir, "inj.json"), encoding="utf-8"))
    cur = meta["currents_mA"]
    names, _comb, compose, _duties = channels(meta)
    E = {}
    for ch in names:
        p = os.path.join(out_dir, f"{ch}_Ebrain.npy")
        if not os.path.exists(p):
            return None
        E[ch] = np.load(p) * np.float32((cur[ch] * 1e-3) / inj[ch])
    if compose == "timeavg":
        #  `tmax_timeavg` contracts the last axis, so flat (N,3) is exactly what it wants.
        return compose_env(E, meta)
    #  `ti_envelope` takes (X,Y,Z,3) and slices **along axis 0** to bound memory.
    #  ⚠ Putting the voxels on axis 0 would loop 1.9 M times and effectively hang — put them
    #    on axis 1.
    E4 = {k: v.reshape(1, -1, 1, 3) for k, v in E.items()}
    return compose_env(E4, meta)[0, :, 0]


def target_report(out_dir, target_npz, lf_metrics=None, thr_mode="target_median"):
    """Target statistics, M1/M2/M3 from the Sim4Life solution, and **which structures actually
    receive stimulation**.

    target_npz : npz holding `target_idx` and `off_idx` (brain-voxel row indices), `name` and
                 `off_def`. It is produced from **the same Target object** the GUI uses — two
                 copies of target-resolution logic would let the metrics diverge silently.
    lf_metrics : the **leadfield** metrics for the same montage, shown side by side as a check.

    ⚠ These are only meaningful if M1/M2/M3 match the definitions in `tip/metrics.py`:
       M1 = median target envelope · M2 = (RMS_t / RMS_o)^2 · M3 = % of off volume above the
       target median
    """
    env = _brain_env(out_dir)
    if env is None:
        return {"error": "no brain-voxel E (*_Ebrain.npy) - older jobs need a re-solve"}
    tz = np.load(target_npz, allow_pickle=True)
    ti = tz["target_idx"].astype(np.int64)
    off = tz["off_idx"].astype(np.int64)
    off = off[~np.isin(off, ti)]
    tt, to = env[ti], env[off]

    m1 = float(np.median(tt))
    m2 = float((np.sqrt(np.mean(tt ** 2)) / max(np.sqrt(np.mean(to ** 2)), 1e-30)) ** 2)
    m3 = float(100.0 * (to > m1).mean())          # fraction of off voxels above the target median

    q = np.percentile(tt, [10, 50, 90, 95])
    res = {
        "name": str(tz["name"]) if "name" in tz else "target",
        "off_def": str(tz["off_def"]) if "off_def" in tz else "?",
        "n_target": int(len(ti)), "n_off": int(len(off)),
        "env": {"median": m1, "mean": float(tt.mean()), "max": float(tt.max()),
                "p10": float(q[0]), "p90": float(q[2]), "p95": float(q[3])},
        "metrics_s4l": {"M1": m1, "M2": m2, "M3": m3},
        "off_max": float(to.max()),
    }

    #  ★Side by side with the leadfield metrics — how closely the two paths agree is the point
    if lf_metrics:
        cmp = {}
        for k in ("M1", "M2", "M3"):
            a, b = lf_metrics.get(k), res["metrics_s4l"][k]
            if a is None:
                continue
            cmp[k] = {"leadfield": float(a), "sim4life": float(b),
                      "ratio": (float(b) / float(a)) if a else None}
        res["compare"] = cmp

    # ── where the stimulation actually lands ──────────────────────
    #  The threshold is the median target envelope, i.e. the same criterion M3 uses — so this
    #  table is **M3 decomposed by structure**.
    #  M3 says only *how much* leaks; this says *where* it leaks.
    thr = m1
    lab = np.load(DD_BLABEL())
    inside = np.zeros(len(env), bool); inside[ti] = True
    rows = []
    for lv, (nm, nm_ko) in blabel_table().items():
        m = (lab == lv)
        n = int(m.sum())
        if n == 0:
            continue
        e = env[m]
        hot = m & (env > thr)
        rows.append({
            "region": nm, "region_ko": nm_ko, "label": int(lv), "voxels": n,
            "in_target": int((m & inside).sum()),
            "env_median": float(np.median(e)), "env_p95": float(np.percentile(e, 95)),
            "env_max": float(e.max()),
            "over_thr_voxels": int(hot.sum()),
            "over_thr_pct": round(float(100.0 * hot.sum() / n), 2),
        })
    rows.sort(key=lambda r: -r["over_thr_pct"])
    res["threshold"] = {"value": thr, "mode": thr_mode,
                        "note": "median target envelope - the same criterion as M3, so this is "
                                "M3 broken down by structure"}
    res["regions"] = rows

    json.dump(res, open(os.path.join(out_dir, "target.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return res


def DD_BLABEL():
    return IN(C.BLABEL_FILE)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    a = sys.argv[1:]
    #  `--target-only` skips the per-tissue statistics and slices and computes target metrics
    #  only. Used on a cache hit, where the full-grid E (128 MB per channel) is already gone
    #  and only the brain-voxel E remains.
    if "--target-only" not in a:
        analyze(a[0], a[1])
    if "--target" in a:
        tnpz = a[a.index("--target") + 1]
        lfm = None
        if "--lfmetrics" in a:
            lfm = json.loads(a[a.index("--lfmetrics") + 1])
        r = target_report(a[0], tnpz, lf_metrics=lfm)
        if "error" in r:
            print("[target] " + r["error"])
        else:
            print(f"[target] {r['name']} · voxels {r['n_target']:,} · off {r['n_off']:,} "
                  f"({r['off_def']})")
            m = r["metrics_s4l"]
            print(f"  Sim4Life metrics  M1 {m['M1']:.4f}  M2 {m['M2']:.3f}  M3 {m['M3']:.2f}%")
            for k, v in (r.get("compare") or {}).items():
                print(f"    {k}: leadfield {v['leadfield']:.4f} vs S4L {v['sim4life']:.4f} "
                      f"= ×{v['ratio']:.3f}")
            print(f"  structures above {r['threshold']['value']:.4f} V/m (top 5):")
            for x in r["regions"][:5]:
                print(f"    {x['region']:<12} {x['over_thr_pct']:>6.2f}% "
                      f"({x['over_thr_voxels']:,}/{x['voxels']:,} voxels)")
    print("=== END · analysis complete")
