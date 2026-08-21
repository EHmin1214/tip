# -*- coding: utf-8 -*-
"""
app.py — TI Planner backend (stdlib http.server, no dependencies)
========================================================
  GET  /                 → index.html
  GET  /api/init         → electrodes, target list, 3D brain point cloud
  POST /api/optimize     → start a job, returns {job} (async)
  GET  /api/progress?job → {pct, stage, done, result|error}
  POST /api/electrode/solve → ★solve an electrode in Sim4Life and add it to the leadfield
                              (about 2 min per electrode)
  GET  /api/jobs         → jobs persisted on disk (they survive closing the browser)

Run:   run_gui.bat  (or python src/tip/gui/app.py)
"""
import os
# Single-threaded numpy/BLAS — avoids oversubscription in a threaded server.
# Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, json, time, threading, uuid, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # repo/src/tip/gui
SRC = os.path.dirname(os.path.dirname(HERE))               # repo/src
ROOT = os.path.dirname(SRC)                                # repo/
sys.path.insert(0, SRC)   # so a double-click works without `pip install -e .`
from tip import LeadField, Target                       # noqa: E402
from tip.optimize.classic import optimize_classic, channel_currents  # noqa: E402
from tip.optimize.nsga import optimize_nsga             # noqa: E402
from tip.optimize.multichannel import optimize_gevd, optimize_distributed  # noqa: E402
#  ⚠ `dual_budget()`, not the `DUAL_BUDGET` constant. The constant is ITOTAL/2 and forces a
#  **total-current** rule; under the GUI's `max_channel` rule `dual_budget()` returns None so
#  each system pins its larger channel, exactly as `_cfields` does when it prices the
#  leadfield metrics. Mixing the two priced the two sides of the comparison differently —
#  measured 2.8-4.0x apart on MIDA (the rat happened to agree only because its per-electrode
#  cap bound first, which is what hid this).
from tip.optimize.dualti import (optimize_dual_ti, combined_env,  # noqa: E402
                                 dual_budget)
from tip.optimize.selective import optimize_selective  # noqa: E402  (beta: selectivity)
from tip.optimize.huang import optimize_huang  # noqa: E402  (beta: Huang 2020 focality)
from tip.optimize.timemux import optimize_timemux  # noqa: E402  (time multiplexing)
from tip.plan import _protocol                          # noqa: E402
from tip import protocol as PROTO                       # noqa: E402  current rule
from tip.report import _montage_fields                  # noqa: E402
from tip import config as C                              # noqa: E402  single source of truth for paths
from tip import models as MODELS                         # noqa: E402  the head-model registry
from tip import metrics as METRICS                       # noqa: E402  M1/M2/M3 are defined in exactly one place
from tip import ti as TI                                 # noqa: E402
from tip.orch import JobStore                            # noqa: E402
from tip.orch import s4l as ORCH_S4L                     # noqa: E402

#  Override with TIP_GUI_PORT to run a second instance (testing a change while the
#  everyday one keeps running — on Windows both bind and the first one wins,
#  which silently serves stale code).
PORT = int(os.environ.get("TIP_GUI_PORT") or 8765)

# ── ★Head model, chosen at runtime (2026-08-18) ──────────────────────────
# The GUI used to be MIDA-only: `hipmask1010`, the grey-matter label 75, the deep-target
# coordinates and the mirror about x = -27 were all baked in at import. With a second head
# (NeuroRat) that silently produced human structures at human coordinates, sitting outside
# the rat brain. Everything derived from the head now lives in `load_model()`, which rebinds
# the module globals below, and the browser can switch heads with `POST /api/model`.
#
# ⚠ Two rules that make the switch safe:
#   · read `C.<constant>` late — a `from tip.config import ICH_MAX` would freeze the human
#     value, which is why that import was removed;
#   · anything built from the old head is thrown away here, not reused. Leadfields are kept
#     per model in `_LF_CACHE` because reloading one costs seconds; masks, targets, clouds
#     and samples are all rebuilt.
MODEL_LABELS = {"human": "MIDA (사람)", "rat": "NeuroRat (쥐)", "mouse": "IT'IS mouse (보류)"}


def available_models():
    """Heads the GUI can actually serve — a leadfield has to exist on disk.

    The mouse entry in `models.py` is parked with no geometry, so it never shows up.
    """
    out = []
    for name, m in MODELS.REGISTRY.items():
        d = os.path.join(C.LEADFIELD_ROOT, m.leadfield_dir)
        n = len(glob.glob(os.path.join(d, "*.npy"))) if os.path.isdir(d) else 0
        if n == 0 and m.leadfield_style == "legacy":
            n = len(glob.glob(os.path.join(C.LEADFIELD_DIR, "M*.npy")))
        if n:
            out.append(dict(id=name, label=MODEL_LABELS.get(name, m.label),
                            species=m.species, n_elec=m.n_elec, ref=m.ref_elec))
    return out


_LF_CACHE = {}
MODEL_NAME = None
#  ★The stimulation current in force. Set from the model's established value when a
#  head is loaded (MIDA 1.0 mA, rat None) and overwritten by any request that carries
#  one. It has to live here because **every endpoint that prices a montage needs it**,
#  not just /api/optimize: /api/field, /api/efield, /api/pointfield, /api/affield,
#  /api/resolution and /api/montage/send all end up in `channel_currents`. Wiring only
#  the optimiser left the rest crashing with `NoneType / float` on the rat, and the
#  browser showed that as a flat "rejected".
CUR_ICH = None


def _mida_targets():
    """The MIDA target list: two hard-coded masks, the coordinate presets, and whatever the
    `masks/mida` manifest adds. Unchanged — every number in here is MIDA's."""
    global HIP, THAL, HAX, MMASK

    HIP = np.load(C.inputs("hipmask1010.npy"))
    try:
        THAL = np.load(C.inputs("thalamus_mask.npy"))
    except Exception:
        THAL = None
    HAX = np.load(C.inputs("hipaxes1010.npz"))

    def _mirror(c):
        return [-54.0 - c[0], c[1], c[2]]

    deep = {  # coordinate-based targets (cortical sites with no mask): (centre, r, ko, en)
        "dlpfc": ([-53, 320, -18], 9, "배외측전전두 DLPFC", "DLPFC"),
        "m1":    ([-58, 332, 34], 9, "일차운동피질 M1", "Primary motor (M1)"),
        "a6dl":  ([-60, 318, 40], 8, "전운동피질 A6dl", "Premotor (A6dl)"),
    }
    targets = [
        dict(id="hippo_left",  kind="mask", ko="해마 (좌)", en="Hippocampus (L)"),
        dict(id="hippo_right", kind="mask", ko="해마 (우)", en="Hippocampus (R)"),
    ]
    if THAL is not None:
        targets += [dict(id="thalamus_left",  kind="mask", ko="시상 (좌)", en="Thalamus (L)"),
                    dict(id="thalamus_right", kind="mask", ko="시상 (우)", en="Thalamus (R)")]
    for k, (c, r, ko, en) in deep.items():
        targets.append(dict(id=k + "_L", kind="sphere", center=[float(x) for x in c],
                            radius=float(r), ko=ko + " (좌)", en=en + " (L)"))
        targets.append(dict(id=k + "_R", kind="sphere", center=_mirror([float(x) for x in c]),
                            radius=float(r), ko=ko + " (우)", en=en + " (R)"))
    targets.append(dict(id="cingulate", kind="sphere", center=[-27.0, 272.0, -18.0],
                        radius=5.0, ko="슬하대상 Cg25", en="Subgenual cingulate"))

    # ---- real masks from the manifest (multi-region, extracted from Sim4Life) ----
    MMASK = {}   # base id -> (2, N) bool
    mdir = os.path.join(C.MASKS_DIR, "mida")
    mpath = os.path.join(mdir, "manifest.json")
    if os.path.exists(mpath):
        try:
            man = json.load(open(mpath, encoding="utf-8"))
        except Exception:
            man = []
        mt, ids = [], set()
        for m in man:
            if m["id"] in ("hippocampus", "thalamus"):   # already HIP/THAL — avoid duplicates
                continue
            try:
                MMASK[m["id"]] = np.load(os.path.join(mdir, m["file"]))
            except Exception:
                continue
            ids.add(m["id"])
            if m.get("bilateral", True):
                mt += [dict(id=m["id"] + "_left", kind="mask",
                            ko=m["ko"] + " (좌)", en=m["en"] + " (L)"),
                       dict(id=m["id"] + "_right", kind="mask",
                            ko=m["ko"] + " (우)", en=m["en"] + " (R)")]
            else:
                mt.append(dict(id=m["id"] + "_left", kind="mask", ko=m["ko"], en=m["en"]))
        # drop the approximate-sphere presets that a real mask now replaces
        targets = [t for t in targets
                   if not (t.get("kind") == "sphere" and t["id"].rsplit("_", 1)[0] in ids)]
        ins = 4 if THAL is not None else 2   # insert after hippocampus (and thalamus)
        targets = targets[:ins] + mt + targets[ins:]
    return targets


#  Korean/English names for the NeuroRat structures, keyed by the names that
#  tools/s4l/rat_extract.py writes into inputs/labels_rat.json.
RAT_STRUCTS = {
    "Hippocampus": ("해마", "Hippocampus"),
    "Thalamus": ("시상", "Thalamus"),
    "Caudo_putamen": ("선조체", "Caudoputamen"),
    "Cerebral_cortex": ("대뇌피질", "Cerebral cortex"),
    "Midbrain": ("중뇌", "Midbrain"),
    "Cerebellum": ("소뇌", "Cerebellum"),
    "Pons": ("교뇌", "Pons"),
    "Medulla_oblongata": ("연수", "Medulla"),
    "Olfactory_bulb": ("후각구", "Olfactory bulb"),
    "Rest_of_brain": ("기타 뇌", "Rest of brain"),
}


def _label_targets():
    """Targets straight from the model's own tissue labels, split by its midline.

    The model-agnostic path — no coordinates, no hand-written masks. Sides come from
    `MODEL.is_left_pts`, which knows about the rat's oblique midplane.

    ⚠ For the rat the split is imperfect and that is the phantom, not a bug: its segmentation
      is lopsided, so a "left" structure carries roughly 7% of the other side. Each entry
      reports its own share so the contamination is visible instead of hidden.
    """
    labels = {}
    p = C.inputs("labels_rat.json") if C.MODEL_NAME == "rat" else None
    if p and os.path.exists(p):
        labels = json.load(open(p, encoding="utf-8"))
    if not labels:
        labels = {k: v for k, v in C.MODEL.labels.items() if v is not None}
    left = C.MODEL.is_left_pts(COORDS)
    targets = []
    for nm, lab in sorted(labels.items(), key=lambda kv: kv[1]):
        sel = BL == lab
        if not sel.any():
            continue
        ko, en = RAT_STRUCTS.get(nm, (nm, nm))
        frac = float(left[sel].mean())
        for side, want, sko, sen in (("left", True, "좌", "L"), ("right", False, "우", "R")):
            idx = np.where(sel & (left == want))[0]
            if len(idx) < 50:
                continue
            share = frac if want else 1.0 - frac
            targets.append(dict(id="label:%d:%s" % (lab, side), kind="label", label=int(lab),
                                side=side, ko="%s (%s)" % (ko, sko), en="%s (%s)" % (en, sen),
                                n=int(len(idx)), share=round(100 * share, 1)))
    return targets


def load_model(name=None):
    """(Re)build every piece of state that depends on the head. Safe to call repeatedly."""
    global MODEL_NAME, LF, HIP, THAL, HAX, BL, COORDS, GMIN, GS, NEURAL, BRAINV, BRAIN_C
    global ELPOS, CLOUD, CLOUD_XYZ, OFFSAMP, TARGETS, MMASK

    global CUR_ICH
    C.use_model(name)
    MODEL_NAME = C.MODEL_NAME
    CUR_ICH = C.ICH_MAX          # None for a head with no established current
    print("[TI Planner] loading model %s..." % MODEL_NAME, flush=True)

    if MODEL_NAME not in _LF_CACHE:
        _LF_CACHE[MODEL_NAME] = LeadField()
    LF = _LF_CACHE[MODEL_NAME]

    HIP = THAL = HAX = None
    MMASK = {}
    BL = np.load(C.inputs(C.BLABEL_FILE))
    COORDS = LF.coords()
    GMIN = COORDS.min(0)                       # grid origin and spacing (target contours)
    GS = np.array([np.median(np.diff(np.unique(COORDS[:, k]))) for k in range(3)])
    GS[GS < 1e-6] = 1.0
    #  Off-target pool for the 3D view and the rich metrics. `NEURAL_LABELS` is the model's
    #  own answer — grey matter (label 75, the tip.lite convention) for MIDA, the cortex for
    #  the rat. An empty result would make every off-target metric NaN in silence, so fall
    #  back to the whole mask and say so out loud.
    NEURAL = np.where(np.isin(BL, C.NEURAL_LABELS))[0] if C.NEURAL_LABELS else np.array([], int)
    if len(NEURAL) == 0:
        print("[TI Planner] model %r has no neural labels — using the whole brain mask as "
              "the off-target pool" % MODEL_NAME, flush=True)
        NEURAL = np.arange(len(BL))
    BRAINV = np.arange(len(BL))                # every brain voxel (3D view, deep targets)
    BRAIN_C = COORDS[NEURAL].mean(0)
    ELPOS = np.array([LF.pos[e] for e in LF.names])
    rng = np.random.default_rng(0)
    #  3D render cloud (~117 points/cm3 on the human head — keeps tight foci visible)
    CLOUD = np.sort(rng.choice(BRAINV, min(100000, len(BRAINV)), replace=False))
    CLOUD_XYZ = COORDS[CLOUD]
    OFFSAMP = np.sort(rng.choice(NEURAL, min(15000, len(NEURAL)), replace=False))

    TARGETS = _mida_targets() if MODEL_NAME == "human" else _label_targets()
    TARGETS.append(dict(id="brain_center", kind="sphere",
                        center=[float(x) for x in BRAIN_C],
                        radius=10.0 if MODEL_NAME == "human" else 2.0,
                        ko="뇌 중심", en="Brain center"))
    TARGETS.append(dict(id="custom", kind="custom", ko="임의 좌표…", en="Custom coords…"))
    return MODEL_NAME


load_model()
def _ready_line():
    return (f"[TI Planner] {MODEL_NAME} ready · {len(LF.names)} electrodes · "
            f"{len(TARGETS)} targets · {len(MMASK)} masks · {len(BRAINV)} brain voxels")


print(_ready_line(), flush=True)


def _label_idx(tid):
    """Row indices for a `label:<n>:<side>` target — the model-agnostic target kind.

    Sides come from `MODEL.is_left_pts`, so this works for MIDA's axis-aligned midline and
    for the rat's oblique plane alike.
    """
    _, lab, side = tid.split(":")
    sel = BL == int(lab)
    if side in ("left", "right"):
        sel &= (C.MODEL.is_left_pts(COORDS) == (side == "left"))
    return np.where(sel)[0]


def _target_center(tid):
    if tid.startswith("label:") and tid.count(":") == 2:
        return COORDS[_label_idx(tid)].mean(0)
    if HIP is not None:
        if tid == "hippo_left":  return COORDS[np.where(HIP[0])[0]].mean(0)
        if tid == "hippo_right": return COORDS[np.where(HIP[1])[0]].mean(0)
    if THAL is not None:
        if tid == "thalamus_left":  return COORDS[np.where(THAL[0])[0]].mean(0)
        if tid == "thalamus_right": return COORDS[np.where(THAL[1])[0]].mean(0)
    _b = tid.rsplit("_", 1)[0]
    if _b in MMASK:
        return COORDS[np.where(MMASK[_b][0 if tid.endswith("_left") else 1])[0]].mean(0)
    return BRAIN_C


def auto_direction(target, tid):
    """Derive the stimulation direction from the target (hippocampus → its axis,
    otherwise → radial from the scalp).

    ⚠ The hippocampal axes are MIDA's, measured on that phantom, so they only apply when the
    human head is loaded. Another model falls back to the radial direction — there is no
    equivalent axis file for it, and reusing MIDA's would be a silently wrong vector.
    """
    if HAX is not None:
        if tid == "hippo_left":  return HAX["nL"]
        if tid == "hippo_right": return HAX["nR"]
        if tid == "label:81":    return HAX["nL"]
    tc = COORDS[target.target_idx].mean(0)
    ne = ELPOS[np.argmin(np.linalg.norm(ELPOS - tc, axis=1))]
    d = tc - ne
    return d / (np.linalg.norm(d) + 1e-9)


def _contour_pts(target_idx, cap=700):
    """Surface (boundary) voxels of a target mask — a voxel is on the boundary if any of its
    six neighbours lies outside. Only the outermost shell, not the filled interior, which is
    what the contour display needs."""
    C = COORDS[np.asarray(target_idx)]
    key = np.round((C - GMIN) / GS).astype(np.int64)
    S = set(map(tuple, key.tolist()))
    surf = np.zeros(len(key), bool)
    for i in range(len(key)):
        k0, k1, k2 = key[i]
        if ((k0 - 1, k1, k2) not in S or (k0 + 1, k1, k2) not in S or
            (k0, k1 - 1, k2) not in S or (k0, k1 + 1, k2) not in S or
            (k0, k1, k2 - 1) not in S or (k0, k1, k2 + 1) not in S):
            surf[i] = True
    idx = np.where(surf)[0]
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(0).choice(idx, cap, replace=False))
    return C[idx]


def build_target(spec):
    kw = dict(off_subsample=22000)
    if spec.get("kind") == "sphere":
        return Target.from_sphere(LF, spec["center"], float(spec["radius"]), name="표적", **kw)
    tid = spec.get("id", "")
    if tid.startswith("label:") and tid.count(":") == 2:
        return Target(LF, _label_idx(tid), name=spec.get("en") or tid, **kw)
    if HIP is not None:
        if tid == "hippo_left":  return Target.from_mask(LF, HIP[0], name="Hippocampus L", **kw)
        if tid == "hippo_right": return Target.from_mask(LF, HIP[1], name="Hippocampus R", **kw)
    if THAL is not None:
        if tid == "thalamus_left":  return Target.from_mask(LF, THAL[0], name="Thalamus L", **kw)
        if tid == "thalamus_right": return Target.from_mask(LF, THAL[1], name="Thalamus R", **kw)
    _b = tid.rsplit("_", 1)[0]
    if _b in MMASK:
        return Target.from_mask(LF, MMASK[_b][0 if tid.endswith("_left") else 1], name=tid, **kw)
    raise ValueError(f"unknown target {tid!r} for model {MODEL_NAME!r}")


def build_dir(spec):
    if spec in (None, "", "none"): return None
    #  MIDA-only axes — see auto_direction. Another head has no equivalent file.
    if spec == "hippo_nL": return HAX["nL"] if HAX is not None else None
    if spec == "hippo_nR": return HAX["nR"] if HAX is not None else None
    if isinstance(spec, (list, tuple)): return np.asarray(spec, float)
    return None


def _used(best):
    used = {}
    if best.get("dual"):     # 4-channel: system A = ch0/1 (reds), system B = ch2/3 (blues);
                             # ITOTAL/2 per system
        for sk, base in (("systemA", 0), ("systemB", 2)):
            s = best[sk]; a, b = s["ch1"]; c, d = s["ch2"]; r = s.get("ratio", 1.0)
            #  ⚠ `dual_budget()` — see the import note. With the `DUAL_BUDGET` constant this
            #  panel reported currents 2.8-4.0x below what the optimiser actually used.
            i1, i2 = channel_currents(r, dual_budget())
            used[a] = {"ch": base, "I": round(i1, 3)}; used[b] = {"ch": base, "I": round(-i1, 3)}
            used[c] = {"ch": base + 1, "I": round(i2, 3)}; used[d] = {"ch": base + 1, "I": round(-i2, 3)}
        return used
    if best.get("timemux"):  # time-mux: an electrode may recur across slots, so show the
                             # duty-weighted mean current
        acc = {}
        for si, s in enumerate(best["slots"]):
            a, b = s["ch1"]; c, d = s["ch2"]
            i1, i2 = channel_currents(s.get("ratio", 1.0)); w = s.get("duty", 0.0)
            for e, I, ch in ((a, i1, 0), (b, -i1, 0), (c, i2, 1), (d, -i2, 1)):
                r = acc.setdefault(e, {"I": 0.0, "ch": ch, "slots": []})
                r["I"] += w * I; r["slots"].append(si + 1)
        for e, r in acc.items():
            used[e] = {"ch": r["ch"], "I": round(r["I"], 3), "slots": r["slots"]}
        return used
    if "currents" in best:
        # distributed: every electrode carries current on both carriers, so the 3D view
        # colours and sizes it by the dominant channel (larger |I|)
        c0 = best["currents"]["ch0"]; c1 = best["currents"]["ch1"]
        for e in set(c0) | set(c1):
            i0 = float(c0.get(e, 0.0)); i1 = float(c1.get(e, 0.0))
            ch = 0 if abs(i0) >= abs(i1) else 1
            used[e] = {"ch": ch, "I": round(i0 if ch == 0 else i1, 3), "I2": round(i1 if ch == 0 else i0, 3)}
    else:
        a, b = best["ch1"]; c, d = best["ch2"]; r = best.get("ratio", 1.0)
        i1, i2 = channel_currents(r)   # normalised so total injected current = ITOTAL
        used[a] = {"ch": 0, "I": round(i1, 3)}; used[b] = {"ch": 0, "I": round(-i1, 3)}
        used[c] = {"ch": 1, "I": round(i2, 3)}; used[d] = {"ch": 1, "I": round(-i2, 3)}
    return used


def _freq_check(f1, f2, freqs=None):
    """Sanity-check the carrier frequencies.

    **The field pattern does not depend on frequency** (in tissue at kHz, sigma >> omega*eps,
    so the quasi-static approximation holds and reusing one leadfield is justified).
    **But Δf decides whether an envelope exists at all**: the envelope formula
    2·min(|n·E1|, |n·E2|) assumes the relative phase sweeps 0→2π over one beat period. With
    Δf = 0 the phase is fixed, the amplitude is constant, and there is **no envelope** — that
    is not TI but a coherent vector sum, i.e. HD-TES."""
    pairs = [(freqs[0], freqs[1]), (freqs[2], freqs[3])] if freqs else [(f1, f2)]
    lvl, msgs = "ok", []
    for a, b in pairs:
        df = abs(float(b) - float(a)); fc = 0.5 * (float(a) + float(b))
        if df < 1e-6:
            lvl = "error"
            msgs.append("Δf=0 (두 채널 동일 주파수) — 맥놀이가 없어 TI 엔벨로프가 형성되지 않습니다. "
                        "두 필드가 결맞게 더해진 단일 주파수 자극(=HD-TES)일 뿐이며, "
                        "표시된 엔벨로프·지표는 물리적으로 무효입니다.")
        elif df < 1.0:
            lvl = "error" if lvl == "error" else "warn"
            msgs.append(f"Δf={df:.3g}Hz — 변조 주기 {1.0/df:.1f}s로 너무 느립니다(신경 반응 부적합).")
        elif df > 0.2 * fc:
            lvl = "error" if lvl == "error" else "warn"
            msgs.append(f"Δf={df:.0f}Hz가 반송파 {fc:.0f}Hz 대비 큽니다 — 엔벨로프 근사(Δf≪f)가 깨집니다.")
        if fc < 500:
            lvl = "error" if lvl == "error" else "warn"
            msgs.append(f"반송파 {fc:.0f}Hz — TI는 보통 1–5kHz입니다(저주파는 반송파 직접 자극 위험).")
    return dict(level=lvl, msg=" · ".join(msgs),
                df=[round(abs(float(b) - float(a)), 3) for a, b in pairs])


def _clean(best):
    o = {}
    for k, v in best.items():
        if isinstance(v, tuple): v = list(v)
        if isinstance(v, np.floating): v = float(v)
        o[k] = v
    return o


def _slot_env(slot, nd, idx):
    E1, E2 = _montage_fields(LF, slot, idx)
    return TI.directional_env(E1, E2, np.asarray(nd, float)) if nd is not None else TI.tmax(E1, E2)


def _envelope(best, idx):
    """Montage envelope — dual: combined, time-mux: time-averaged, distributed: directional,
    classic: isotropic Tmax.

    ★Every mode must use **the same yardstick**. `optimize_timemux` always puts a `direction`
    in its result, so time-mux used to be displayed as a single-axis directional envelope.
    A directional value is by definition smaller than the isotropic Tmax (measured 0.73× on
    the left thalamus), so next to classic it made time-mux look like a bad montage — when in
    fact the two were measuring different quantities.
    Time-mux is now reported as the **isotropic equivalent**: time-average per direction
    first, then take the maximum over directions.
    """
    if best.get("dual"):
        return combined_env(LF, best, idx)
    if best.get("timemux"):                      # time average — what a neuron sees under
                                                 # fast switching
        pairs = [_montage_fields(LF, s, idx) for s in best["slots"]]
        duties = [s["duty"] for s in best["slots"]]
        return TI.tmax_timeavg(pairs, duties)
    _nd = best.get("direction")
    E1, E2 = _montage_fields(LF, best, idx)
    return TI.directional_env(E1, E2, np.asarray(_nd, float)) if _nd is not None else TI.tmax(E1, E2)


def _hf(best, idx):
    """Carrier (high-frequency) peak. For dual, the maximum over both systems."""
    if best.get("dual"):
        from tip.optimize.dualti import _cfields
        return np.maximum(TI.carrier_max(*_cfields(LF, best["systemA"], idx)),
                          TI.carrier_max(*_cfields(LF, best["systemB"], idx)))
    if best.get("timemux"):        # instantaneous maximum across slots — safety is judged on
                                   # the worst slot
        return np.maximum.reduce([TI.carrier_max(*_montage_fields(LF, s, idx)) for s in best["slots"]])
    E1, E2 = _montage_fields(LF, best, idx)
    return TI.carrier_max(E1, E2)


def rich_metrics(best, target):
    """Analysis metrics over the target, the off-target pool and the render cloud."""
    ti_idx = target.target_idx
    off_idx = OFFSAMP[~np.isin(OFFSAMP, ti_idx)]        # off-target = brain minus target
    tt = _envelope(best, ti_idx); to = _envelope(best, off_idx)
    hf_t = _hf(best, ti_idx); hf_o = _hf(best, off_idx)
    tc = _envelope(best, CLOUD)                          # render cloud (3D)
    tin = np.isin(CLOUD, ti_idx)
    # peak location (highest point outside the target)
    ci_off = np.where(~tin)[0]
    pk_i = ci_off[np.argmax(tc[ci_off])] if len(ci_off) else int(np.argmax(tc))
    peak_xyz = CLOUD_XYZ[pk_i]
    tcen = COORDS[ti_idx].mean(0)
    peak_dist = float(np.linalg.norm(peak_xyz - tcen))
    peak_in_target = bool(tin.any() and np.max(tc[tin]) >= (np.max(tc[ci_off]) if len(ci_off) else 0.0))
    # depth (target centre to the nearest electrode)
    depth = float(np.min(np.linalg.norm(ELPOS - tcen, axis=1)))
    # histogram (cumulative fraction above each level)
    lv = np.linspace(0, float(np.percentile(np.r_[tt, to], 99.5)) or 1e-6, 26)
    hist = dict(levels=[round(float(x), 3) for x in lv],
                target=[round(float((tt > L).mean()), 4) for L in lv],
                off=[round(float((to > L).mean()), 4) for L in lv])
    q = np.percentile(tt, [10, 50, 90])
    return dict(
        field=dict(tmax=[round(float(x), 4) for x in tc], target=tin.astype(int).tolist(),
                   tmax_max=round(float(np.percentile(tc, 99)), 3),
                   peak=[round(float(x), 1) for x in peak_xyz]),
        target_info=dict(n_vox=int(len(ti_idx)), vol_mm3=int(len(ti_idx) * 0.5 ** 3 * 1000 / 1000),
                         center=[round(float(x), 1) for x in tcen], depth_mm=round(depth, 1)),
        strength=dict(median=round(float(q[1]), 3), mean=round(float(tt.mean()), 3),
                      max=round(float(tt.max()), 3), p10=round(float(q[0]), 3), p90=round(float(q[2]), 3)),
        focus=dict(peak_in_target=peak_in_target, peak_dist_mm=round(peak_dist, 1),
                   off_max=round(float(to.max()), 3)),
        safety=dict(hf_peak_target=round(float(hf_t.max()), 3), hf_peak_off=round(float(hf_o.max()), 3)),
        hist=hist,
    )


# ---------- job management ----------
JOBS = {}                      # optimisation jobs — they finish in seconds, memory is enough
LOCK = threading.Lock()

# Backend jobs (Sim4Life, NEURON) **outlive the browser**, so they go to disk (PIPELINE §3-D3)
STORE = JobStore(C.JOBS_DIR)


def _job_protocol(req):
    """The current rule for one request. Returns a `protocol.Protocol`.

    ★The stimulator current is a **protocol choice, not a property of the head.** MIDA carries
    an established `ich_max` (1.0 mA, the tip.lite convention) so nothing changes for it. The
    rat carries `None` — nobody has established what a rat protocol runs at, and inventing a
    number here would silently set the scale of every field the GUI reports. So the browser
    has to send `ich_max` for such a model, and the value it sends is echoed back in the
    protocol block rather than hidden.

    Current is first order in Tmax: it scales M1 and leaves M2/M3 untouched. It therefore
    changes none of the rankings — only the absolute numbers, which is exactly why it must not
    be guessed.
    """
    global CUR_ICH
    v = (req or {}).get("ich_max", None)
    if v in (None, ""):
        v = CUR_ICH if CUR_ICH is not None else C.ICH_MAX
    if v is None:
        raise ValueError(
            f"model {MODEL_NAME!r} has no established stimulation current — "
            f"enter one (mA on the larger channel) before computing.")
    v = float(v)
    if not (v > 0):
        raise ValueError("stimulation current must be greater than zero")
    CUR_ICH = v
    imax = C.IMAX if C.IMAX is not None else v          # no separate cap declared -> the pin
    return v, PROTO.Protocol(name=f"gui({MODEL_NAME})", current_norm="max_channel",
                             budget=v, imax=max(imax, v), off_labels=C.OFF_DEFAULT,
                             note="GUI 입력 전류")


def run_job(job, req):
    def prog(p, stage):
        with LOCK: JOBS[job].update(pct=round(p, 3), stage=stage)
    try:
        prog(0.03, "준비")
        ich, prot = _job_protocol(req)
        target = build_target(req["target"])
        allowed = req.get("allowed") or None
        mode = req.get("mode", "classic")
        names = [e for e in (allowed or LF.names) if LF.has(e)]
        if len(names) < 4: raise ValueError("select at least 4 electrodes")
        f1 = float(req.get("f1", 2000)); f2 = float(req.get("f2", 2100))
        weights = tuple(req.get("weights") or (0.5, 0.5, 0.5))
        pctl = int(req.get("pctl", 50))            # M3 iso-percentile (50 = median threshold)
        tid = req["target"].get("id", "")
        # Unified exhaustive-search parameters. Vectorised with a closed-form Tmax, so it is
        # fast, globally optimal, and a dense ratio grid raises hypervolume.
        BKW = dict(ratio_n=5, ratio_fine=15, max_pairs=50, off_scan=1200, tgt_scan=700,
                   tgt_refine=4000, off_refine=6000, n_refine=80)
        #  ★The search and the scoring have to run under **one** rule. `protocol.use` is a
        #  context manager for that reason: wrapping only the search would let the metrics
        #  fall back to `config` and quietly report a montage at a current it was not found at.
        _pctx = PROTO.use(prot); _pctx.__enter__()
        if mode == "classic":
            if len(names) <= 14:                   # exhaustive = globally optimal; the closed
                                                   # form made this threshold affordable
                r = optimize_classic(LF, target, allowed=names, weights=weights, pctl=pctl,
                                     verbose=False, progress=prog, **BKW)
                method = "전수탐색"
            else:
                r = optimize_nsga(LF, target, allowed=names, precision="low", seeds=4,
                                  off_scan=5000, off_refine=8000, weights=weights, pctl=pctl,
                                  verbose=False, progress=prog)
                method = "NSGA-II"
            best = r["best"]
            pareto = [{"ch1": list(m["ch1"]), "ch2": list(m["ch2"]), "ratio": round(m["ratio"], 3),
                       "M1": round(m["M1"], 4), "M2": round(m["M2"], 3), "M3": round(m["M3"], 2)}
                      for m in r["montages"] if m["pareto"]]
            extra = dict(method=method, n_pareto=r["n_pareto"], n_eval=r["n_eval"],
                         hypervolume=round(r.get("hypervolume", 0.0), 3))
        elif mode == "gevd":                       # distributed (directional array) — shares
                                                   # the Huang 2020 backend
            # hippocampus: anatomical axis · cortex: radial · other deep: 3D-GEVD
            # (chosen automatically by _af_direction)
            direction = _af_direction(target, tid)
            k = req.get("k"); sk = int(k) if k else 32     # more electrodes → tighter deep
                                                          # focus (the paper's point)
            pmax = float(req.get("pmax_rel", 0.1))         # Pmax trades focality against strength
            prog(0.3, "분산 최적화")
            best = optimize_distributed(LF, target, names, direction=direction, select_k=sk, pmax_rel=pmax, verbose=False)
            pareto = []
            extra = dict(method="분산 (방향성·Huang)", pmax_rel=best.get("pmax_rel"), target_mod=best.get("target_mod"))
        elif mode == "dual":
            freqs = [float(x) for x in (req.get("freqs") or [2000, 2010, 2500, 2510])]
            prog(0.3, "이중 TI 최적화")
            r = optimize_dual_ti(LF, target, allowed=names, weights=weights, pctl=pctl,
                                  verbose=False, progress=prog, **BKW)
            best = r["best"]; best["freqs"] = freqs
            pareto = []
            extra = dict(method="이중 TI (2+2)",
                         strength_gain=round(best["M1"] / max(best["M1_A"], 1e-9), 2))
        elif mode == "timemux":                    # time-mux: K slots switched in sequence
                                                   # within an electrode budget
            direction = _af_direction(target, tid)
            prog(0.3, "시분할 최적화")
            best = optimize_timemux(LF, target, names, direction,
                                    K_slots=int(req.get("k_slots", 4)),
                                    max_electrodes=int(req.get("max_elec", 8)),
                                    m1_floor=float(req.get("m1_floor", 0.94)),
                                    progress=prog)
            pareto = []
            extra = dict(method=f"시분할 ({best['K']}슬롯·{best['n_electrodes']}전극)",
                         n_slots=best["K"], n_electrodes=best["n_electrodes"],
                         solo_M2=best["solo_M2"], focality_gain=best["focality_gain"],
                         m1_floor=best["m1_floor"])
        elif mode == "selective":                  # (beta) dual-mechanism, worst-direction
                                                   # selectivity — answers NEURON Q3
            direction = _af_direction(target, tid)
            prog(0.3, "선택성 최적화")
            best = optimize_selective(LF, target, names, direction, weights=weights, pctl=pctl,
                                      ratio_n=5, progress=prog, verbose=False)
            pareto = []
            extra = dict(method="(β) 선택성", M2_field=best["M2_field"], M2_af=best["M2_af"],
                         M3_field=best["M3_field"], M3_af=best["M3_af"])
        elif mode == "huang":                      # (beta) Huang, Datta & Parra 2020 array IFS
                                                   # focality (Pmax)
            direction = _af_direction(target, tid)
            pmax = float(req.get("pmax_rel", 0.15))
            prog(0.3, "Huang 초점 최적화")
            best = optimize_huang(LF, target, names, direction, pmax_rel=pmax)
            for _k in ("s1", "s2", "els"): best.pop(_k, None)     # drop numpy so it serialises
            pareto = []
            extra = dict(method="(β) Huang 초점", pmax_rel=best.get("pmax_rel"),
                         target_mod=best.get("target_mod"), converged=best.get("converged"))
        else:
            raise ValueError(f"unknown mode: {mode}")
        prog(0.9, "필드 계산")
        rm = rich_metrics(best, target)
        result = dict(mode=mode, best=_clean(best), pareto=pareto, used=_used(best),
                      model=MODEL_NAME, ich_max=ich,
                      target_center=[round(float(x), 1) for x in COORDS[target.target_idx].mean(0)],
                      protocol=_protocol(best, f1, f2, ich),
                      freq_check=_freq_check(f1, f2, best.get("freqs")), **extra, **rm)
        _pctx.__exit__(None, None, None); _pctx = None
        with LOCK: JOBS[job].update(pct=1.0, stage="완료", done=True, result=result)
    except Exception as e:
        import traceback; traceback.print_exc()
        with LOCK: JOBS[job].update(done=True, error=str(e))
    finally:
        #  A failure inside the block must not leave this job's rule in force for the next one.
        if locals().get("_pctx") is not None:
            _pctx.__exit__(None, None, None)


def field_for(req):
    """Field and metrics for one specific montage (used when clicking a top-N entry or
    re-ranking by weights)."""
    target = build_target(req["target"])
    m = req["montage"]
    if "ch1" in m:
        best = dict(ch1=tuple(m["ch1"]), ch2=tuple(m["ch2"]), ratio=float(m.get("ratio", 1.0)),
                    M1=m.get("M1"), M2=m.get("M2"), M3=m.get("M3"), WP=m.get("WP"))
    else:
        best = m
    f1 = float(req.get("f1", 2000)); f2 = float(req.get("f2", 2100))
    ich, prot = _job_protocol(req)
    with PROTO.use(prot):
        return dict(best=best, used=_used(best), protocol=_protocol(best, f1, f2, ich),
                    model=MODEL_NAME, ich_max=ich,
                    target_center=[round(float(x), 1) for x in COORDS[target.target_idx].mean(0)],
                    **rich_metrics(best, target))


# ---------- E-field vector visualisation ----------
def _best_from_req(m):
    if "ch1" in m:
        return dict(ch1=tuple(m["ch1"]), ch2=tuple(m["ch2"]), ratio=float(m.get("ratio", 1.0)))
    return m


def _two_fields(best, idx):
    """The two carrier fields (E1, E2), shape (N,3) — the two terms of the TI envelope
    2·min(|n·E1|, |n·E2|). For dual, system A's two channels are shown."""
    if best.get("dual"):
        from tip.optimize.dualti import _cfields
        return _cfields(LF, best["systemA"], idx)
    if best.get("timemux"):        # time-mux: the highest-duty slot represents it in the
                                   # vector and AF views
        return _montage_fields(LF, max(best["slots"], key=lambda s: s.get("duty", 0)), idx)
    return _montage_fields(LF, best, idx)


_SPH = None
def _sph_dirs(D=240):
    global _SPH
    if _SPH is None:
        i = np.arange(D) + 0.5
        ph = np.arccos(1 - 2 * i / D); th = np.pi * (1 + 5 ** 0.5) * i
        _SPH = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], 1)
    return _SPH


def efield_for(req):
    """E1 and E2 vectors at sample points around the target, for the arrow display."""
    target = build_target(req["target"]); best = _best_from_req(req["montage"])
    cen = COORDS[target.target_idx].mean(0)
    R = float(req.get("radius", 34.0))
    d = np.linalg.norm(COORDS[CLOUD] - cen, axis=1)
    sel = CLOUD[d <= R]
    if len(sel) > int(req.get("cap", 1400)):
        sel = np.sort(np.random.default_rng(1).choice(sel, int(req.get("cap", 1400)), replace=False))
    E1, E2 = _two_fields(best, sel)
    env = TI.tmax(E1, E2)
    return dict(pos=[[round(float(x), 1) for x in p] for p in COORDS[sel]],
                idx=[int(x) for x in sel],
                e1=[[round(float(x), 4) for x in v] for v in E1],
                e2=[[round(float(x), 4) for x in v] for v in E2],
                env=[round(float(x), 4) for x in env],
                emax=round(float(max(np.abs(E1).max(), np.abs(E2).max(), 1e-9)), 4))


def pointfield_for(req):
    """E1 and E2 at one point, plus the per-axis envelope and the best direction — this
    feeds the point-analysis panel."""
    best = _best_from_req(req["montage"])
    vi = int(req.get("vi", 0)) if "vi" in req else \
        int(np.argmin(np.linalg.norm(COORDS - np.asarray(req["xyz"], float), axis=1)))
    E1, E2 = _two_fields(best, np.array([vi])); e1 = E1[0]; e2 = E2[0]
    ax = [2.0 * float(min(abs(e1[k]), abs(e2[k]))) for k in range(3)]   # per-axis modulation depth
    dirs = _sph_dirs()
    p1 = np.abs(dirs @ e1); p2 = np.abs(dirs @ e2); envd = 2.0 * np.minimum(p1, p2)
    bi = int(np.argmax(envd)); bd = dirs[bi]

    # Activating function (per axis plus best direction): AF_k(d) = ∂(d·E_k)/∂s,
    # by finite differences over neighbours
    from tip.activating import _tree
    tr = _tree(LF); p0 = COORDS[vi]; h = 2.0; axm = np.eye(3)
    offs = []
    for k in range(3): offs += [p0 + h * axm[k], p0 - h * axm[k]]
    offs += [p0 + h * bd, p0 - h * bd]
    _, iv = tr.query(np.asarray(offs)); E1o, E2o = _two_fields(best, iv)

    def _afm(d, ep, em): return float(abs((ep @ d - em @ d) / (2 * h)))
    af1 = [_afm(axm[k], E1o[2 * k], E1o[2 * k + 1]) for k in range(3)]
    af2 = [_afm(axm[k], E2o[2 * k], E2o[2 * k + 1]) for k in range(3)]
    afenv = [2.0 * min(af1[k], af2[k]) for k in range(3)]
    af1b = _afm(bd, E1o[6], E1o[7]); af2b = _afm(bd, E2o[6], E2o[7])

    return dict(pos=[round(float(x), 1) for x in COORDS[vi]],
                e1=[round(float(x), 4) for x in e1], e2=[round(float(x), 4) for x in e2],
                e1n=round(float(np.linalg.norm(e1)), 4), e2n=round(float(np.linalg.norm(e2)), 4),
                axenv=[round(float(x), 4) for x in ax],
                a1=round(float(abs(e1 @ bd)), 4), a2=round(float(abs(e2 @ bd)), 4),
                tmax=round(float(envd[bi]), 4), bestdir=[round(float(x), 3) for x in bd],
                bestenv=round(float(envd[bi]), 4),
                af1=[round(x, 5) for x in af1], af2=[round(x, 5) for x in af2],   # per-axis AF carrier amplitude
                afenv=[round(x, 5) for x in afenv],                              # per-axis AF modulation depth
                af1b=round(af1b, 5), af2b=round(af2b, 5),
                aftmax=round(2.0 * min(af1b, af2b), 5))


# ---------- activating-function map ----------
def _af_direction(target, tid):
    """Axon direction: hippocampus → anatomical axis; cortex → radial (surface normal, i.e.
    pyramidal cells); other deep targets → the principal 3D-GEVD component."""
    if tid.startswith("hippo") or tid == "label:81":
        d = auto_direction(target, tid)
        if d is not None:
            return np.asarray(d, float)
    from tip.geometry import is_cortical, radial_direction
    if is_cortical(LF, target):                    # cortical target → radial (pyramidal cells)
        return radial_direction(LF, target)
    from tip.benchmark import principal_direction
    return principal_direction(LF, target)


def _af_env_at(best, n, idx, h=None):
    """AF envelope 2·min(|AF1|, |AF2|) at voxels `idx` along direction n, with
    AF_k = ∂(n·E_k)/∂s.

    Trilinear interpolation plus a windowed least-squares gradient — this removes the
    staircase artefact and smooths at the internodal scale (see `fieldsample` §6).
    Works for any montage; the two carrier fields come from `_two_fields`."""
    from tip.fieldsample import af_env
    return af_env(LF, lambda ix: _two_fields(best, ix), n, COORDS[np.asarray(idx)], smooth_mm=h)


def affield_for(req):
    """AF envelope map over the render cloud, along the target axis — for the
    "activating function" display mode."""
    target = build_target(req["target"]); best = _best_from_req(req["montage"])
    tid = req["target"].get("id", "")
    n = _af_direction(target, tid); n = n / (np.linalg.norm(n) + 1e-30)
    af = _af_env_at(best, n, CLOUD)
    tin = np.isin(CLOUD, np.asarray(target.target_idx))
    aft = af[tin]
    return dict(af=[round(float(x), 5) for x in af],
                afmax=round(float(np.percentile(af, 99)), 5),
                direction=[round(float(x), 3) for x in n],
                af_target_med=round(float(np.median(aft)) if tin.any() else 0.0, 5))


def resolution_for(req):
    """Resolution (activated region and focus size), computed on a dense stratified sample:
    every target voxel plus a sample of the rest.
    `mode` = field/af, `ref` = target/max, `frac` sets the threshold. Voxels above the
    threshold count as activated, giving volume, effective diameter, coverage and precision."""
    target = build_target(req["target"]); best = _best_from_req(req["montage"])
    tid = req["target"].get("id", ""); mode = req.get("mode", "field")
    ref = req.get("ref", "target"); frac = float(req.get("frac", 0.5))
    ns = int(req.get("nsamp", 300000))
    ti = np.asarray(target.target_idx)
    rest = BRAINV[~np.isin(BRAINV, ti)]
    k = max(0, ns - len(ti))
    rs = rest if len(rest) <= k else np.sort(np.random.default_rng(0).choice(rest, k, replace=False))
    samp = np.concatenate([ti, rs])
    tin = np.concatenate([np.ones(len(ti), bool), np.zeros(len(rs), bool)])
    if mode == "af":
        n = _af_direction(target, tid); drive = _af_env_at(best, n, samp)
    else:
        drive = _envelope(best, samp)                        # same as the front-end field cloud
                                                             # (dual: combined, distributed: directional)
    gmax = float(drive.max()); tpk = float(drive[tin].max()) if tin.any() else gmax
    thr = frac * max(tpk if ref == "target" else gmax, 1e-12)
    act = drive >= thr
    voxvol = float(np.prod(GS))
    n_t = int(tin.sum()); act_t = int(act[tin].sum())
    rest_frac = float(act[~tin].mean()) if (~tin).any() else 0.0
    total_act = act_t + rest_frac * (len(BRAINV) - len(ti))     # unbiased stratified estimate
    Vact = total_act * voxvol
    diam = 2.0 * (3.0 * max(Vact, 1e-9) / (4 * np.pi)) ** (1.0 / 3.0)
    return dict(Vact=round(Vact, 1), diam=round(diam, 2),
                coverage=round(100.0 * act_t / max(n_t, 1), 1),
                precision=round(100.0 * act_t / max(total_act, 1e-9), 1),
                gmax=round(gmax, 5), tpk=round(tpk, 5), nsamp=int(len(samp)))


# ---------- HTTP ----------
INDEX = os.path.join(HERE, "index.html")


def _left_dir():
    """Unit vector towards the anatomical left, or None if the head does not say.

    The 3D view needs a left/right axis to build its anatomical frame. Deriving it from
    electrode names works on a full 10-20 cap but not on a sparse one: the rat's reference
    PO8 is excluded from the electrode pool, which tilts the derived axis by 17.9 deg. A
    model that carries a fitted midline knows better, so ask it first.
    """
    m = C.MODEL
    if m.midline_normal is not None:                 # oblique midline, normal points left
        n = np.asarray(m.midline_normal, float)
        return [round(float(x), 6) for x in n / np.linalg.norm(n)]
    if m.midline_x is not None:                      # axis-aligned, sign says which side
        return [1.0 if m.left_is_plus_x else -1.0, 0.0, 0.0]
    return None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str): body = body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path.startswith("/index"):
            self._send(open(INDEX, encoding="utf-8").read(), "text/html; charset=utf-8")
        elif u.path.startswith("/vendor/"):
            #  three.js 를 CDN 대신 여기서 준다 — 인터넷이 막힌 자리에서도 3D 가 뜬다.
            #  basename 만 쓴다: "/vendor/../.." 같은 경로로 저장소 밖을 읽지 못하게.
            f = os.path.join(HERE, "vendor", os.path.basename(u.path))
            if not os.path.isfile(f):
                self._send({"error": "not found"}, code=404); return
            self._send(open(f, encoding="utf-8").read(), "application/javascript; charset=utf-8")
        elif u.path == "/api/init":
            #  `label` targets carry a real centre too — they are masks in everything but
            #  name, so the 3D view needs the same treatment as `mask`.
            def _ctr(t):
                if t["kind"] in ("mask", "label"):
                    return [round(float(x), 1) for x in _target_center(t["id"])]
                return t.get("center") if t["kind"] == "sphere" else None
            self._send(dict(
                model=MODEL_NAME, models=available_models(),
                species=C.MODEL.species, ref_elec=C.REF_ELEC,
                #  null = this head has no established stimulation current, so the browser
                #  must ask for one instead of letting the backend invent a scale.
                ich_max=C.ICH_MAX, imax=C.IMAX,
                #  false = the value above is a working number someone picked, not a
                #  convention for this head. The field looks the same either way, so the
                #  UI has to say which, and the reader has to quote it with the result.
                ich_established=bool(C.MODEL.ich_established),
                #  Unit vector pointing to the anatomical LEFT, for the 3D view's frame.
                #  The viewer used to derive left/right from electrode names alone; with the
                #  rat's reference (PO8) missing from the pool that came out 17.9 deg off.
                #  The model's own midline plane is fitted from 9 midline structures with a
                #  0.506 mm spread, so it is the better source. `null` = unknown, and the
                #  viewer falls back to the electrodes.
                left_dir=_left_dir(),
                electrodes=[{"name": e, "xyz": [round(float(v), 1) for v in LF.pos[e]]}
                            for e in LF.names if e in LF.pos],
                cloud=[[round(float(c), 1) for c in p] for p in CLOUD_XYZ],
                targets=[dict(t, center=_ctr(t)) for t in TARGETS],
                brain_center=[round(float(x), 1) for x in BRAIN_C],
                n_brain=int(len(BRAINV)), voxvol=round(float(np.prod(GS)), 4)))   # used to convert resolution into activated volume
        elif u.path == "/api/progress":
            job = parse_qs(u.query).get("job", [""])[0]
            with LOCK: st = dict(JOBS.get(job, {}))
            if not st:                       # backend jobs live in the disk store
                st = STORE.get(job) or {"error": "no job"}
            self._send(st)
        elif u.path == "/api/jobs":
            q = parse_qs(u.query)
            self._send({"jobs": STORE.list(kind=q.get("kind", [None])[0])})
        elif u.path == "/api/cache":
            #  Previously computed analyses, so a person can see whether the same thing is
            #  about to be run again.
            from tip.orch import cache as CACHE
            self._send({"entries": CACHE.listing(
                parse_qs(u.query).get("kind", [None])[0])})
        elif u.path == "/api/montage/result":
            self._send(ORCH_S4L.montage_result(STORE, parse_qs(u.query).get("job", [""])[0]))
        elif u.path == "/api/montage/file":
            #  Slice PNGs from the job folder. `basename` blocks path traversal and the
            #  extension is restricted as well.
            q = parse_qs(u.query)
            d = ORCH_S4L.montage_dir(STORE, q.get("job", [""])[0])
            fn = os.path.basename(q.get("f", [""])[0])
            p = os.path.join(d, fn) if d else None
            if not p or not fn.endswith(".png") or not os.path.exists(p):
                return self._send({"error": "not found"}, code=404)
            self._send(open(p, "rb").read(), "image/png")
        else:
            self._send({"error": "not found"}, code=404)

    def do_POST(self):
        if self.path == "/api/model":
            #  ★Switch head model. Everything derived from the old head is rebuilt; the
            #  browser must reload /api/init afterwards, because its electrodes, targets and
            #  point cloud all belong to the previous head.
            #  Refused while a job is running: an optimisation holds a `Target` built on the
            #  old leadfield, and swapping the globals underneath it would mix two heads in
            #  one result without any error.
            try:
                n = int(self.headers.get("Content-Length", 0))
                want = (json.loads(self.rfile.read(n) or b"{}") or {}).get("model")
                with LOCK:
                    busy = [k for k, v in JOBS.items() if not v.get("done")]
                if busy:
                    return self._send({"error": "작업이 실행 중입니다. 끝난 뒤에 모델을 "
                                                "바꾸세요."}, code=409)
                if want == MODEL_NAME:
                    return self._send({"model": MODEL_NAME, "changed": False})
                if want not in {m["id"] for m in available_models()}:
                    return self._send({"error": f"unknown or unsolved model {want!r}"}, code=400)
                t0 = time.time()
                load_model(want)
                print(_ready_line(), flush=True)
                self._send({"model": MODEL_NAME, "changed": True,
                            "secs": round(time.time() - t0, 1)})
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
            return
        if self.path == "/api/field":
            try:
                n = int(self.headers.get("Content-Length", 0))
                self._send(field_for(json.loads(self.rfile.read(n) or b"{}")))
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
            return
        if self.path in ("/api/efield", "/api/pointfield", "/api/affield", "/api/resolution"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                fn = {"/api/efield": efield_for, "/api/pointfield": pointfield_for,
                      "/api/affield": affield_for, "/api/resolution": resolution_for}[self.path]
                #  Every one of these turns a montage into a field, so every one of them needs
                #  the current — see the CUR_ICH note.
                _, prot = _job_protocol(req)
                with PROTO.use(prot):
                    self._send(fn(req))
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
            return
        if self.path == "/api/tgtvox":     # target boundary voxels, drawn as the green shell
            try:
                n = int(self.headers.get("Content-Length", 0))
                tg = build_target(json.loads(self.rfile.read(n) or b"{}"))
                pts = _contour_pts(tg.target_idx)
                self._send({"pts": [[round(float(x), 1) for x in p] for p in pts]})
            except Exception as e:
                self._send({"error": str(e)}, code=400)
            return
        if self.path == "/api/electrode/solve":
            # ★Solve an electrode in Sim4Life and add it to the leadfield, ~2 min each.
            #  There is one licence seat, so **concurrent runs are refused** — starting two
            #  makes both stall or die.
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                busy = STORE.running("electrode") + STORE.running("montage")
                if busy:
                    return self._send({"error": "a Sim4Life job is already running "
                                                f"(job {busy[0]['id']}); there is only one licence seat.",
                                       "job": busy[0]["id"]}, code=409)
                names = [str(x) for x in (req.get("names") or [])]
                jid = ORCH_S4L.spawn(names, STORE,
                                     out_dir=req.get("out_dir") or None,
                                     dry=bool(req.get("dry")))
                self._send({"job": jid})
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
            return
        if self.path == "/api/job/cancel":
            #  A job takes 7 minutes, so starting one by mistake used to mean waiting it out.
            #  This kills the whole process tree.
            try:
                n = int(self.headers.get("Content-Length", 0))
                jid = str(json.loads(self.rfile.read(n) or b"{}").get("job", ""))
                st = STORE.get(jid)
                if not st:
                    return self._send({"error": "no such job"}, code=404)
                if st.get("done"):
                    return self._send({"ok": False, "note": "job already finished"})
                killed = ORCH_S4L.cancel(jid)
                self._send({"ok": True, "killed": bool(killed), "job": jid})
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
            return
        if self.path == "/api/montage/send":
            # ★Send the chosen montage to Sim4Life and solve the **whole head** (~7 min).
            #  The leadfield holds brain voxels only, so safety numbers such as scalp current
            #  density are invisible there — getting them is the whole point.
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                #  The leadfield metrics computed below for the side-by-side check price the
                #  montage, so this needs the same current rule as everything else.
                ich, prot = _job_protocol(req)
                _pc = PROTO.use(prot); _pc.__enter__()
                #  One seat — refuse if **anything** is running, electrode solve or montage.
                busy = STORE.running("electrode") + STORE.running("montage")
                if busy:
                    return self._send({"error": "a Sim4Life job is already running "
                                                f"(job {busy[0]['id']}); there is only one licence seat.",
                                       "job": busy[0]["id"]}, code=409)
                ch1 = [str(x) for x in (req.get("ch1") or [])]
                ch2 = [str(x) for x in (req.get("ch2") or [])]
                #  ★Dual TI: four pairs forming two independent TI systems. Each pair is
                #  still one two-terminal solve, which is the only thing the exporter can
                #  express — that is why dual TI fits and distributed TI does not (there a
                #  channel is a current distribution over many electrodes, not a pair).
                dual = req.get("dual")
                #  ★Time multiplexing: K slots, each a two-pair montage with a duty cycle.
                #  Same two-terminal solves as everything else — 2K of them — but the slots
                #  do **not** run at once, so the analysis composes them by duty-weighted
                #  time average rather than by adding their envelopes.
                tmux = req.get("timemux")
                pairs = currents = combine = None
                compose, duties = "sum", None
                if tmux:
                    slots = tmux.get("slots") or []
                    if len(slots) < 2:
                        raise ValueError("time multiplexing needs at least two slots")
                    pairs, currents, duties = [], [], []
                    for s in slots:
                        a = [str(x) for x in (s.get("ch1") or [])]
                        b = [str(x) for x in (s.get("ch2") or [])]
                        if len(a) != 2 or len(b) != 2:
                            raise ValueError("every time-mux slot needs two pairs of two "
                                             "electrodes")
                        #  Each slot is a whole montage while it is on, so it is priced by
                        #  the ordinary rule — the same `channel_currents` the optimiser and
                        #  `_envelope` use for a slot. Passing a per-slot budget here would
                        #  make the solve a different stimulation from the one scored.
                        i1, i2 = channel_currents(float(s.get("ratio") or 1.0))
                        pairs += [a, b]
                        currents += [i1, i2]
                        duties.append(float(s.get("duty") or 0.0))
                    tot = sum(duties)
                    if tot <= 0:
                        raise ValueError("time-mux duties are all zero")
                    #  Normalise rather than reject: the optimiser rounds duties to 3 decimals
                    #  when it reports them, so they arrive summing to 0.999 or 1.001.
                    duties = [w / tot for w in duties]
                    combine = [["ch%d" % (2 * k + 1), "ch%d" % (2 * k + 2)]
                               for k in range(len(slots))]
                    compose = "timeavg"
                    ch1, ch2 = pairs[0], pairs[1]
                elif dual:
                    A, B = dual.get("A") or {}, dual.get("B") or {}
                    pairs = [[str(x) for x in (A.get("ch1") or [])],
                             [str(x) for x in (A.get("ch2") or [])],
                             [str(x) for x in (B.get("ch1") or [])],
                             [str(x) for x in (B.get("ch2") or [])]]
                    if any(len(p) != 2 for p in pairs):
                        raise ValueError("dual TI needs four pairs of two electrodes")
                    #  Each system carries half the budget — the rule `optimize_dual_ti`
                    #  optimises under, so the solve must use the same one or the two sides
                    #  of the comparison would be priced differently.
                    ia1, ia2 = channel_currents(float(A.get("ratio") or 1.0), dual_budget())
                    ib1, ib2 = channel_currents(float(B.get("ratio") or 1.0), dual_budget())
                    currents = [ia1, ia2, ib1, ib2]
                    combine = [["ch1", "ch2"], ["ch3", "ch4"]]
                    ch1, ch2 = pairs[0], pairs[1]
                elif len(ch1) != 2 or len(ch2) != 2:
                    raise ValueError("ch1 and ch2 need exactly 2 electrodes each "
                                     "(classic/NSGA), or send `dual` / `timemux`")
                #  The electrode body must exist in the project, i.e. it must be solved
                #  in the leadfield
                bad = [e for e in sum(pairs or [ch1, ch2], []) if e not in LF.names]
                if bad:
                    raise ValueError(f"electrodes not in the leadfield: {bad}")
                #  ★Pass the target along. The Target built here is reused as-is, so the
                #    Sim4Life metrics and the leadfield metrics are computed over **the same
                #    voxel set with the same off-target definition**.
                tgt = lfm = None
                if req.get("target"):
                    tg = build_target(req["target"])
                    tgt = {"target_idx": tg.target_idx.tolist(),
                           "off_idx": tg.off_idx.tolist(),
                           "name": tg.name, "off_def": tg.off_def}
                    #  Leadfield metrics for the same montage, shown side by side as a check
                    #  ⚠ The same `best` shape the optimiser returns, so `_envelope` picks
                    #  the dual rule by itself — the leadfield and Sim4Life sides must be
                    #  built by the same code or the comparison compares two conventions.
                    best = (dict(timemux=True, slots=tmux["slots"]) if tmux
                            else dict(dual=True, systemA=dual["A"], systemB=dual["B"]) if dual
                            else dict(ch1=tuple(ch1), ch2=tuple(ch2),
                                      ratio=float(req.get("ratio") or 1.0)))
                    tt = _envelope(best, tg.target_idx)
                    oi = tg.off_idx[~np.isin(tg.off_idx, tg.target_idx)]
                    to = _envelope(best, oi)
                    lfm = {k: float(v) for k, v in
                           METRICS.all_metrics(tt, to, p=50, reference="target").items()
                           if k in ("M1", "M2", "M3")}
                #  ★Solve it at **the current the user actually chose**, not a constant.
                #  `itotal` here means the *total* injected current, which is what
                #  `s4l_montage.export` feeds to `channel_currents(..., budget=...)`. Under the
                #  GUI rule (max_channel at `ich`) the two channels are i1 = ich/max(1,r) and
                #  i2 = r*i1, so the equivalent total is their sum — passing that reproduces
                #  the same two channel currents in Sim4Life.
                #  ⚠ It used to be hard-coded to 2.0 mA. For MIDA that silently solved the
                #  whole head at twice the 1.0 mA the panel showed; on a rat head a fixed
                #  2 mA is not even the right order of magnitude.
                r_ = float(req.get("ratio") or 1.0)
                if dual or tmux:
                    cur_ = {("ch%d" % (i + 1)): round(c, 6) for i, c in enumerate(currents)}
                    itot_ = float(sum(currents))
                else:
                    i1_, i2_ = channel_currents(r_)
                    cur_ = {"ch1": round(i1_, 6), "ch2": round(i2_, 6)}
                    itot_ = float(req.get("itotal") or (i1_ + i2_))
                jid = ORCH_S4L.spawn_montage(ch1, ch2, STORE,
                                             ratio=r_, itotal=itot_,
                                             use_cache=not req.get("force"),
                                             target=tgt, lf_metrics=lfm,
                                             pairs=pairs, currents=currents, combine=combine,
                                             compose=compose, duties=duties)
                self._send({"job": jid, "ich_max": ich, "currents_mA": cur_,
                            "n_channels": len(pairs or [ch1, ch2])})
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
            finally:
                #  The rule must not outlive this request — a later job would inherit it.
                try:
                    _pc.__exit__(None, None, None)
                except Exception:
                    pass
            return
        if self.path != "/api/optimize":
            return self._send({"error": "not found"}, code=404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            job = uuid.uuid4().hex[:12]
            with LOCK: JOBS[job] = {"pct": 0.0, "stage": "대기", "done": False}
            threading.Thread(target=run_job, args=(job, req), daemon=True).start()
            self._send({"job": job})
        except Exception as e:
            self._send({"error": str(e)}, code=400)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[TI Planner] ▶ {url}  (Ctrl+C to stop)", flush=True)
    try: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception: pass
    srv.serve_forever()


if __name__ == "__main__":
    main()
