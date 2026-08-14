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
import sys, json, time, threading, uuid, webbrowser
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
from tip.optimize.dualti import optimize_dual_ti, combined_env, DUAL_BUDGET  # noqa: E402
from tip.optimize.selective import optimize_selective  # noqa: E402  (beta: selectivity)
from tip.optimize.huang import optimize_huang  # noqa: E402  (beta: Huang 2020 focality)
from tip.optimize.timemux import optimize_timemux  # noqa: E402  (time multiplexing)
from tip.plan import _protocol                          # noqa: E402
from tip.report import _montage_fields                  # noqa: E402
from tip.config import ICH_MAX                           # noqa: E402
from tip import config as C                              # noqa: E402  single source of truth for paths
from tip import metrics as METRICS                       # noqa: E402  M1/M2/M3 are defined in exactly one place
from tip import ti as TI                                 # noqa: E402
from tip.orch import JobStore                            # noqa: E402
from tip.orch import s4l as ORCH_S4L                     # noqa: E402

PORT = 8765
print("[TI Planner] loading leadfield...", flush=True)
LF = LeadField()
HIP = np.load(C.inputs("hipmask1010.npy"))
try:
    THAL = np.load(C.inputs("thalamus_mask.npy"))
except Exception:
    THAL = None
HAX = np.load(C.inputs("hipaxes1010.npz"))
BL = np.load(C.inputs("blabel1010.npy"))
COORDS = LF.coords()
GMIN = COORDS.min(0)                                  # grid origin and spacing (target contours)
GS = np.array([np.median(np.diff(np.unique(COORDS[:, k]))) for k in range(3)])
GS[GS < 1e-6] = 1.0
NEURAL = np.where(BL == 75)[0]                       # off-target = grey matter only (tip.lite)
BRAINV = np.arange(len(BL))                          # every brain voxel (3D view, deep targets)
BRAIN_C = COORDS[NEURAL].mean(0)
ELPOS = np.array([LF.pos[e] for e in LF.names])
_rng = np.random.default_rng(0)
CLOUD = np.sort(_rng.choice(BRAINV, min(100000, len(BRAINV)), replace=False))  # 3D render cloud (~117 points/cm3, ~2.0 mm spacing — keeps tight foci visible)
CLOUD_XYZ = COORDS[CLOUD]
OFFSAMP = np.sort(_rng.choice(NEURAL, min(15000, len(NEURAL)), replace=False))

# Standard stimulation targets (approximate MIDA coordinates, user-adjustable).
# Mirrored about the midline at X = -27.
def _mirror(c): return [-54.0 - c[0], c[1], c[2]]

_DEEP = {  # coordinate-based targets (cortical sites with no mask) — key: (left centre, r, ko, en)
    "dlpfc": ([-53, 320, -18], 9, "배외측전전두 DLPFC", "DLPFC"),
    "m1":    ([-58, 332, 34], 9, "일차운동피질 M1", "Primary motor (M1)"),
    "a6dl":  ([-60, 318, 40], 8, "전운동피질 A6dl", "Premotor (A6dl)"),
}
TARGETS = [
    dict(id="hippo_left",  kind="mask", ko="해마 (좌)", en="Hippocampus (L)"),
    dict(id="hippo_right", kind="mask", ko="해마 (우)", en="Hippocampus (R)"),
]
if THAL is not None:
    TARGETS += [dict(id="thalamus_left",  kind="mask", ko="시상 (좌)", en="Thalamus (L)"),
                dict(id="thalamus_right", kind="mask", ko="시상 (우)", en="Thalamus (R)")]
for _k, (_c, _r, _ko, _en) in _DEEP.items():
    TARGETS.append(dict(id=_k + "_L", kind="sphere", center=[float(x) for x in _c], radius=float(_r),
                        ko=f"{_ko} (좌)", en=f"{_en} (L)"))
    TARGETS.append(dict(id=_k + "_R", kind="sphere", center=_mirror([float(x) for x in _c]), radius=float(_r),
                        ko=f"{_ko} (우)", en=f"{_en} (R)"))
TARGETS.append(dict(id="cingulate", kind="sphere", center=[-27.0, 272.0, -18.0], radius=5.0,
                    ko="슬하대상 Cg25", en="Subgenual cingulate"))
TARGETS.append(dict(id="brain_center", kind="sphere", center=[float(x) for x in BRAIN_C], radius=10.0,
                    ko="뇌 중심", en="Brain center"))
TARGETS.append(dict(id="custom", kind="custom", ko="임의 좌표…", en="Custom coords…"))

# ---- real masks from the manifest (multi-region, extracted from Sim4Life) ----
MMASK = {}   # base id -> (2, N) bool
_mdir = os.path.join(C.MASKS_DIR, "mida")
_mpath = os.path.join(_mdir, "manifest.json")
if os.path.exists(_mpath):
    try:
        _man = json.load(open(_mpath, encoding="utf-8"))
    except Exception:
        _man = []
    _mt, _ids = [], set()
    for _m in _man:
        if _m["id"] in ("hippocampus", "thalamus"):   # already hard-coded as HIP/THAL — avoid duplicates
            continue
        try:
            MMASK[_m["id"]] = np.load(os.path.join(_mdir, _m["file"]))
        except Exception:
            continue
        _ids.add(_m["id"])
        if _m.get("bilateral", True):
            _mt += [dict(id=_m["id"] + "_left", kind="mask", ko=_m["ko"] + " (좌)", en=_m["en"] + " (L)"),
                    dict(id=_m["id"] + "_right", kind="mask", ko=_m["ko"] + " (우)", en=_m["en"] + " (R)")]
        else:
            _mt.append(dict(id=_m["id"] + "_left", kind="mask", ko=_m["ko"], en=_m["en"]))
    # drop the approximate-sphere presets that a real mask now replaces
    TARGETS = [t for t in TARGETS if not (t.get("kind") == "sphere" and t["id"].rsplit("_", 1)[0] in _ids)]
    _ins = 4 if THAL is not None else 2   # insert after hippocampus (and thalamus)
    TARGETS = TARGETS[:_ins] + _mt + TARGETS[_ins:]

print(f"[TI Planner] ready · {len(LF.names)} electrodes · {len(TARGETS)} targets · "
      f"{len(MMASK)} masks", flush=True)


def _target_center(tid):
    if tid == "hippo_left":  return COORDS[np.where(HIP[0])[0]].mean(0)
    if tid == "hippo_right": return COORDS[np.where(HIP[1])[0]].mean(0)
    if tid == "thalamus_left"  and THAL is not None: return COORDS[np.where(THAL[0])[0]].mean(0)
    if tid == "thalamus_right" and THAL is not None: return COORDS[np.where(THAL[1])[0]].mean(0)
    _b = tid.rsplit("_", 1)[0]
    if _b in MMASK:
        return COORDS[np.where(MMASK[_b][0 if tid.endswith("_left") else 1])[0]].mean(0)
    return BRAIN_C


def auto_direction(target, tid):
    """Derive the stimulation direction from the target (hippocampus → its axis,
    otherwise → radial from the scalp)."""
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
    if tid == "hippo_left":  return Target.from_mask(LF, HIP[0], name="Hippocampus L", **kw)
    if tid == "hippo_right": return Target.from_mask(LF, HIP[1], name="Hippocampus R", **kw)
    if tid == "thalamus_left"  and THAL is not None: return Target.from_mask(LF, THAL[0], name="Thalamus L", **kw)
    if tid == "thalamus_right" and THAL is not None: return Target.from_mask(LF, THAL[1], name="Thalamus R", **kw)
    _b = tid.rsplit("_", 1)[0]
    if _b in MMASK:
        return Target.from_mask(LF, MMASK[_b][0 if tid.endswith("_left") else 1], name=tid, **kw)
    raise ValueError("unknown target")


def build_dir(spec):
    if spec in (None, "", "none"): return None
    if spec == "hippo_nL": return HAX["nL"]
    if spec == "hippo_nR": return HAX["nR"]
    if isinstance(spec, (list, tuple)): return np.asarray(spec, float)
    return None


def _used(best):
    used = {}
    if best.get("dual"):     # 4-channel: system A = ch0/1 (reds), system B = ch2/3 (blues);
                             # ITOTAL/2 per system
        for sk, base in (("systemA", 0), ("systemB", 2)):
            s = best[sk]; a, b = s["ch1"]; c, d = s["ch2"]; r = s.get("ratio", 1.0)
            i1, i2 = channel_currents(r, DUAL_BUDGET)
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


def run_job(job, req):
    def prog(p, stage):
        with LOCK: JOBS[job].update(pct=round(p, 3), stage=stage)
    try:
        prog(0.03, "준비")
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
                      target_center=[round(float(x), 1) for x in COORDS[target.target_idx].mean(0)],
                      protocol=_protocol(best, f1, f2, ICH_MAX),
                      freq_check=_freq_check(f1, f2, best.get("freqs")), **extra, **rm)
        with LOCK: JOBS[job].update(pct=1.0, stage="완료", done=True, result=result)
    except Exception as e:
        import traceback; traceback.print_exc()
        with LOCK: JOBS[job].update(done=True, error=str(e))


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
    return dict(best=best, used=_used(best), protocol=_protocol(best, f1, f2, ICH_MAX),
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
        elif u.path == "/api/init":
            self._send(dict(
                electrodes=[{"name": e, "xyz": [round(float(v), 1) for v in LF.pos[e]]}
                            for e in LF.names if e in LF.pos],
                cloud=[[round(float(c), 1) for c in p] for p in CLOUD_XYZ],
                targets=[dict(t, center=([round(float(x), 1) for x in _target_center(t["id"])]
                                         if t["kind"] == "mask" else
                                         (t.get("center") if t["kind"] == "sphere" else None)))
                         for t in TARGETS],
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
                #  One seat — refuse if **anything** is running, electrode solve or montage.
                busy = STORE.running("electrode") + STORE.running("montage")
                if busy:
                    return self._send({"error": "a Sim4Life job is already running "
                                                f"(job {busy[0]['id']}); there is only one licence seat.",
                                       "job": busy[0]["id"]}, code=409)
                ch1 = [str(x) for x in (req.get("ch1") or [])]
                ch2 = [str(x) for x in (req.get("ch2") or [])]
                if len(ch1) != 2 or len(ch2) != 2:
                    raise ValueError("ch1 and ch2 need exactly 2 electrodes each "
                                     "(only two-pair montages are supported)")
                #  The electrode body must exist in the project, i.e. it must be solved
                #  in the leadfield
                bad = [e for e in ch1 + ch2 if e not in LF.names]
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
                    best = dict(ch1=tuple(ch1), ch2=tuple(ch2),
                                ratio=float(req.get("ratio") or 1.0))
                    tt = _envelope(best, tg.target_idx)
                    oi = tg.off_idx[~np.isin(tg.off_idx, tg.target_idx)]
                    to = _envelope(best, oi)
                    lfm = {k: float(v) for k, v in
                           METRICS.all_metrics(tt, to, p=50, reference="target").items()
                           if k in ("M1", "M2", "M3")}
                #  ★force=True skips the cache and re-solves (when the model is in doubt).
                jid = ORCH_S4L.spawn_montage(ch1, ch2, STORE,
                                             ratio=float(req.get("ratio") or 1.0),
                                             itotal=float(req.get("itotal") or 2.0),
                                             use_cache=not req.get("force"),
                                             target=tgt, lf_metrics=lfm)
                self._send({"job": jid})
            except Exception as e:
                import traceback; traceback.print_exc(); self._send({"error": str(e)}, code=400)
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
