# -*- coding: utf-8 -*-
"""
benchmark.py — the fair-comparison benchmark harness
====================================
Whatever stimulation method goes in — classic, dual, distributed, or something new — it is
compared automatically on **the same target, the same total injected current (config.ITOTAL)
and the same metrics**. This is the infrastructure that keeps every experiment on one yardstick.

  · add a method : register a callable (lf, target, allowed, n, weights, pctl) -> best in METHODS
  · add a target : extend standard_targets(), or pass (label, Target, direction) straight to
                   benchmark()

Metrics, all at fixed total current:
  isotropic (Tmax over all directions)  M1 strength · M2 selectivity · M3 leakage %
  directional (along the target axis)   M1 · M2 · M3   (axis = anatomical, or the principal
                                                        3D-GEVD component)
  I_total — this must come out ~equal across methods, otherwise the comparison is not fair

Example:
    from tip import LeadField
    from tip.benchmark import benchmark, standard_targets
    LF = LeadField()
    res = benchmark(LF, standard_targets(LF, which=["해마", "시상"]))
"""
from . import config as C
import numpy as np
from . import ti as TI, metrics as MM
from .report import _montage_fields

# Unified exhaustive / joint parameters, identical to the GUI
BKW = dict(ratio_n=7, ratio_fine=15, max_pairs=50, off_scan=1400, tgt_scan=700,
           tgt_refine=4000, off_refine=6000, n_refine=80)


# ===================== shared evaluation =====================
def total_current(best):
    """Total injected current I_total = sum(current into + electrodes) = 0.5·sum|all currents|.

    Under time multiplexing only one slot is active at a time, so the **peak equals the largest
    component current** (~ITOTAL, i.e. the same budget as a static montage)."""
    from .optimize.classic import channel_currents
    from .optimize.dualti import DUAL_BUDGET
    if best.get("timemux"):
        return max(total_current(c) for c in best["components"])
    if best.get("complex"):
        return 0.5 * (float(np.abs(best["c0"]).sum()) + float(np.abs(best["c1"]).sum()))
    if best.get("dual"):
        t = 0.0
        for sk in ("systemA", "systemB"):
            i1, i2 = channel_currents(best[sk]["ratio"], DUAL_BUDGET); t += i1 + i2
        return t
    if "currents" in best:
        c = best["currents"]
        return 0.5 * (sum(abs(v) for v in c["ch0"].values()) + sum(abs(v) for v in c["ch1"].values()))
    i1, i2 = channel_currents(best["ratio"])
    return i1 + i2


def _pairs(lf, best, idx):
    """Method-agnostic list of (E1, E2) channel pairs. Isotropic = sum of tmax over the pairs,
    directional = sum of directional_env over the pairs.
    classic and distributed give [(E1,E2)]; dual gives [(A1,A2), (B1,B2)]."""
    from .optimize.dualti import _cfields
    if best.get("dual"):
        return [_cfields(lf, best["systemA"], idx), _cfields(lf, best["systemB"], idx)]
    return [_montage_fields(lf, best, idx)]   # classic and distributed (the `currents` branch
                                              # is handled inside)


def principal_direction(lf, target, seed=7):
    """For a target with no anatomical axis, use the principal 3D-GEVD component — the
    direction the field naturally concentrates along."""
    from .optimize.multichannel import _gevd3d
    names = [e for e in lf.names if lf.has(e)]
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    ts = tgt if len(tgt) <= 700 else np.sort(rng.choice(tgt, 700, replace=False))
    ob = off if len(off) <= 1400 else np.sort(rng.choice(off, 1400, replace=False))
    Ft = np.stack([lf.elec_field(e, ts) for e in names]); Fo = np.stack([lf.elec_field(e, ob) for e in names])
    _, _, n3 = _gevd3d(Ft, Fo)
    return n3


def _complex_env(lf, best, idx, n, dirs):
    """Isotropic and directional envelopes for a complex (phase) solution. Each carrier is
    elliptically polarised, so |direction·E| is the magnitude of a complex projection.
    Isotropic = 2·max_dir min(|d·E0|, |d·E1|), found by scanning directions — with ellipses
    there is no closed form."""
    els = best["electrodes"]; c0 = best["c0"]; c1 = best["c1"]
    F = np.stack([lf.elec_field(e, idx) for e in els])          # (K,N,3) real
    E0 = np.einsum("k,knd->nd", c0, F); E1 = np.einsum("k,knd->nd", c1, F)   # (N,3) complex
    dr = 2.0 * np.minimum(np.abs(E0 @ n), np.abs(E1 @ n))       # modulation depth along n
    p0 = np.abs(E0 @ dirs.T); p1 = np.abs(E1 @ dirs.T)          # (N, D)
    iso = 2.0 * np.max(np.minimum(p0, p1), axis=1)             # isotropic
    return iso, dr


def _envelope_arrays(lf, best, idx, n):
    """(isotropic env, directional env) arrays, handling every montage structure.
    dual sums its components (they run simultaneously); time-mux averages them over time, or
    takes the peak."""
    if best.get("complex"):
        from .optimize.multichannel import _sphere_dirs
        return _complex_env(lf, best, idx, n, _sphere_dirs(48))
    if best.get("timemux"):
        comps = [_montage_fields(lf, c, idx) for c in best["components"]]
        iso = [TI.tmax(*p) for p in comps]
        dr = [TI.directional_env(p[0], p[1], n) for p in comps]
        if best.get("combine", "avg") == "peak":     # instantaneous peak response: max per point
            return np.max(iso, axis=0), np.max(dr, axis=0)
        d = best.get("duties") or [1.0 / len(comps)] * len(comps)   # time average: convex combination
        return sum(dk * e for dk, e in zip(d, iso)), sum(dk * e for dk, e in zip(d, dr))
    pairs = _pairs(lf, best, idx)                    # classic/distributed: 1 pair;
                                                     # dual: 2 pairs summed simultaneously
    return (sum(TI.tmax(*p) for p in pairs),
            sum(TI.directional_env(p[0], p[1], n) for p in pairs))


def _af_env(lf, best, idx, n, h):
    """Directional AF (activating-function) envelope of a classic montage. Returns None for
    non-classic structures (dual, distributed, time-mux, complex)."""
    if "ch1" not in best:
        return None
    from .activating import af_proj, dir_envelope
    (a, b) = best["ch1"]; (c, d) = best["ch2"]
    AF = af_proj(lf, [a, b, c, d], n, idx, h)
    return dir_envelope(AF, ((0, 1), (2, 3)), best.get("ratio", 1.0))


def evaluate(lf, best, target, n, ti_n=4000, off_n=8000, pctl=50, seed=3, h_af=None):
    """Isotropic, directional and AF metrics plus I_total for one solution. Every method sees
    the same subsample of a given target."""
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)
    tr = tgt if len(tgt) <= ti_n else np.sort(rng.choice(tgt, ti_n, replace=False))
    orf = off if len(off) <= off_n else np.sort(rng.choice(off, off_n, replace=False))
    iso_t, dir_t = _envelope_arrays(lf, best, tr, n)
    iso_o, dir_o = _envelope_arrays(lf, best, orf, n)

    def trio(t, o):
        return dict(M1=round(float(np.median(t)), 3), M2=round(float(MM.M2(t, o)), 2),
                    M3=round(100.0 * float(np.mean(o > np.percentile(t, pctl))), 1))
    out = dict(iso=trio(iso_t, iso_o), dir=trio(dir_t, dir_o), I_total=round(float(total_current(best)), 2))
    af_t = _af_env(lf, best, tr, n, h_af)
    if af_t is not None:
        out["af"] = trio(af_t, _af_env(lf, best, orf, n, h_af))
    return out


# ===================== method registry =====================
def _gevd_top(lf, target, k, seed=7):
    from .optimize.multichannel import _gevd3d
    names = [e for e in lf.names if lf.has(e)]
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    ts = tgt if len(tgt) <= 700 else np.sort(rng.choice(tgt, 700, replace=False))
    ob = off if len(off) <= 1400 else np.sort(rng.choice(off, 1400, replace=False))
    Ft = np.stack([lf.elec_field(e, ts) for e in names]); Fo = np.stack([lf.elec_field(e, ob) for e in names])
    _, cstar, _ = _gevd3d(Ft, Fo)
    return [names[i] for i in sorted(np.argsort(-np.abs(cstar))[:k].tolist())]


def _run_classic(lf, target, allowed, n, weights, pctl):
    from .optimize.classic import optimize_classic
    from .optimize.nsga import optimize_nsga
    if len(allowed) <= 14:
        return optimize_classic(lf, target, allowed=allowed, weights=weights, pctl=pctl,
                                verbose=False, **BKW)["best"]
    pool = _gevd_top(lf, target, 14)   # reduce to the GEVD top 14 to keep the search exhaustive
    return optimize_classic(lf, target, allowed=pool, weights=weights, pctl=pctl,
                            verbose=False, **BKW)["best"]


def _run_dual(lf, target, allowed, n, weights, pctl):
    from .optimize.dualti import optimize_dual_ti
    return optimize_dual_ti(lf, target, allowed=allowed, weights=weights, pctl=pctl,
                            verbose=False, **BKW)["best"]


def _run_distributed(lf, target, allowed, n, weights, pctl):
    from .optimize.multichannel import optimize_distributed
    sk = 30 if len(allowed) > 30 else None
    return optimize_distributed(lf, target, allowed, direction=n, off_subsample=6000,
                                select_k=sk, weights=weights, pctl=pctl, verbose=False)


def _run_timemux(lf, target, allowed, n, weights, pctl, K=3, combine="avg"):
    """Time multiplexing: greedily build K **diverse** component montages (excluding already
    used electrodes) and share time between them.
    Each component runs at the full current (ITOTAL) during its slot, which keeps both peak and
    mean current identical to a static montage — that is what makes the comparison fair.
    `combine='avg'` is the time average (a convex combination); `'peak'` is the instantaneous
    maximum response (max per point)."""
    from .optimize.classic import optimize_classic
    pool = allowed if len(allowed) <= 14 else _gevd_top(lf, target, min(len(allowed), 4 * K + 6))
    comps = []; used = set()
    for _ in range(K):
        rem = [e for e in pool if e not in used]
        if len(rem) < 4:
            break
        b = optimize_classic(lf, target, allowed=rem, weights=weights, pctl=pctl, verbose=False, **BKW)["best"]
        comps.append(b); used |= set(b["ch1"]) | set(b["ch2"])
    return dict(timemux=True, components=comps,
                duties=[1.0 / len(comps)] * len(comps), combine=combine)


def _subs(target, seed=1, tn=1500, on=3000):
    rng = np.random.default_rng(seed)
    ti = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    tr = ti if len(ti) <= tn else np.sort(rng.choice(ti, tn, replace=False))
    orr = off if len(off) <= on else np.sort(rng.choice(off, on, replace=False))
    return tr, orr


def _run_field_dir(lf, target, allowed, n, weights, pctl):
    """Montage that maximises the |n·E| field-projection envelope along n — searched
    exhaustively over the same electrode set as the AF method, so the two compare directly."""
    from .activating import field_proj, optimize_projected
    els = allowed if len(allowed) <= 14 else _gevd_top(lf, target, 14)
    tr, orr = _subs(target)
    Pt = field_proj(lf, els, n, tr); Po = field_proj(lf, els, n, orr)
    return optimize_projected(Pt, Po, els, weights, pctl)


def _run_af(lf, target, allowed, n, weights, pctl):
    """Montage that maximises the activating-function envelope AF = d(n·E)/ds — the drive for
    a long fibre."""
    from .activating import af_proj, optimize_projected
    els = allowed if len(allowed) <= 14 else _gevd_top(lf, target, 14)
    tr, orr = _subs(target)
    Pt = af_proj(lf, els, n, tr); Po = af_proj(lf, els, n, orr)
    m = optimize_projected(Pt, Po, els, weights, pctl); m["af_opt"] = True
    return m


def _run_gaf(lf, target, allowed, n, weights, pctl):
    """Montage that maximises the generalised activating function (GAF, the MDF2 form of
    Peterson & Grill).
    The AF is spatially filtered at the node spacing with a myelinated axon's passive-cable
    kernel, which removes the high-frequency content of the raw AF."""
    from .fieldsample import gaf_proj_elec
    from .activating import optimize_projected
    els = allowed if len(allowed) <= 14 else _gevd_top(lf, target, 14)
    tr, orr = _subs(target)
    Pt = gaf_proj_elec(lf, els, n, tr); Po = gaf_proj_elec(lf, els, n, orr)
    m = optimize_projected(Pt, Po, els, weights, pctl); m["gaf_opt"] = True
    return m


METHODS = {"classic": _run_classic, "dual": _run_dual, "distributed": _run_distributed,
           "field_dir": _run_field_dir, "af": _run_af, "gaf": _run_gaf,
           # Time-mux with greedy K components and equal duty. `_envelope_arrays` understands
           # components/duties, so this evaluates as-is. The constrained variant (strength
           # floor plus SLSQP duties) is registered separately via
           # `optimize.timemux.make_benchmark_method()` — register both and compare.
           "timemux": _run_timemux}


def register_method(name, fn):
    """Register a new method: fn(lf, target, allowed_names, n_dir, weights, pctl) -> best dict.

    `best` is evaluated automatically if it is one of the known shapes: classic-like
    (ch1/ch2/ratio), dual (dual=True with systemA/systemB), or distributed (currents). Any
    other structure needs `_pairs` and `total_current` extended first."""
    METHODS[name] = fn


# ===================== runner and table =====================
def _row(m, r):
    i, d = r["iso"], r["dir"]
    return (f"{m:14}|{i['M1']:8.3f}{i['M2']:7.2f}{i['M3']:7.1f}  |"
            f"{d['M1']:8.3f}{d['M2']:7.2f}{d['M3']:7.1f}  | {r['I_total']:5.2f}")


def _print_target(label, has_axis, res):
    print(f"\n{'=' * 80}")
    print(f"■ {label}   (fixed total current, axis = "
          f"{'anatomical' if has_axis else 'principal 3D-GEVD component'})")
    print("=" * 80)
    print(f"{'method':14}|{'iso M1':>8}{'M2':>7}{'M3%':>7}  |{'dir M1':>8}{'M2':>7}{'M3%':>7}  | I_tot")
    print("-" * 80)
    for m, r in res.items():
        print(_row(m, r))


def benchmark(lf, targets, methods=None, weights=(0.5, 0.5, 0.5), pctl=50, verbose=True):
    """
    targets : [(label, Target, direction_or_None), ...]
    methods : list of names (all of them by default).
    Returns {label: {"n": ..., "methods": {m: {iso, dir, I_total, best}}}}.
    """
    methods = methods or list(METHODS)
    names = [e for e in lf.names if lf.has(e)]
    out = {}
    for label, target, direction in targets:
        n = np.asarray(direction, float) if direction is not None else principal_direction(lf, target)
        n = n / (np.linalg.norm(n) + 1e-30)
        res = {}
        for mname in methods:
            best = METHODS[mname](lf, target, names, n, weights, pctl)
            m = evaluate(lf, best, target, n, pctl=pctl); m["best"] = best
            res[mname] = m
        out[label] = dict(n=[float(x) for x in n], methods=res)
        if verbose:
            _print_target(label, direction is not None, res)
    return out


def standard_targets(lf, which=None):
    """Standard targets as [(label, Target, direction)].
    `which` filters by label substring, e.g. ["해마", "시상"].

    ⚠ The labels stay Korean on purpose: research scripts under `research/` call this with
    `which=["해마"]` and renaming them here would break those callers silently."""
    import os
    import json
    from . import Target
    dd = C.INPUTS_DIR
    out = []
    try:
        HAX = np.load(os.path.join(dd, "hipaxes1010.npz"))
        HIP = np.load(os.path.join(dd, "hipmask1010.npy"))
        out.append(("해마 L", Target.from_mask(lf, HIP[0], off_subsample=22000), HAX["nL"]))
        out.append(("해마 R", Target.from_mask(lf, HIP[1], off_subsample=22000), HAX["nR"]))
    except Exception:
        pass
    tp = os.path.join(dd, "thalamus_mask.npy")
    if os.path.exists(tp):
        TH = np.load(tp)
        out.append(("시상 L", Target.from_mask(lf, TH[0], off_subsample=22000), None))
    mdir = os.path.join(dd, "masks"); mpath = os.path.join(mdir, "manifest.json")
    if os.path.exists(mpath):
        try:
            man = json.load(open(mpath, encoding="utf-8"))
        except Exception:
            man = []
        for m in man:
            if m["id"] in ("hippocampus", "thalamus"):
                continue
            try:
                arr = np.load(os.path.join(mdir, m["file"]))
            except Exception:
                continue
            out.append((m.get("ko", m["id"]) + " L", Target.from_mask(lf, arr[0], off_subsample=22000), None))
    if which:
        out = [t for t in out if any(w in t[0] for w in which)]
    return out
