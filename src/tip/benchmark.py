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
    component current** (~ITOTAL, i.e. the same budget as a static montage).

    ⚠ That is the **peak** basis, and it is the one basis under which sequential TI has an
    advantage: a K-slot schedule may spend the full budget in every slot while the peak counts
    one slot. Per-electrode time-averaged load — the constraint under which that advantage
    disappears — is **not** computed here. `Protocol.dose_basis` names the choice so a table
    cannot be read as "sequential wins" without saying under which constraint; it currently
    refuses any value but "peak" rather than silently measuring the wrong thing."""
    from .optimize.classic import channel_currents
    from .optimize.dualti import dual_budget
    if best.get("timemux"):
        return max(total_current(c) for c in best["components"])
    if best.get("complex"):
        return 0.5 * (float(np.abs(best["c0"]).sum()) + float(np.abs(best["c1"]).sum()))
    if best.get("dual"):
        t = 0.0
        for sk in ("systemA", "systemB"):
            #  ★`dual_budget()`, not the `DUAL_BUDGET` constant. The constant hard-codes a
            #  total rule (ITOTAL/2 per system), so it is right under FAIR — the default here —
            #  and wrong under every other rule. `benchmark(protocol=...)` documents TIPLITE as
            #  an argument, and under it each system really draws the pinned channel: the
            #  constant reported 해마 L and 시상 L as I_total 1.00 mA when the montage draws
            #  ~4.0 mA. `dual_budget()` reads `protocol.current()` — the same source
            #  `dualti._cfields` uses to build the fields these metrics are measured on — so
            #  the current counter and the field can no longer disagree.
            #  FAIR output verified bit-identical before and after (2026-08-21).
            i1, i2 = channel_currents(best[sk]["ratio"], dual_budget()); t += i1 + i2
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


def evaluate(lf, best, target, n, ti_n=4000, off_n=8000, pctl=50, seed=3, h_af=None,
             protocol=None):
    """Isotropic, directional and AF metrics plus I_total for one solution. Every method sees
    the same subsample of a given target.

    ★`protocol` (see `protocol.py`) is what makes methods comparable. Under a `"total"` rule
    the solution is **renormalised to the protocol's budget before scoring**. That is exact:
    Tmax is positively homogeneous of degree 1 in the channel currents, so scaling every
    current by k scales the envelope by k — M1 scales, M2 and M3 are invariant.

    ⚠ Without this, `config.CURRENT_NORM="max_channel"` (the tip.lite rule, the default since
    2026-08-13) lets classic draw **2.00 mA** while dual and distributed draw 1.00 mA, and the
    table still says "fixed total current". Measured on 좌시상: classic iso M1 0.141 vs dual
    0.119 / distributed 0.118 — an ordering produced by dose, not by method."""
    from . import protocol as P
    prot = protocol if protocol is not None else P.active()
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)
    tr = tgt if len(tgt) <= ti_n else np.sort(rng.choice(tgt, ti_n, replace=False))
    orf = off if len(off) <= off_n else np.sort(rng.choice(off, off_n, replace=False))
    raw_total = float(total_current(best))
    k = prot.scale_for(raw_total)
    iso_t, dir_t = (a * k for a in _envelope_arrays(lf, best, tr, n))
    iso_o, dir_o = (a * k for a in _envelope_arrays(lf, best, orf, n))

    def trio(t, o):
        return dict(M1=round(float(np.median(t)), 3), M2=round(float(MM.M2(t, o)), 2),
                    M3=round(100.0 * float(np.mean(o > np.percentile(t, pctl))), 1))
    out = dict(iso=trio(iso_t, iso_o), dir=trio(dir_t, dir_o),
               I_total=round(raw_total * k, 2),      # after renormalisation — must match
               I_raw=round(raw_total, 2),            # what the optimiser actually produced
               scale=round(k, 4), protocol=prot.label)
    af_t = _af_env(lf, best, tr, n, h_af)
    if af_t is not None:
        out["af"] = trio(af_t * k, _af_env(lf, best, orf, n, h_af) * k)
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
    raw = r.get("I_raw")
    #  원래 전류를 함께 찍는다 — 환산이 있었는지 표에서 바로 보이게 하기 위함
    tail = f" {r['I_total']:5.2f} {(f'({raw:.2f})' if raw is not None else ''):>8}"
    return (f"{m:14}|{i['M1']:8.3f}{i['M2']:7.2f}{i['M3']:7.1f}  |"
            f"{d['M1']:8.3f}{d['M2']:7.2f}{d['M3']:7.1f}  |{tail}")


def _print_target(label, has_axis, res):
    print(f"\n{'=' * 84}")
    print(f"■ {label}   (axis = {'anatomical' if has_axis else 'principal 3D-GEVD component'})")
    any_r = next(iter(res.values()), None)
    if any_r and any_r.get("protocol"):
        print(f"  규약: {any_r['protocol']}")
    print("=" * 84)
    print(f"{'method':14}|{'iso M1':>8}{'M2':>7}{'M3%':>7}  |{'dir M1':>8}{'M2':>7}{'M3%':>7}"
          f"  | {'I_tot':>5} {'(원래)':>8}")
    print("-" * 84)
    for m, r in res.items():
        print(_row(m, r))

    #  ★자가검증 — 규약을 강제한 뒤에도 dose 가 어긋나면 표를 읽어선 안 된다.
    from . import protocol as P
    ok, spread = P.check_equal_dose([r.get("I_total") for r in res.values()])
    raws = {m: r.get("I_raw") for m, r in res.items() if r.get("I_raw")}
    #  ★"구조상 반드시 이 값" 칸 — 총전류 규약에서는 탐색이 이미 예산을 지켰어야 하므로
    #  환산 배율이 1.0 이어야 한다. 아니면 전류를 세는 코드가 규약을 못 읽은 것이다(결함).
    #  실제로 이 검사가 `evaluate` 를 컨텍스트 밖에서 부르던 누수를 잡아냈다(2026-08-18).
    prot0 = next((r.get("protocol", "") for r in res.values()), "")
    if "총전류 고정" in prot0:
        bad = {m: r["scale"] for m, r in res.items()
               if abs(r.get("scale", 1.0) - 1.0) > 0.02}
        if bad:
            print(f"\n  ★★ 결함: 총전류 규약인데 환산 배율이 1 이 아니다 → {bad}")
            print(f"     탐색이 규약을 못 읽었거나 `total_current` 가 그 구조를 못 센다.")
    if not ok:
        print(f"\n  ★★ 경고: 채점 후에도 총전류가 {spread:.1%} 벌어져 있다 — **이 표를 비교에 쓰지 말 것.**")
        print(f"     구조별 전류 환산(`total_current`)에 빠진 방법이 있다는 뜻이다(모델링 선택이 아니라 결함).")
    elif raws and (max(raws.values()) - min(raws.values())) / max(raws.values()) > 0.02:
        lo = min(raws, key=raws.get); hi = max(raws, key=raws.get)
        print(f"\n  ⓘ 탐색 단계의 전류는 달랐다 ({hi} {raws[hi]:.2f} mA ↔ {lo} {raws[lo]:.2f} mA) — "
              f"채점은 같은 예산으로 환산했다.")
        print(f"     환산은 정확하지만(포락선이 전류에 1차 동차), 전극당 상한이 걸리는 지점이 달라")
        print(f"     **탐색이 고른 비율 자체는 방법마다 다를 수 있다.**")


def benchmark(lf, targets, methods=None, weights=(0.5, 0.5, 0.5), pctl=50, verbose=True,
              protocol=None):
    """
    targets : [(label, Target, direction_or_None), ...]
    methods : list of names (all of them by default).
    protocol: `protocol.Protocol` — **the yardstick**. Defaults to `protocol.FAIR`
              (total injected current fixed), which is the only rule under which methods
              may be compared. Pass `protocol.TIPLITE` to reproduce published values instead.
    Returns {label: {"n": ..., "methods": {m: {iso, dir, I_total, I_raw, scale, protocol, best}}}}.
    """
    from . import protocol as P
    #  ★A benchmark that measured nothing must not return successfully. `standard_targets()`
    #  used to hand back `[]` after silently failing to find its masks, and this loop then
    #  returned `{}` without printing a line — indistinguishable from a clean pass.
    targets = list(targets)
    if not targets:
        raise ValueError("benchmark() 에 표적이 하나도 없다. 빈 표는 통과가 아니라 결함이다 "
                         "— `standard_targets()` 의 반환값이나 `which=` 필터를 확인할 것.")
    prot = protocol if protocol is not None else P.FAIR
    methods = methods or list(METHODS)
    names = [e for e in lf.names if lf.has(e)]
    out = {}
    for label, target, direction in targets:
        n = np.asarray(direction, float) if direction is not None else principal_direction(lf, target)
        n = n / (np.linalg.norm(n) + 1e-30)
        res = {}
        for mname in methods:
            #  ★탐색과 채점이 **같은 블록 안**에 있어야 한다. 예전엔 채점만 맞추고 탐색은 방법마다
            #  달랐고(classic=config max_channel, dual·huang=총전류 강제, multichannel=`Imax=2.0`
            #  하드코딩), 처음 고칠 때는 탐색만 감싸 **채점이 규약 밖으로 새어** classic 의 총전류가
            #  총전류 규약인데도 1.72 mA 로 찍혔다(구조상 1.00 이어야 함). 전류를 다시 세는 코드가
            #  `evaluate` 안에도 있기 때문이다 — 한 블록으로 묶어야 그 누수가 없다.
            with P.use(prot):
                best = METHODS[mname](lf, target, names, n, weights, pctl)
                m = evaluate(lf, best, target, n, pctl=pctl, protocol=prot)
            m["best"] = best
            res[mname] = m
        out[label] = dict(n=[float(x) for x in n], protocol=prot.label, methods=res)
        if verbose:
            _print_target(label, direction is not None, res)
    return out


def standard_targets(lf, which=None):
    """Standard targets as [(label, Target, direction)].
    `which` filters by label substring, e.g. ["해마", "시상"].

    ⚠ The labels stay Korean on purpose: research scripts under `research/` call this with
    `which=["해마"]` and renaming them here would break those callers silently.

    ★Paths go through `C.inputs(name)` — which searches `inputs/`'s subfolders — and
    `C.MASKS_DIR`, **never** `os.path.join(C.INPUTS_DIR, name)`. They were the latter, written
    before the 2026-08-13 reorg moved these files into `inputs/geometry/` and
    `inputs/masks/mida/`. With a bare `except Exception: pass` around the load, every path
    missed and this returned **an empty list in silence**: `benchmark(lf, standard_targets(lf))`
    ran to completion, printed nothing and returned `{}` — a benchmark that looks like it
    passed having measured nothing. It stayed that way long enough for
    `research/tvb/_targets.py` to be written around it. So nothing here is optional any more:
    a missing or unreadable core file raises, a mask that does not fit the leadfield raises,
    and a `which` that matches nothing raises.

    ★This set is **MIDA's** — the masks are keyed to the human leadfield's voxel rows, so it
    refuses another head rather than guessing. That refusal matters more than it looks: the rat
    leadfield has 1,904,254 voxels against MIDA's 1,907,678, a 0.18% difference, so a
    mismatched mask does not conveniently blow up — it indexes a different brain.
    """
    import os
    import json
    from . import Target

    if C.MODEL_NAME != "human":
        raise ValueError(
            f"standard_targets() 는 MIDA(사람) 전용이다 — 현재 모델은 {C.MODEL_NAME!r}. "
            f"여기 마스크들은 사람 리드필드의 복셀 행에 묶여 있어 다른 머리에 쓰면 엉뚱한 "
            f"복셀을 가리킨다. 다른 머리는 (label, Target, direction) 세 쌍을 직접 만들어 "
            f"`benchmark()` 에 넘길 것 — 라벨 기반 표적 생성 예시는 `gui/app.py` 의 "
            f"`_label_targets()` 에 있다. (이 머리의 표준 표적 세트는 아직 정해진 바 없다.)")

    nvox = len(lf.coords())

    def _load(fn):
        p = C.inputs(fn)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"표준 표적 파일이 없다: {fn} (`config.inputs()` 가 {p} 로 풀었다). "
                f"이 함수는 하위폴더까지 뒤지므로, 없다는 건 입력 세트가 불완전하다는 뜻이다.")
        return np.load(p)

    def _fits(fn, arr):
        """★구조상 반드시 이 값 — 마스크 길이는 리드필드 복셀 수와 같아야 한다."""
        if arr.shape[-1] != nvox:
            raise ValueError(f"{fn}: 마스크 길이 {arr.shape[-1]} != 리드필드 복셀 {nvox} "
                             f"— 다른 머리의 마스크다.")
        return arr

    out = []
    HAX = _load("hipaxes1010.npz")
    HIP = _fits("hipmask1010.npy", _load("hipmask1010.npy"))
    out.append(("해마 L", Target.from_mask(lf, HIP[0], off_subsample=22000), HAX["nL"]))
    out.append(("해마 R", Target.from_mask(lf, HIP[1], off_subsample=22000), HAX["nR"]))
    TH = _fits("thalamus_mask.npy", _load("thalamus_mask.npy"))
    out.append(("시상 L", Target.from_mask(lf, TH[0], off_subsample=22000), None))

    #  The extra structures. Missing manifest = a reduced set, which changes what "the standard
    #  targets" means — degraded but still usable, so it is announced rather than raised (the
    #  same call `gui/app.py` makes when a head has no neural labels). A file the manifest
    #  promised and does not deliver **is** an error.
    mdir = os.path.join(C.MASKS_DIR, "mida")
    mpath = os.path.join(mdir, "manifest.json")
    if not os.path.exists(mpath):
        print(f"[benchmark] ⚠ {mpath} 가 없다 — 표준 표적이 해마·시상 {len(out)}개로 줄었다. "
              f"이 표는 축소된 세트의 결과다.", flush=True)
    else:
        for m in json.load(open(mpath, encoding="utf-8")):
            if m["id"] in ("hippocampus", "thalamus"):   # already above — avoid duplicates
                continue
            arr = _fits(m["file"], np.load(os.path.join(mdir, m["file"])))
            out.append((m.get("ko", m["id"]) + " L",
                        Target.from_mask(lf, arr[0], off_subsample=22000), None))

    if which:
        sel = [t for t in out if any(w in t[0] for w in which)]
        if not sel:
            raise ValueError(f"which={which!r} 가 아무 표적과도 안 맞는다 "
                             f"(라벨은 한국어다). 가능한 값: {[t[0] for t in out]}")
        out = sel
    return out
