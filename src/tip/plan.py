# -*- coding: utf-8 -*-
"""
plan.py — the end-to-end entry point
=========================================
plan(lf, target, mode, allowed, ...) → optimal electrodes and currents, metrics, the Pareto
front, a report and a stimulator protocol.
The full workflow (target → optimise → report → protocol) in one function.

  target : a Target (build it with Target.from_sphere / from_mask / from_label)
  mode   : 'classic' (2 channels) | 'multichannel' (LP, strength) | 'gevd' (focality)
  allowed: subset of usable electrodes (None = all)
  method : for classic, 'brute' | 'nsga' | 'auto' (exhaustive at <= 14 electrodes, else NSGA)
"""
import os
import json


def plan(lf, target, mode="classic", allowed=None, direction=None,
         method="auto", weights=(0.5, 0.5, 0.5), Ecap=0.25, Imax=2.0,
         base_mA=1.0, f1=2000.0, f2=2010.0, report_dir=None, verbose=True):
    from .optimize.classic import optimize_classic
    from .optimize.nsga import optimize_nsga
    from .optimize.multichannel import optimize_currents, optimize_gevd
    from . import report as R

    names = [e for e in (allowed if allowed is not None else lf.names) if lf.has(e)]
    out = dict(mode=mode, target=target.name, allowed=names, n_allowed=len(names),
               f1=f1, f2=f2, df=abs(f2 - f1))

    if mode == "classic":
        if method == "auto":
            method = "brute" if len(names) <= 14 else "nsga"
        opt = optimize_classic if method == "brute" else optimize_nsga
        r = opt(lf, target, allowed=names, weights=weights, verbose=verbose)
        out.update(montages=r["montages"], n_pareto=r["n_pareto"], n_eval=r["n_eval"], method=method)
        best = r["best"]
    elif mode in ("multichannel", "lp"):
        if direction is None:
            raise ValueError("multichannel/LP mode needs a direction "
                             "(the stimulation axis, e.g. the hippocampal nL)")
        best = optimize_currents(lf, target, names, direction, Ecap=Ecap, Imax=Imax, verbose=verbose)
    elif mode == "gevd":
        if direction is None:
            raise ValueError("GEVD mode needs a direction (the stimulation axis)")
        best = optimize_gevd(lf, target, names, direction, Imax=Imax, verbose=verbose)
    else:
        raise ValueError(f"unknown mode: {mode}")

    out["best"] = best
    out["protocol"] = _protocol(best, f1, f2, base_mA)

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        fig = {}
        fig["montage"] = R.plot_montage(lf, best, target, out=os.path.join(report_dir, "montage.png"),
                                        title=f"{target.name} — {mode}")
        if "montages" in out:
            fig["pareto"] = R.plot_pareto(out, out=os.path.join(report_dir, "pareto.png"))
        try:
            fig["field"] = R.plot_field_slice(lf, best, target, out=os.path.join(report_dir, "field.png"))
        except Exception as e:
            fig["field_err"] = str(e)
        out["figures"] = fig
    return out


def _protocol(best, f1, f2, base_mA=None):
    """Channels, currents and frequencies in TIBS-R form. Currents follow **the protocol in
    force** (`protocol.current()`), so this block reports the currents the field and the
    metrics beside it were actually computed at.
    `base_mA` is kept for backward compatibility and ignored.

    ★Every branch reads the rule: classic through `channel_currents(r)`, dual through
    `dual_budget()`. Dual used the `DUAL_BUDGET` constant, which forces a **total**-current
    rule (ITOTAL/2 per system) whatever the protocol says. The GUI calls this under its
    `max_channel` rule (`gui/app.py` `run_job` and `field_for`), so the exported protocol
    under-reported the currents by 3.7~4.0x while `_used()` — in the same JSON response, off
    the same montage — reported the real ones. Measured on 해마 L, 2026-08-21:
    0.250/0.250/0.269/0.231 mA reported against 1.000/0.997/1.000/0.859 mA actually used.
    Under a total rule `dual_budget()` is exactly ITOTAL/2, so nothing computed under FAIR
    moves."""
    from .optimize.classic import channel_currents
    from .optimize.dualti import dual_budget
    if best.get("dual"):     # 4-channel dual TI: system A (2ch) + B (2ch), four frequencies.
                             # Per-system budget = the protocol's: ITOTAL/2 under a total rule,
                             # None under max_channel so the larger channel is pinned instead.
        fq = best.get("freqs") or [f1, f2, f1 + 500, f2 + 500]
        ch = []
        for si, sk in enumerate(("systemA", "systemB")):
            s = best[sk]; a, b = s["ch1"]; c, d = s["ch2"]; r = s.get("ratio", 1.0)
            i1, i2 = channel_currents(r, dual_budget())
            ch.append(dict(freq_Hz=fq[2 * si], currents_mA={a: round(i1, 3), b: round(-i1, 3)}))
            ch.append(dict(freq_Hz=fq[2 * si + 1], currents_mA={c: round(i2, 3), d: round(-i2, 3)}))
        return dict(channels=ch)
    if best.get("timemux"):  # time-mux: two channels per slot, switched by duty cycle
        ch = []
        for si, s in enumerate(best["slots"]):
            a, b = s["ch1"]; c, d = s["ch2"]
            i1, i2 = channel_currents(s.get("ratio", 1.0)); duty = s.get("duty", 0.0)
            ch.append(dict(freq_Hz=f1, slot=si + 1, duty=duty,
                           currents_mA={a: round(i1, 3), b: round(-i1, 3)}))
            ch.append(dict(freq_Hz=f2, slot=si + 1, duty=duty,
                           currents_mA={c: round(i2, 3), d: round(-i2, 3)}))
        return dict(channels=ch, timemux=True, n_slots=len(best["slots"]))
    if "currents" in best:   # distributed — the currents are already budget-normalised
        ch = [dict(freq_Hz=f1, currents_mA=dict(best["currents"]["ch0"])),
              dict(freq_Hz=f2, currents_mA=dict(best["currents"]["ch1"]))]
    else:                    # classic 2ch — normalised so total injected current = ITOTAL
        a, b = best["ch1"]; c, d = best["ch2"]; r = best.get("ratio", 1.0)
        i1, i2 = channel_currents(r)
        ch = [dict(freq_Hz=f1, currents_mA={a: round(i1, 3), b: round(-i1, 3)}),
              dict(freq_Hz=f2, currents_mA={c: round(i2, 3), d: round(-i2, 3)})]
    return dict(channels=ch)


def print_plan(out):
    b = out["best"]
    print("=" * 62)
    print(f"  TI stimulation plan — target {out['target']} | mode {out['mode']}"
          + (f" ({out.get('method')})" if out.get('method') else ""))
    print(f"  allowed electrodes {out['n_allowed']} | carriers {out['f1']:.0f}/{out['f2']:.0f} Hz "
          f"(df {out['df']:.0f} Hz)")
    print("-" * 62)
    if "currents" in b:
        print(f"  [distributed TI]  directional M1 {b['M1_dir']:.3f}  M2 {b['M2_dir']:.2f}  "
              f"M3 {b['M3_dir']:.1f}%   (focality bound {b.get('focality_bound', float('nan')):.2f})")
        print(f"             isotropic M1 {b['M1']:.3f} M2 {b['M2']:.2f} M3 {b['M3']:.1f}%")
    else:
        print(f"  [Classic]  {b['ch1'][0]}(+)/{b['ch1'][1]}(-) x {b['ch2'][0]}(+)/{b['ch2'][1]}(-)  "
              f"ratio {b['ratio']:.2f}")
        print(f"             M1 {b['M1']:.3f}  M2 {b['M2']:.2f}  M3 {b['M3']:.1f}%  WP {b.get('WP', 0):+.3f}"
              + (f"  | pareto {out['n_pareto']} (evaluated {out['n_eval']})" if 'n_pareto' in out else ""))
    print("-" * 62)
    print("  protocol (TIBS-R form):")
    for i, chn in enumerate(out["protocol"]["channels"], 1):
        cur = ", ".join(f"{e} {I:+.2f}mA" for e, I in chn["currents_mA"].items())
        print(f"    channel {i} @ {chn['freq_Hz']:.0f} Hz: {cur}")
    if "figures" in out:
        print("-" * 62)
        for k, v in out["figures"].items():
            print(f"  [{k}] {v}")
    print("=" * 62)


def export_protocol(out, path):
    """Save the plan as a JSON protocol."""
    rec = dict(target=out["target"], mode=out["mode"], f1=out["f1"], f2=out["f2"],
               protocol=out["protocol"])
    b = out["best"]
    rec["metrics"] = ({k: b[k] for k in ("M1_dir", "M2_dir", "M3_dir", "M1", "M2", "M3") if k in b}
                      if "currents" in b else {k: b[k] for k in ("M1", "M2", "M3", "WP") if k in b})
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return path
