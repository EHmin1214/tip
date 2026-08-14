# -*- coding: utf-8 -*-
"""add_electrodes.py — unattended fill of the 10-10 extension electrodes (71 -> 85)
================================================================================
Of the 85 vertices `ViP.Create1010System(..., add_outer_ring=True)` returns, this picks the
ones missing from the rebuilt leadfield, **places an electrode, solves it** and adds it to the
leadfield.
The rebuilt grid is 10.7 MCells, so budget ~150-210 s and a few GB of memory per electrode.

    python add_electrodes.py --list          # list the vertices only, no solve
    python add_electrodes.py --dry           # show the targets and their spacing
    python add_electrodes.py                 # everything not yet solved
    python add_electrodes.py N1 N2           # only the named ones

⚠ **Never run this alongside an MCP worker or another iSolve** — there is one QS_SOLVER seat.
   If a tip.lite reference solve is running, memory reaches 73 GB and the two kill each other.

Produces `<OUT>/{electrode}.npy` (**normalised to 1 mA**) and updates `positions.json`.
`OUT` defaults to `rebuild_solve_batch.OUT`. To write straight into the deployed set, set
`TIP_REBUILD_OUT=.../inputs/leadfield/leadfield_rebuild`.
"""
import json
import os
import sys
import time

os.environ.setdefault("S4L_API_AUTO_INIT", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

LOG = os.environ.get("TIP_ADDELEC_LOG") or os.path.join(HERE, "add_electrodes.log")
MIN_GAP_MM = 10.0        # the original set's minimum inter-electrode distance is 21.5 mm;
                         # anything much closer than this suggests an overlap


def say(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def main(argv):
    import s4l_v1 as s4l
    import XCoreModeling as xm
    import electrode_add as A
    import rebuild_solve_batch as R

    only = [a for a in argv if not a.startswith("--")]
    list_only = "--list" in argv
    dry = "--dry" in argv

    s4l.document.Open(R.SMASH)
    sim = list(s4l.document.AllSimulations)[0]
    os.makedirs(R.OUT, exist_ok=True)

    pos = A.vertices_1010(add_outer_ring=True)
    say(f"{len(pos)} vertices · output {R.OUT}")
    if list_only:
        for n in sorted(pos):
            say(f"    {n:<8} {[round(x, 2) for x in pos[n]]}")
        say(f"=== END · listed only, {len(pos)}")   # ★required terminator - orch detects completion by this
        os._exit(0)

    todo = A.missing_names(pos, R.OUT)
    if only:
        unknown = [n for n in only if n not in pos]
        if unknown:
            say(f"★names not among the vertices: {unknown}")
            os._exit(1)
        todo = [n for n in only if n in todo] or only

    say(f"already solved {len(pos) - len(todo) - 1} · targets {len(todo)}: {' '.join(todo)}")
    for n in todo:
        g = A.min_gap(pos, n, list(pos))
        flag = "  ⚠possible overlap" if g < MIN_GAP_MM else ""
        say(f"    {n:<8} {[round(x, 2) for x in pos[n]]}  min gap {g:.1f} mm{flag}")
    if dry:
        say(f"=== END · dry run only, {len(todo)}")   # ★required terminator (orch completion)
        os._exit(0)
    if not todo:
        say("=== END · nothing to do (everything already solved)"); os._exit(0)

    t00 = time.time(); ok, bad = [], []
    for i, n in enumerate(todo, 1):
        t0 = time.time()
        try:
            _, I, el = A.add_and_solve(sim, xm, n, pos[n])
            ok.append(n)
            eta = (time.time() - t00) / len(ok) * (len(todo) - i) / 60.0
            say(f"[{i}/{len(todo)}] {n} done · {time.time()-t0:.0f}s · "
                f"I={None if I is None else round(I*1e3, 4)} mA · eta {eta:.0f} min")
        except Exception as ex:
            bad.append(n)
            say(f"[{i}/{len(todo)}] {n} failed: {type(ex).__name__}: {ex}")

    if ok:
        n_pos = A.update_positions(pos, ok, R.OUT)
        say(f"positions.json updated -> {n_pos} entries")
    say(f"=== END · succeeded {len(ok)}/{len(todo)} · {(time.time()-t00)/60:.0f} min"
        + (f" · failed {bad}" if bad else ""))
    os._exit(0 if not bad else 1)      # a clean shutdown takes minutes in DLL detach


if __name__ == "__main__":
    main(sys.argv[1:])
