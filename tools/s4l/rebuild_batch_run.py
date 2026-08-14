# -*- coding: utf-8 -*-
"""rebuild_batch_run.py — unattended run of the electrode leadfield on the rebuilt model
====================================================================
The rebuilt model is 10.7 MCells, so each solve is light (about 150-210 s per electrode, a few
GB of memory). Unlike the reference model (156.8 MCells, 1-2.5 hours per electrode) there is
**no need to cycle processes** — the document is opened once and the run goes straight through.

    python rebuild_batch_run.py                 # every unsolved electrode in enames1010
    python rebuild_batch_run.py AFz C1 C2       # only the named ones

⚠ **Never run this alongside an MCP worker** — two Sim4Life instances contend for the same
  licence seat.
"""
import json
import os
import sys
import time

os.environ.setdefault("S4L_API_AUTO_INIT", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

#  Working directory for large intermediates. Override with TIP_SCRATCH.
SP = os.environ.get("TIP_SCRATCH") or os.path.join(REPO, "outputs", "scratch")
LOG = os.environ.get("TIP_REBUILD_LOG") or os.path.join(SP, "rebuild_batch.log")


def say(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    sys.stderr.write(line + "\n"); sys.stderr.flush()


def main(argv):
    import s4l_v1 as s4l
    import XCoreModeling as xm
    import rebuild_solve_batch as R

    s4l.document.Open(R.SMASH)
    sim = list(s4l.document.AllSimulations)[0]
    os.makedirs(R.OUT, exist_ok=True)
    done = {os.path.splitext(f)[0].lower() for f in os.listdir(R.OUT)}
    if argv:
        todo = [n for n in argv if n.lower() not in done]
    else:
        # The standard 61 (minus Cz) plus 10 lower-ring = 70. The lower ring is absent from
        # `enames1010`, so it is appended here.
        enames = json.load(open(os.path.join(R.DD, "enames1010.json")))
        ring = ["F9", "F10", "FT9", "FT10", "T9", "T10", "TP9", "TP10", "P9", "P10"]
        todo = [e for e in list(enames) + ring
                if e != "Cz" and e.lower() not in done]

    say(f"=== rebuild batch start · done {len(done)} · remaining {len(todo)}: {' '.join(todo)}")
    t00 = time.time(); ok, bad = 0, []
    for i, n in enumerate(todo, 1):
        t0 = time.time()
        try:
            _, I, el = R.solve_one(sim, xm, n)
            ok += 1
            eta = (time.time() - t00) / ok * (len(todo) - i) / 60.0
            say(f"[{i}/{len(todo)}] {n} done · {time.time()-t0:.0f}s · "
                f"I={None if I is None else round(I*1e3, 4)} mA · eta {eta:.0f} min")
        except Exception as ex:
            bad.append(n)
            say(f"[{i}/{len(todo)}] {n} failed: {type(ex).__name__}: {ex}")
    say(f"=== END · succeeded {ok}/{len(todo)} · {(time.time()-t00)/60:.0f} min"
        + (f" · failed {bad}" if bad else ""))
    os._exit(0 if not bad else 1)          # a clean shutdown takes minutes in DLL detach


if __name__ == "__main__":
    main(sys.argv[1:])
