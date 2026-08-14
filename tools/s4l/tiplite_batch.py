# -*- coding: utf-8 -*-
"""tiplite_batch.py — unattended solve of the tip.lite reference model for several electrodes
=====================================================================
Each electrode is a 156.8 MCell solve, so **memory is the only real constraint** (peak 73.6 GB
against 63.4 GB on this machine). If the Sim4Life application still holds the document, iSolve
dies with `Out of memory` while building the preconditioner. Hence **a fresh process per
stage**:

    1. configure and write the input file — fresh Python with Sim4Life loaded, then fully exit
    2. iSolve                             — has the memory to itself
    3. extract                            — fresh Python, no Sim4Life needed
    4. delete Output.h5                   — 11.7 GB each; leaving them fills the disk

⚠ Stage 1 ends with `os._exit(0)`: a clean shutdown takes minutes in DLL detach.
⚠ `CreateVoxels` deletes `_Results`, so stage 3 must precede stage 4 and stage 2 must follow
  stage 1 immediately.

Usage:
    python tiplite_batch.py AF3 AF4 AF7 ...      # electrodes already present are skipped
"""
import os
import subprocess
import sys
import time

PY = os.environ.get("TIP_S4L_PYTHON") or \
    os.path.join(os.path.dirname(REPO), ".venv-s4l", "Scripts", "python.exe")
#  Sim4Life solver. Override with TIP_ISOLVE if installed elsewhere.
ISOLVE = os.environ.get("TIP_ISOLVE") or \
    r"C:\Program Files\Sim4Life_9.6\Solvers\iSolve.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
#  Working directory for large intermediates. Override with TIP_SCRATCH.
SP = os.environ.get("TIP_SCRATCH") or os.path.join(REPO, "outputs", "scratch")
#  ⚠ The project path is overridable. If a Sim4Life MCP worker has the same `.smash` open,
#  the `doc.Save()` inside `CreateVoxels` dies on an **HDF5 lock**:
#      [Error] Can't create HDF5 file ... Unable to lock file, errno = 22
#      s4l_v1.exceptions.DocumentSaveError: Unable to save document
#  Killing the worker does not help — the server revives it within 6 seconds — so the reliable
#  fix is solving from a **copy** at a path the worker does not know about (this blocked T7 on
#  2026-08-12).
SMASH = os.environ.get("TIP_TIPLITE_SMASH") or os.path.join(SP, "tiplite_test.smash")
OUT = os.environ.get("TIP_TIPLITE_OUT") or os.path.join(SP, "tiplite_lf")
LOG = os.path.join(SP, "tiplite_batch.log")

SETUP = r'''
import os, sys
os.environ.setdefault("S4L_API_AUTO_INIT", "1")
sys.path.insert(0, r"{here}")
import tiplite_solve_one as T
sim, EL = T.prepare(r"{smash}")
T.set_electrodes(sim, EL, "{name}", "Cz")
inp, msg = T.write_input(sim)
sys.stderr.write("INPUT=" + inp + "\n" + msg + "\n")
sys.stderr.flush()
os._exit(0)          # a clean shutdown takes minutes in DLL detach
'''


def say(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def kill_stragglers():
    for exe in ("AresApplication.exe",):
        subprocess.run(["taskkill", "/F", "/T", "/IM", exe],
                       capture_output=True, text=True)


def one(name):
    dst = os.path.join(OUT, f"{name}.npy")
    if os.path.exists(dst):
        say(f"{name}: already present - skipping")
        return True
    t0 = time.time()

    r = subprocess.run([PY, "-c", SETUP.format(here=HERE, smash=SMASH, name=name)],
                       capture_output=True, text=True, errors="replace")
    inp = None
    for ln in (r.stderr or "").splitlines():
        if ln.startswith("INPUT="):
            inp = ln[6:].strip()
    if not inp or not os.path.exists(inp):
        say(f"{name}: configuration failed\n{(r.stderr or '')[-1500:]}")
        return False
    say(f"{name}: input file written ({time.time()-t0:.0f}s)")
    kill_stragglers()

    t1 = time.time()
    r = subprocess.run([ISOLVE, inp], cwd=os.path.dirname(inp),
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        say(f"{name}: iSolve failed with {r.returncode}\n{(r.stdout or '')[-1500:]}")
        return False
    say(f"{name}: solve complete ({(time.time()-t1)/60:.0f} min)")

    out_h5 = inp.replace("_Input.h5", "_Output.h5")
    r = subprocess.run([PY, os.path.join(HERE, "extract_tiplite_solve.py"), out_h5, name],
                       capture_output=True, text=True, errors="replace")
    if not os.path.exists(dst):
        say(f"{name}: extraction failed\n{(r.stdout or '')[-800:]}\n{(r.stderr or '')[-800:]}")
        return False
    inj = [l for l in (r.stdout or "").splitlines() if "I =" in l]
    try:
        os.remove(out_h5)
    except OSError:
        pass
    say(f"{name}: done · {(time.time()-t0)/60:.0f} min total · {inj[-1] if inj else ''}")
    return True


def main(names):
    os.makedirs(OUT, exist_ok=True)
    say(f"=== batch start · {len(names)} electrodes: {' '.join(names)}")
    ok, bad = 0, []
    for i, n in enumerate(names, 1):
        say(f"--- [{i}/{len(names)}] {n}")
        if one(n):
            ok += 1
        else:
            bad.append(n)
    say(f"=== END · succeeded {ok}/{len(names)}" + (f" · failed {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
