# -*- coding: utf-8 -*-
"""s4l.py — Sim4Life backend runner (on-demand electrode solves, montage analysis)

Why subprocesses (`PIPELINE.md` D1)
-----------------------------------
No resident Sim4Life worker.
- **There is a single licence seat.** A resident worker holding it blocks everything else.
- One process per job gives **crash isolation** for free, and Sim4Life does crash.
- The current `worker.py` protocol has no request IDs, so two clients would get each
  other's replies.

So we launch the already-validated `add_electrodes.py` as-is and **follow its log**.
The GUI runs in the `tip` conda environment, which has no `s4l_v1`, so these tools must be
started with the **Sim4Life bundled Python** (`s4l_python()`).

⚠ Never run two at once — with one seat, both will stall or die. Guard with
  `JobStore.running(...)` before accepting a job.

Note on language: log-matching regexes and the progress strings shown in the UI are still
Korean, because they are paired with the Korean output of the scripts under `tools/`.
Changing one side alone would silently break progress reporting.
"""
import os
import re
import shutil
import subprocess
import sys
import time
import threading

from .. import config as C
from . import cache as CACHE

HERE = C.ROOT_DIR                     # repo root (config is the single source of truth for paths)
TOOLS = C.TOOLS_DIR
ADD = os.path.join(TOOLS, "s4l", "add_electrodes.py")
MONT = os.path.join(TOOLS, "s4l", "s4l_montage.py")
ANALYZE = os.path.join(TOOLS, "analyze", "montage_analyze.py")
#  ★The montage project **reuses a single slot.** Each montage costs 263 MB for the copy
#    plus 289 MB when saved, so a fresh project per job would reach several GB in no time.
#    With one licence seat there is never a concurrent job, so reuse is safe.
S4L_PROJECTS = os.environ.get("TIP_S4L_PROJECTS") or \
    os.path.join(os.path.dirname(HERE), "s4l_projects")
MONT_SMASH = os.path.join(S4L_PROJECTS, "montage_gui.smash")


def BASE_SMASH():
    """The **source model** each montage project is copied from. Its fingerprint goes into
    the cache key: adding electrodes or changing the grid changes the result for the same
    montage while the file name stays identical."""
    return os.environ.get("TIP_REBUILD_SMASH") or \
        os.path.join(S4L_PROJECTS, "mida1010_rebuild.smash")
#  ★The default output is the **deployed** directory. `rebuild_solve_batch.OUT` defaults to
#    a scratch folder; leaving it would point the GUI at something that can be wiped, and it
#    would then offer to re-solve electrodes that are already done ("14 missing"). Happened.
DEFAULT_OUT = os.path.join(HERE, "data", "leadfield_rebuild")

#  matches: "[08-13 00:10] [7/12] PO1 done · 132s · I=1.0577 mA · eta 11 min"
#  ⚠ These patterns are paired with the output of `tools/s4l/add_electrodes.py`.
#    Changing one side alone silently breaks progress reporting.
_PROG = re.compile(r"\[(\d+)/(\d+)\]\s+(\S+)\s+(done|failed)")
_TODO = re.compile(r"targets\s+(\d+):")


def s4l_python():
    """A Python that has `s4l_v1`. Same search order as `run_gui.bat`."""
    venv = os.path.join(os.path.dirname(HERE), ".venv-s4l")
    cands = [
        os.environ.get("TIP_S4L_PYTHON") or "",
        os.path.join(venv, "Scripts", "python.exe"),   # Windows venv layout
        os.path.join(venv, "bin", "python"),           # POSIX venv layout
        r"C:\Program Files\Sim4Life_9.6\Python\python.exe",
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise RuntimeError(
        "No Sim4Life Python found - an environment with `s4l_v1` is required. "
        "Sim4Life is Windows-only; on macOS and Linux the planner and the UI work, "
        "but montages cannot be re-solved. Set TIP_S4L_PYTHON to override the search.")


#  ── Cancellation ────────────────────────────────────────────────────
#  A job takes ~7 minutes, so starting one by mistake used to mean waiting it out. We kill
#  the whole process tree (iSolve is a child and dies with it). What matters most is that a
#  killed job **leaves no half-finished result in the cache** — `montage_run` checks for
#  cancellation before the cache write and bails out.
_PROCS = {}          # jid -> Popen of the running child
_CANCEL = set()      # jids for which cancellation was requested


class Cancelled(Exception):
    """The user cancelled. Must stay distinct from failure — this is not an error."""


def cancel(jid):
    """Cancel a job. Returns True if a process was actually killed."""
    _CANCEL.add(jid)
    p = _PROCS.get(jid)
    if p is not None and p.poll() is None:
        _kill_tree(p.pid)
        return True
    return False


def _check_cancel(jid):
    if jid in _CANCEL:
        raise Cancelled()


def _kill_tree(pid):
    """Tear down the process tree. Skipping this leaves a worker holding 2-3 GB and, worse,
    **the Sim4Life licence seat** (observed).

    Sim4Life is Windows-only, so in practice this runs under `taskkill`. The POSIX branch
    exists so the module behaves correctly on macOS and Linux, where the montage analysis
    stage can still be run on its own.
    """
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=30)
        else:
            #  Kill the children first, then the process itself. `pkill -P` is enough here
            #  because these trees are one level deep.
            subprocess.run(["pkill", "-9", "-P", str(pid)], capture_output=True, timeout=30)
            os.kill(pid, 9)
    except Exception:
        pass


def solve_electrodes(names, store, jid, out_dir=None, dry=False, timeout_s=7200):
    """Solve electrodes `names` and add them to the leadfield, reporting into `store`.

    names   : electrode names; an empty list means every unsolved one.
    dry     : only list the targets, do not solve (quick check).
    returns : the electrodes that succeeded.

    ★ **Never read the stdout pipe.** `add_electrodes.py` ends with `os._exit()` because a
      clean shutdown takes minutes in DLL detach — and at that moment a Sim4Life child still
      holds the pipe, so `for line in p.stdout` **never sees EOF** (observed: the solve
      finished but the job never went to done). The exit code is not trustworthy either;
      the Python process itself survives. **Tailing the log file and looking for the
      `=== END` marker** is the only reliable signal.
    """
    py = s4l_python()
    log = os.path.join(store.root, f"{jid}.log")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["S4L_API_AUTO_INIT"] = "1"
    env["TIP_ADDELEC_LOG"] = log
    env["TIP_REBUILD_OUT"] = out_dir or DEFAULT_OUT
    cmd = [py, "-u", ADD] + (["--dry"] if dry else []) + list(names)

    store.update(jid, stage="starting Sim4Life", pct=0.02)
    p = subprocess.Popen(cmd, cwd=HERE, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _PROCS[jid] = p                     # register so it can be cancelled
    store.update(jid, params=dict(store.get(jid)["params"], pid=p.pid, log=log))

    done, total, ok, bad, seen, t0 = 0, max(len(names), 1), [], [], 0, time.time()
    finished = False
    while not finished:
        time.sleep(1.0)
        try:
            lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            lines = []
        for line in lines[seen:]:
            store.append_log(jid, line)
            m = _TODO.search(line)
            if m:
                total = max(int(m.group(1)), 1)
                store.update(jid, stage=f"{total} target(s)", pct=0.05)
            m = _PROG.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                (ok if m.group(4) == "done" else bad).append(m.group(3))
                store.update(jid, pct=0.05 + 0.93 * done / max(total, 1),
                             stage=f"{m.group(3)} {m.group(4)} ({done}/{total})")
            if "=== END" in line:
                finished = True
        seen = len(lines)          # ★without this we re-read from the top every second and\n                                   #   `ok` accumulates duplicates
        if not finished and jid in _CANCEL:
            _kill_tree(p.pid)
            store.append_log(jid, "[orch] cancelled by the user")
            store.update(jid, done=True, pct=1.0, stage="cancelled", cancelled=True,
                         result={"ok": ok, "bad": bad})
            _CANCEL.discard(jid); _PROCS.pop(jid, None)
            return ok
        if not finished and p.poll() is not None and time.time() - t0 > 20:
            finished = True                     # died without leaving a log
        if not finished and time.time() - t0 > timeout_s:
            store.append_log(jid, f"[orch] exceeded {timeout_s}s - aborting")
            finished = True
    _kill_tree(p.pid)                           # do not leave the seat and memory held

    #  ⚠ The exit code lies (`os._exit`). **The success list in the log is the truth.**
    rc = p.poll()
    if bad:
        store.update(jid, done=True, pct=1.0, stage="partly failed",
                     error=f"failed {bad}", result={"ok": ok, "bad": bad})
    elif not ok and not dry:
        store.update(jid, done=True, pct=1.0, stage="failed",
                     error=f"no electrode succeeded (exit code {rc})", result={"ok": [], "bad": []})
    else:
        store.update(jid, done=True, pct=1.0, stage="done",
                     result={"ok": ok, "bad": [], "dry": dry})
    return ok


def spawn(names, store, out_dir=None, dry=False):
    """Create a job and run it on a background thread. Returns the job id."""
    jid = store.create("electrode", {"names": list(names), "dry": bool(dry)})

    def _run():
        try:
            solve_electrodes(names, store, jid, out_dir=out_dir, dry=dry)
        except Exception as e:
            import traceback
            traceback.print_exc()
            store.update(jid, done=True, error=f"{type(e).__name__}: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return jid


# ────────────────────────── Montage → Sim4Life ──────────────────────────
#  "Choose electrodes with the leadfield, analyse with Sim4Life." E inside the brain is
#  already exact by superposition (correlation 0.99997 against a re-solve). The value is in
#  the **whole head**: our leadfield holds only the 1,907,678 brain voxels (18% of the grid),
#  so safety numbers such as scalp current density are simply not there.

#  matches: "[s4l_montage] ch1 done · 138s · I=1.3514 mA · E (185, 254, 228, 3)"
#  ⚠ Paired with the output of `tools/s4l/s4l_montage.py` — change both together.
_MCH = re.compile(r"(ch[12])\s+done\s+·\s+(\d+)s\s+·\s+I=([\d.]+)\s*mA")


def _run_logged(cmd, env, log, store, jid, on_line, timeout_s=7200, kill=True):
    """Launch a subprocess and follow its **log file**. Returns the exit code.

    ★For the same reason as `solve_electrodes`, **the stdout pipe is never read.** Sim4Life
      scripts end with `os._exit()` (a clean shutdown takes minutes in DLL detach) and at
      that moment a Sim4Life child still holds the pipe, so `for line in p.stdout` never
      sees EOF (observed). We redirect to a file and treat `on_line` returning True as done.

    ⚠ Drop `-u` and the file redirect becomes block-buffered — progress then arrives all at
      once at the end.
    """
    _check_cancel(jid)                     # it may already have been cancelled
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=HERE, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
    _PROCS[jid] = p
    t0, seen, finished = time.time(), 0, False
    while not finished:
        time.sleep(1.0)
        if jid in _CANCEL:
            _kill_tree(p.pid); _PROCS.pop(jid, None)
            store.append_log(jid, "[orch] cancelled by the user - process tree cleaned up")
            raise Cancelled()
        try:
            lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            lines = []
        for line in lines[seen:]:
            store.append_log(jid, line)
            if on_line(line):
                finished = True
        seen = len(lines)              # ★without this we re-read from the top every second
        if not finished and p.poll() is not None and time.time() - t0 > 20:
            finished = True            # died without leaving a log
        if not finished and time.time() - t0 > timeout_s:
            store.append_log(jid, f"[orch] exceeded {timeout_s}s - aborting")
            finished = True
    rc = p.poll()
    if kill:
        _kill_tree(p.pid)              # do not leave the seat and memory held
    _PROCS.pop(jid, None)
    return rc


def montage_dir(store, jid):
    """The job\\'s result folder, or None."""
    d = os.path.join(os.path.dirname(store.root), "montage", jid)
    return d if os.path.isdir(d) else None


def montage_result(store, jid):
    """Return `analysis.json`; if it does not exist yet, return the job state instead."""
    import json
    d = montage_dir(store, jid)
    p = os.path.join(d, "analysis.json") if d else None
    if p and os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return store.get(jid) or {"error": "no job"}


#  ★What the cache keeps. `ch?_Ebrain.npy` (22.9 MB per channel) is included because
#    **the target is deliberately not part of the cache key** — the field does not depend on
#    it. Keeping brain-voxel E means switching targets recomputes the metrics in seconds
#    instead of re-solving for 7 minutes. The full grid (128 MB per channel) is discarded
#    once the per-tissue statistics and slices have been produced.
CACHE_FILES = ("analysis.json", "montage.json", "inj.json",
               "ch1_Ebrain.npy", "ch2_Ebrain.npy",
               "slice_sagittal.png", "slice_coronal.png", "slice_axial.png")


def _drop_ebrain(out_dir, keep=False):
    """Delete brain-voxel E from the job folder. **The canonical copy lives in the cache**;
    the job folder only ever held a duplicate. Without this, 46 MB accumulates per job —
    458 MB across ten jobs, measured."""
    if keep:
        return 0
    n = 0
    for ch in ("ch1", "ch2"):
        p = os.path.join(out_dir, f"{ch}_Ebrain.npy")
        try:
            n += os.path.getsize(p); os.remove(p)
        except OSError:
            pass
    return n


def _target_step(out_dir, target, lf_metrics, store, jid):
    """Target statistics, M1/M2/M3 from the Sim4Life solution, and the per-structure
    breakdown. Only when a target is given.

    This step runs **identically on a cache hit and on a fresh solve** — the target is not
    part of the key, so a cached result still has to be evaluated against the current target.
    """
    import json
    import numpy as np
    if not target:
        return None
    tp = os.path.join(out_dir, "target.npz")
    np.savez(tp, target_idx=np.asarray(target["target_idx"], np.int64),
             off_idx=np.asarray(target["off_idx"], np.int64),
             name=str(target.get("name", "target")),
             off_def=str(target.get("off_def", "?")))
    log = os.path.join(store.root, f"{jid}.target.log")
    #  ★`--target-only` is mandatory. The full analysis already ran right after the solve,
    #    and on a cache hit the full-grid E does not exist at all (only brain-voxel E is
    #    kept in the cache).
    cmd = [sys.executable, "-u", ANALYZE, out_dir,
           os.path.join(out_dir, "montage.json"), "--target-only", "--target", tp]
    if lf_metrics:
        cmd += ["--lfmetrics", json.dumps(lf_metrics)]
    _run_logged(cmd, dict(os.environ, PYTHONIOENCODING="utf-8"), log, store, jid,
                lambda ln: "=== END" in ln, timeout_s=900, kill=False)
    p = os.path.join(out_dir, "target.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def montage_run(ch1, ch2, store, jid, ratio=1.0, itotal=2.0,
                keep_raw=False, timeout_s=5400, use_cache=True,
                target=None, lf_metrics=None):
    """Export the montage as a Sim4Life project, **solve it**, and run the whole-head
    analysis.

    ch1 · ch2 : (anode, cathode) electrode-name pairs
    returns   : the contents of analysis.json

    It splits into two stages **because the Python differs**:
      1. Sim4Life Python: export, solve, extract the full grid (`h5py` only exists here)
      2. tip Python: TI envelope, per-tissue statistics, slice rendering (`matplotlib` only
         exists here)

    ★Asking for the same analysis again **returns it from the cache immediately** instead of
      spending another 7 minutes. The key hashes the electrode pairs, currents, leadfield
      set, model-file fingerprint and algorithm version, so any difference forces a re-solve
      (`orch/cache.py`).
    """
    import json

    t_start = time.time()
    out_dir = os.path.join(os.path.dirname(store.root), "montage", jid)
    os.makedirs(out_dir, exist_ok=True)

    key, spec = CACHE.montage_key(ch1, ch2, ratio, itotal, smash=BASE_SMASH())
    store.update(jid, params=dict(store.get(jid)["params"], cache_key=key))
    hit = CACHE.lookup("montage_s4l", key) if use_cache else None
    if hit:
        rec = CACHE.info("montage_s4l", key) or {}
        for f in os.listdir(hit):
            if f != "_cache.json":
                shutil.copy2(os.path.join(hit, f), os.path.join(out_dir, f))
        res = json.load(open(os.path.join(out_dir, "analysis.json"), encoding="utf-8"))
        res["cached"] = True
        res["cached_from"] = rec.get("created_str")
        res["cache_key"] = key
        res["_dir"] = out_dir
        store.append_log(jid, f"[cache hit] {key} - reusing the result computed at {rec.get('created_str')}")
        store.append_log(jid, "  same electrodes, currents, leadfield and model, so no re-solve is needed (~7 min saved)")
        #  The target is not in the key, so evaluate against **the current target**\n        #  (brain-voxel E is in the cache, so this takes seconds)
        if target:
            store.update(jid, stage="cache hit - computing target metrics", pct=0.7)
            res["target"] = _target_step(out_dir, target, lf_metrics, store, jid)
        _drop_ebrain(out_dir, keep=keep_raw)      # the canonical copy is in the cache
        store.update(jid, done=True, pct=1.0, stage="done (cache hit)", result=res)
        return res
    log1 = os.path.join(store.root, f"{jid}.log")
    log2 = os.path.join(store.root, f"{jid}.analyze.log")
    store.update(jid, params=dict(store.get(jid)["params"], out_dir=out_dir, log=log1))

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["S4L_API_AUTO_INIT"] = "1"

    # ── Stage 1 — Sim4Life: export and solve ──
    def _on1(line):
        if "] project " in line:
            store.update(jid, stage="project created - starting solve", pct=0.40)
        m = _MCH.search(line)
        if m:
            store.update(jid, pct=0.40 + 0.25 * (1 if m.group(1) == "ch1" else 2),
                         stage=f"{m.group(1)} solved ({m.group(2)}s · I={m.group(3)} mA)")
        return "=== END" in line

    store.update(jid, stage="starting Sim4Life - creating the project (~6 min)", pct=0.02)
    cmd = [s4l_python(), "-u", MONT, "export", MONT_SMASH,
           ch1[0], ch1[1], ch2[0], ch2[1],
           "--ratio", str(ratio), "--itotal", str(itotal), "--solve", out_dir]
    _run_logged(cmd, env, log1, store, jid, _on1, timeout_s=timeout_s)

    _check_cancel(jid)      # bail out if cancelled mid-solve, so no half result is cached
    #  ⚠ The exit code lies (`os._exit`). **The produced files are the truth.**
    need = ["ch1_E1V.npy", "ch2_E1V.npy", "inj.json"]
    miss = [f for f in need if not os.path.exists(os.path.join(out_dir, f))]
    if miss:
        raise RuntimeError(f"solve produced nothing: {miss} - log {log1}")

    #  Copy the metadata into the job folder so it is **self-contained** — the project slot\n    #  gets overwritten by the next job.
    meta = os.path.join(out_dir, "montage.json")
    shutil.copy2(MONT_SMASH.replace(".smash", "_montage.json"), meta)

    # ── Stage 2 — tip: envelope, statistics, slices ──
    store.update(jid, stage="analysing - TI envelope, per-tissue stats, slices", pct=0.90)
    _run_logged([sys.executable, "-u", ANALYZE, out_dir, meta],
                dict(os.environ, PYTHONIOENCODING="utf-8"), log2, store, jid,
                lambda ln: "=== END" in ln, timeout_s=1800, kill=False)

    res_p = os.path.join(out_dir, "analysis.json")
    if not os.path.exists(res_p):
        raise RuntimeError(f"analysis failed - log {log2}")
    res = json.load(open(res_p, encoding="utf-8"))

    #  ★Drop the big files. `ch?_E1V.npy` is 128 MB each, i.e. 300 MB per job. What stays is
    #    analysis.json, three slice PNGs, montage.json and inj.json — under 1 MB together.
    if not keep_raw:
        for f in ("ch1_E1V.npy", "ch2_E1V.npy", "sigma.npy"):
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass
    #  ★Write to the cache — the next identical montage is answered from here
    have = [f for f in CACHE_FILES if os.path.exists(os.path.join(out_dir, f))]
    cdir = CACHE.store("montage_s4l", key, out_dir, spec, files=have,
                       meta={"solve_seconds": round(time.time() - t_start)})
    store.append_log(jid, f"[cache store] {key} -> {cdir}")

    if target:
        store.update(jid, stage="computing target metrics", pct=0.96)
        res["target"] = _target_step(out_dir, target, lf_metrics, store, jid)
    _drop_ebrain(out_dir, keep=keep_raw)          # the canonical copy is in the cache

    res["cached"] = False
    res["cache_key"] = key
    res["_dir"] = out_dir
    store.update(jid, done=True, pct=1.0, stage="done", result=res)
    return res


def spawn_montage(ch1, ch2, store, ratio=1.0, itotal=2.0, keep_raw=False,
                  use_cache=True, target=None, lf_metrics=None):
    """Create a montage job and run it in the background. Returns the job id.
    Seconds on a cache hit, about 7 minutes otherwise.

    target : `{"target_idx": [...], "off_idx": [...], "name":…, "off_def":…}`.
             Taken from **the very Target** the GUI's `build_target` produced. Two copies of
             target-resolution logic would let the metrics diverge silently.
    """
    jid = store.create("montage", {"ch1": list(ch1), "ch2": list(ch2),
                                   "ratio": float(ratio), "itotal": float(itotal),
                                   "use_cache": bool(use_cache),
                                   "target": (target or {}).get("name")})

    def _run():
        try:
            montage_run(ch1, ch2, store, jid, ratio=ratio, itotal=itotal,
                        keep_raw=keep_raw, use_cache=use_cache,
                        target=target, lf_metrics=lf_metrics)
        except Cancelled:
            #  Cancellation is **not an error.** Filling `error` would show it as a red\n            #  failure in the UI.
            store.update(jid, done=True, pct=1.0, stage="cancelled",
                         cancelled=True, error=None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            store.update(jid, done=True, pct=1.0, stage="failed",
                         error=f"{type(e).__name__}: {e}")
        finally:
            _CANCEL.discard(jid)
            _PROCS.pop(jid, None)

    threading.Thread(target=_run, daemon=True).start()
    return jid
