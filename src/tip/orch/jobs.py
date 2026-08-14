# -*- coding: utf-8 -*-
"""jobs.py — a job registry that survives on disk

The in-memory `JOBS` dict in `app.py` disappears when the server restarts. Solving an
electrode (~2 min each) or regenerating a leadfield (hours) has to **outlive the browser
session**, so those move to disk.

One file per job (`outputs/jobs/<id>.json`). Putting several jobs in one file would let
concurrent writes overwrite each other — jobs are updated from threads, so that risk is real.

State contract
--------------
    id · kind · created · updated · pct(0-1) · stage · done · error · result · params
`done=True` with no `error` means success. `pct` is for display only and need not be exact.
"""
import json
import os
import threading
import time
import uuid

_LOCK = threading.RLock()


class JobStore:
    """Thread-safe job store: in-memory cache plus disk persistence."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._cache = {}
        self._load_all()

    # ---------- internal ----------
    def _path(self, jid):
        return os.path.join(self.root, f"{jid}.json")

    def _load_all(self):
        for f in os.listdir(self.root):
            if not f.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(self.root, f), encoding="utf-8"))
                self._cache[d["id"]] = d
            except Exception:
                pass                      # one corrupt file must not block startup
        # Jobs that were running when the server died cannot be resumed — do not leave
        # ghost "running" entries behind.
        for d in self._cache.values():
            if not d.get("done"):
                d["done"] = True
                d["error"] = d.get("error") or "interrupted by a server restart"
                self._write(d)

    def _write(self, d):
        tmp = self._path(d["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._path(d["id"]))      # atomic — readers never see a half file

    # ---------- public ----------
    def create(self, kind, params=None, stage="queued"):
        jid = uuid.uuid4().hex[:12]
        d = {"id": jid, "kind": kind, "params": params or {},
             "created": time.time(), "updated": time.time(),
             "pct": 0.0, "stage": stage, "done": False,
             "error": None, "result": None, "log": []}
        with _LOCK:
            self._cache[jid] = d
            self._write(d)
        return jid

    def update(self, jid, **kw):
        with _LOCK:
            d = self._cache.get(jid)
            if d is None:
                return None
            d.update(kw)
            d["updated"] = time.time()
            self._write(d)
            return dict(d)

    def append_log(self, jid, line, keep=200):
        with _LOCK:
            d = self._cache.get(jid)
            if d is None:
                return
            d.setdefault("log", []).append(line)
            if len(d["log"]) > keep:
                d["log"] = d["log"][-keep:]
            d["updated"] = time.time()
            self._write(d)

    def get(self, jid):
        with _LOCK:
            d = self._cache.get(jid)
            return dict(d) if d else None

    def list(self, kind=None, limit=50):
        with _LOCK:
            ds = [dict(d) for d in self._cache.values()
                  if kind is None or d.get("kind") == kind]
        return sorted(ds, key=lambda d: -d["created"])[:limit]

    def running(self, kind=None):
        """Jobs still running. **There is one licence seat, so Sim4Life jobs cannot overlap** —
        check this before accepting a new one."""
        return [d for d in self.list(kind) if not d.get("done")]
