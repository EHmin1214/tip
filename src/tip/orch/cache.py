# -*- coding: utf-8 -*-
"""cache.py — content-addressed cache. **The same analysis is never computed twice.**

Why content, not name
---------------------
Conclusions in this project have been overturned several times, and every time the argument
came down to *"which inputs produced that number"*. A cache keyed on a file name would hand
back a stale result after the inputs changed — giving a wrong answer **silently**, which is
worse than having no cache at all.

So the key is a hash of **every** input. Change one thing and it lands in a different slot.

    key = sha256(normalised input JSON)[:16]
    outputs/cache/<kind>/<key>/  ← result files + _cache.json (what produced them)

What goes into a montage key
----------------------------
- both electrode pairs and their **polarity** (F9→T10 is not T10→F9)
- current ratio and total budget (these set the field magnitude)
- the **leadfield set name** — `rebuild` and `leadfieldF` differ by 2.3× per unit current
- **size and mtime of the model project file** — adding electrodes or changing the grid
  changes the result while the file name stays put. Without this, a change like "14 new
  electrodes" would go unnoticed by the cache.
- the version of the computing code (`ALGO_VERSION`) — if we change a metric definition,
  older results must be discarded

⚠ **A cache must never change the answer.** Hit or miss, the value is identical; a hit is
  merely faster. That is why a hit is tagged with `cached=True` and the original timestamp —
  so it is always visible where a number came from.
"""
import hashlib
import json
import os
import shutil
import time

from .. import config as C

#  Bump this whenever the metric or envelope computation changes — it invalidates old slots.
ALGO_VERSION = "2026-08-13.1"


def _stamp(path):
    """Size and mtime of a file. Hashing the contents would mean reading 263 MB; editing the
    project always changes at least one of these two."""
    try:
        st = os.stat(path)
        return {"size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return None


def montage_key(ch1, ch2, ratio, itotal, smash=None, leadfield_set=None):
    """Return `(key, spec)` for a montage analysis.

    The spec is returned alongside so that "what was this result?" can be answered later
    **without inverting the hash**. It is stored inside the cache slot.
    """
    spec = {
        "kind": "montage_s4l",
        "algo": ALGO_VERSION,
        #  Polarity is meaningful, so pairs are not sorted internally. Swapping channel 1
        #  and 2 is the same stimulation (the carriers just trade places), so the two pairs
        #  are sorted against each other to land on one key.
        "channels": sorted([list(ch1), list(ch2)]),
        "ratio": round(float(ratio), 6),
        "itotal": round(float(itotal), 6),
        "leadfield_set": leadfield_set or getattr(C, "LEADFIELD_SET", "?"),
        "model": _stamp(smash) if smash else None,
    }
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16], spec


def slot(kind, key):
    return os.path.join(C.CACHE_DIR, kind, key)


def lookup(kind, key, require=("analysis.json",)):
    """Path to the cache slot on a hit, otherwise None.

    `require` is checked because a job that died midway can leave an empty directory — the
    directory existing does not mean the result does.
    """
    d = slot(kind, key)
    if all(os.path.exists(os.path.join(d, f)) for f in require):
        return d
    return None


def store(kind, key, src_dir, spec, files=None, meta=None):
    """Copy results from `src_dir` into the cache slot; return the slot path.

    Copies rather than moves: the job folder is the record of that job, while the cache is a
    derivative that may be deleted at any time. Keeping them independent matters.
    """
    d = slot(kind, key)
    os.makedirs(d, exist_ok=True)
    names = files if files is not None else os.listdir(src_dir)
    for f in names:
        s = os.path.join(src_dir, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(d, f))
    rec = {"key": key, "spec": spec, "created": time.time(),
           "created_str": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source_job_dir": src_dir, "files": sorted(os.listdir(d))}
    if meta:
        rec.update(meta)
    json.dump(rec, open(os.path.join(d, "_cache.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return d


def info(kind, key):
    """The slot's `_cache.json`, or None."""
    p = os.path.join(slot(kind, key), "_cache.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


def listing(kind=None, limit=100):
    """Cached analyses, newest first. Used by the UI to show what has already been run."""
    root = C.CACHE_DIR
    out = []
    if not os.path.isdir(root):
        return out
    for k in (os.listdir(root) if kind is None else [kind]):
        kd = os.path.join(root, k)
        if not os.path.isdir(kd):
            continue
        for key in os.listdir(kd):
            rec = info(k, key)
            if rec:
                rec["kind"] = k
                out.append(rec)
    return sorted(out, key=lambda r: -r.get("created", 0))[:limit]
