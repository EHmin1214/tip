# -*- coding: utf-8 -*-
"""
leadfield.py — the leadfield data layer (floating basis)
=================================================
An adapter that reuses the most expensive artefact in the pipeline: the EM simulation results.

Conventions (identical to the validated `extract_eval.py`):
  - leadfieldF/M{j}.npy : the 1 V floating-basis E-field of electrode j (index into `enames`),
                          shape (N,3) float32
  - M18 (= Cz) is excluded, leaving 60 files. The file number j is the `enames` index.
  - unitnorm.json[j]    : normalisation to 1 mA injection.
                          Mn_j = M_j * unitnorm_j (the 1 mA field, V/m)
  - any montage         = sum_i I_i · Mn_i, with I in mA and sum I_i = 0
  - an electrode pair (A,B) driven at I mA: I·(Mn_A - Mn_B)

Coordinates: bmask1010.npy[r] gives grid indices (i,j,k) → (cx[i], cy[j], cz[k]) in mm.
      ★Always obtain coordinates through this path. A past coordinate-decoding bug
      invalidated a set of results; this is the guard against a repeat.
"""
import os, re, glob, json
import numpy as np
from . import config as C


class LeadField:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or C.DATA_DIR
        lf_dir = C.LEADFIELD_DIR

        self.enames = json.load(open(C.inputs("enames1010.json")))  # 61
        un = json.load(open(C.inputs("unitnorm.json")))             # dict, 60

        # Files are in column order: the glob sort (M00, M01, ..., M60 with M18 absent) and
        # sorted(unitnorm.keys(), key=int) give the same ordering (asserted in extract_eval.py).
        self.files = sorted(glob.glob(os.path.join(lf_dir, "M*.npy")))
        assert len(self.files) == C.N_ELEC, f"leadfield has {len(self.files)} files (expected 60)"
        keys = sorted(un.keys(), key=int)
        # Per-column normalisation to 1 mA. C.LEADFIELD_AMP_FIX (= 0.5) corrects for unitnorm
        # having been built against **half the current amplitude**; the derivation is in the
        # config.py comment. Set it to 1.0 to reproduce the historical numbers.
        self.scale = np.array([un[k] for k in keys], dtype=np.float64) * C.LEADFIELD_AMP_FIX

        # file number j (= enames index) → electrode name; column i maps to that name
        self.col_name, self.col_j = [], []
        for f in self.files:
            j = int(re.findall(r"M(\d+)\.npy$", os.path.basename(f))[0])
            self.col_j.append(j)
            self.col_name.append(self.enames[j])
        self.name2col = {n: i for i, n in enumerate(self.col_name)}

        # assets for coordinate decoding
        self.bmask = np.load(C.inputs("bmask1010.npy"))  # (N,3) int
        g = np.load(C.inputs("gaxes1010.npz"))
        self.cx, self.cy, self.cz = g["cx"], g["cy"], g["cz"]
        self.N = self.bmask.shape[0]

        self._cache = {}  # normalised field per electrode name, loaded on demand

        # Unified electrode registry: the standard 61 plus the lower ring extras if present.
        # name -> (file path, unitnorm)
        self.reg = {self.col_name[i]: (self.files[i], float(self.scale[i]))
                    for i in range(len(self.files))}
        self._load_extras()
        self.set_name = "original"
        if getattr(C, "LEADFIELD_SET", "original") == "rebuild":
            self._use_rebuild()
        self.names = list(self.reg)

        # electrode coordinates, for native constraints and visualisation (standard 61 plus
        # the lower ring extras)
        self.pos = {}
        try:
            self.pos.update(json.load(open(C.inputs("pos1010.json"))))
        except Exception:
            pass
        sub = ("leadfield_rebuild" if self.set_name == "rebuild" else "leadfield_extra")
        ep = os.path.join(C.LEADFIELD_ROOT, sub, "positions.json")
        if os.path.exists(ep):
            try:
                for k, v in json.load(open(ep)).items():
                    self.pos[k] = v if isinstance(v, (list, tuple)) else \
                        (v.get("pos") or v.get("position") or v.get("xyz"))
            except Exception:
                pass

    def _use_rebuild(self):
        """Swap the whole electrode pool for the rebuilt set (`config.LEADFIELD_SET`).

        ⚠ The rebuilt `.npy` files are **already normalised to 1 mA**, in (N,3) V/m. Unlike the
          old set they must not be multiplied by `unitnorm x LEADFIELD_AMP_FIX`, hence
          scale = 1.0.
        If the folder is missing we fall back to the old set silently — the tool has to run on
        a machine without the data.
        """
        d = getattr(C, "LEADFIELD_REBUILD_DIR", "")
        fs = sorted(glob.glob(os.path.join(d, "*.npy"))) if os.path.isdir(d) else []
        if not fs:
            return
        self.reg = {os.path.splitext(os.path.basename(f))[0]: (f, 1.0) for f in fs}
        self.set_name = "rebuild"

    def _load_extras(self):
        """Add `leadfield_extra/{name}.npy` (raw 1 V) plus `inj_current.json` to the pool."""
        exdir = os.path.join(C.LEADFIELD_ROOT, "leadfield_extra")
        injp = os.path.join(exdir, "inj_current.json")
        if not (os.path.isdir(exdir) and os.path.exists(injp)):
            return
        inj = json.load(open(injp))
        for f in sorted(glob.glob(os.path.join(exdir, "*.npy"))):
            nm = os.path.splitext(os.path.basename(f))[0]
            rec = inj.get(nm)
            if rec is None:
                continue
            un = rec["unitnorm"] if isinstance(rec, dict) else float(rec)
            self.reg[nm] = (f, float(un) * C.LEADFIELD_AMP_FIX)   # amplitude correction (config.py)

    # ---- coordinates ----
    def coords(self, idx=None):
        """Voxel row indices `idx` → (M,3) coordinates in mm. `idx=None` returns all."""
        b = self.bmask if idx is None else self.bmask[idx]
        return np.stack([self.cx[b[:, 0]], self.cy[b[:, 1]], self.cz[b[:, 2]]], axis=1)

    # ---- electrode fields ----
    def elec_field(self, name, idx=None):
        """The 1 mA floating-basis E-field of electrode `name` (V/m). `idx` selects a voxel
        subset (None = all)."""
        path, scale = self.reg[name]
        if idx is None:
            if name not in self._cache:
                self._cache[name] = np.load(path).astype(np.float64) * scale
            return self._cache[name]
        M = np.load(path, mmap_mode="r")
        return np.asarray(M[idx], dtype=np.float64) * scale

    def pair_field(self, anode, cathode, I=1.0, idx=None):
        """Channel E-field (M,3) for an electrode pair (anode +, cathode -) driven at I mA."""
        return I * (self.elec_field(anode, idx) - self.elec_field(cathode, idx))

    def montage_field(self, currents, idx=None):
        """General montage. `currents` is {electrode name: I_mA}; sum I = 0 is recommended.
        Returns (M,3)."""
        names = list(currents)
        E = None
        for nm in names:
            e = self.elec_field(nm, idx) * currents[nm]
            E = e if E is None else E + e
        return E

    def has(self, name):
        return name in self.reg
