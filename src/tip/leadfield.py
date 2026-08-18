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


def _refuse_if_marked(d):
    """사용 금지 표식이 있는 리드필드 폴더를 **로드 시점에** 거부한다.

    사람이 읽는 `DO_NOT_USE*.md` 만으로는 코드가 못 읽는다. 스크래치패드에 백질 전도도가
    정본값(0.34795)의 **1/5.56** 인 진단용 판이 두 개 남아 있고(`wmsig_lf`·`e3cm2_lf` —
    이방성을 끄다 생긴 값이라 tip.lite 배포 `Materials.db` 를 포함해 어느 데이터베이스에도
    없다), 그것을 리드필드로 쓰면 백질 필드를 5.6배 과소평가한 채 **아무 경고 없이** 돌아간다.

    ⚠ 두 로드 경로(`_init_direct` · `_use_rebuild`) **모두**에 걸어야 한다. 처음에는
    `_init_direct` 에만 넣었는데 human 모델은 `leadfield_style="legacy"` + `LEADFIELD_SET
    ="rebuild"` 라 그 경로를 타지 않아 가드가 조용히 무력했다.
    """
    if not os.path.isdir(d):
        return
    mark = [f for f in os.listdir(d) if f.upper().startswith("DO_NOT")]
    if not mark:
        return
    nl = chr(10)
    raise RuntimeError(
        "리드필드 세트에 사용 금지 표식이 있다: " + os.path.join(d, mark[0])
        + nl + "  그 파일을 읽고 왜 금지인지 확인할 것."
        + nl + "  실사용 세트는 'leadfield_rebuild_3cm2' 다."
        " 굳이 쓰려면 표식 파일을 지우고 그 이유를 커밋에 남겨라.")


class LeadField:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or C.DATA_DIR
        self._cache = {}          # normalised field per electrode name, loaded on demand

        #  "legacy" = the original human pipeline: `M*.npy` columns keyed by `enames*.json`
        #             and scaled by `unitnorm.json` x LEADFIELD_AMP_FIX.
        #  "direct" = one `{electrode}.npy` per electrode, already at unit current.
        #  A model built the direct way has none of the legacy scaffolding on disk, so the
        #  two paths must not share code — reading `enames1010.json` for a mouse would fail
        #  on the very first line.
        if getattr(C, "LEADFIELD_STYLE", "legacy") == "direct":
            self._init_direct()
        else:
            self._init_legacy()
        self._init_geometry()

    # ---- construction, one method per leadfield style ----
    def _init_legacy(self):
        """The original human pipeline. Unchanged — this path must stay bit-identical."""
        lf_dir = C.LEADFIELD_DIR

        self.enames = json.load(open(C.inputs("enames1010.json")))  # 61
        un = json.load(open(C.inputs("unitnorm.json")))             # dict, 60

        # Files are in column order: the glob sort (M00, M01, ..., M60 with M18 absent) and
        # sorted(unitnorm.keys(), key=int) give the same ordering (asserted in extract_eval.py).
        self.files = sorted(glob.glob(os.path.join(lf_dir, "M*.npy")))
        assert len(self.files) == C.N_ELEC, (
            f"leadfield has {len(self.files)} files, model {C.MODEL_NAME!r} expects {C.N_ELEC}")
        keys = sorted(un.keys(), key=int)
        # Per-column normalisation to 1 mA. C.LEADFIELD_AMP_FIX (= 0.5) corrects for unitnorm
        # having been built against **half the current amplitude**; the derivation is in the
        # config.py comment. Set it to 1.0 to reproduce the historical numbers.
        self.scale = np.array([un[k] for k in keys], dtype=np.float64) * C.LEADFIELD_AMP_FIX

        # file number j (= enames index) -> electrode name; column i maps to that name
        self.col_name, self.col_j = [], []
        for f in self.files:
            j = int(re.findall(r"M(\d+)\.npy$", os.path.basename(f))[0])
            self.col_j.append(j)
            self.col_name.append(self.enames[j])
        self.name2col = {n: i for i, n in enumerate(self.col_name)}

        # Unified electrode registry: the standard 61 plus the lower ring extras if present.
        # name -> (file path, unitnorm)
        self.reg = {self.col_name[i]: (self.files[i], float(self.scale[i]))
                    for i in range(len(self.files))}
        self._load_extras()
        self.set_name = "original"
        if getattr(C, "LEADFIELD_SET", "original") == "rebuild":
            self._use_rebuild()
        self.names = list(self.reg)

    def _init_direct(self):
        """One `{electrode}.npy` per electrode, already normalised to unit current.

        The reference electrode is grounded in the solve, so its field is identically zero:
        it has no file and `N_ELEC` excludes it. For the mouse that reference is PO8, read
        out of the project (the `Passive` boundary) rather than assumed.
        """
        d = os.path.join(C.LEADFIELD_ROOT, C.MODEL.leadfield_dir)
        #  ★진단용·폐기된 세트를 실수로 집는 것을 로드 시점에 막는다. 사람이 읽는 표식
        #  (`DO_NOT_USE*.md`)만으로는 코드가 못 읽는다 — 실제로 스크래치패드에 백질 σ 가
        #  정본값의 1/5.56 인 판이 두 개 남아 있고(`wmsig_lf`·`e3cm2_lf`, 이방성을 끄다 생긴
        #  값이라 어느 데이터베이스에도 없다), 그걸 리드필드로 쓰면 백질 필드를 5.6배
        #  과소평가한 채 아무 경고 없이 돌아간다.
        _refuse_if_marked(d)
        fs = sorted(glob.glob(os.path.join(d, "*.npy")))
        if not fs:
            raise FileNotFoundError(
                f"model {C.MODEL_NAME!r} has no leadfield: expected {{electrode}}.npy under "
                f"{d}. Solve it first, or set TIP_MODEL to a model that has one.")
        self.reg = {os.path.splitext(os.path.basename(f))[0]: (f, 1.0) for f in fs}
        self.names = list(self.reg)
        self.set_name = C.MODEL.leadfield_dir
        # attributes the legacy path exposes and some scripts still read
        self.files, self.enames = fs, list(self.reg)
        self.col_name, self.col_j = list(self.reg), []
        self.name2col = {n: i for i, n in enumerate(self.col_name)}
        self.scale = np.ones(len(fs), dtype=np.float64)
        if len(self.names) != C.N_ELEC:
            #  Not fatal — a partial pool is a legitimate state while a solve is still
            #  running. Loud, because a silently short pool changes every montage it picks.
            print(f"[leadfield] warning: model {C.MODEL_NAME!r} expects {C.N_ELEC} electrodes, "
                  f"found {len(self.names)} in {d}")

    def _init_geometry(self):
        """Voxel decoding and electrode coordinates. Shared by both styles."""
        self.bmask = np.load(C.inputs(C.BMASK_FILE))  # (N,3) int
        g = np.load(C.inputs(C.GAXES_FILE))
        self.cx, self.cy, self.cz = g["cx"], g["cy"], g["cz"]
        self.N = self.bmask.shape[0]

        # electrode coordinates, for native constraints and visualisation
        self.pos = {}
        try:
            self.pos.update(json.load(open(C.inputs(C.POS_FILE))))
        except Exception:
            pass
        #  A set that was solved here ships its own positions.json; prefer it, since the
        #  legacy rule-of-thumb coordinates sat up to 19 mm from the official vertices.
        for sub in (C.MODEL.leadfield_dir, "leadfield_extra"):
            ep = os.path.join(C.LEADFIELD_ROOT, sub, "positions.json")
            if not os.path.exists(ep):
                continue
            try:
                for k, v in json.load(open(ep)).items():
                    self.pos[k] = v if isinstance(v, (list, tuple)) else                         (v.get("pos") or v.get("position") or v.get("xyz"))
            except Exception:
                pass
            break

    def _use_rebuild(self):
        """Swap the whole electrode pool for the rebuilt set (`config.LEADFIELD_SET`).

        ⚠ The rebuilt `.npy` files are **already normalised to 1 mA**, in (N,3) V/m. Unlike the
          old set they must not be multiplied by `unitnorm x LEADFIELD_AMP_FIX`, hence
          scale = 1.0.
        If the folder is missing we fall back to the old set silently — the tool has to run on
        a machine without the data.
        """
        d = getattr(C, "LEADFIELD_REBUILD_DIR", "")
        _refuse_if_marked(d)
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
