# -*- coding: utf-8 -*-
"""
models.py — head model descriptors. **The single source of truth for what varies per model.**
=============================================================================================
`config.py` owns paths; this file owns everything that changes when you swap the head.
Until now a single human head was baked in: file names ended in `1010`, the electrode count
was asserted to be 60, the midline was a literal `-27.0`, and tissue labels were the MIDA
numbers 75 / 131 / 81. None of that survives contact with a second model.

Select with the environment variable **`TIP_MODEL`** (default `human`)::

    TIP_MODEL=mouse python -m tip.gui.app

Adding a model means adding one `Model(...)` entry below. Nothing else should learn its name.

Two rules
---------
**Unknown is `None`, never a guess.** A field nobody has measured yet stays `None` and
`Model.require()` raises a clear error the moment something needs it. A plausible-looking
placeholder that silently produces numbers is the worst outcome available — this project has
already lost weeks to a coordinate convention that looked reasonable and was wrong.

**Left/right needs a side convention, not just a plane.** `midline_x` alone is ambiguous:
in MIDA the left hemisphere sits at x < midline, in the IT'IS mouse phantom it sits at
x > midline. Getting this backwards silently swaps every lateralised target, so the sign is
stored explicitly in `left_is_plus_x`.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict


@dataclass(frozen=True)
class Model:
    """Everything that changes when the head changes."""

    # --- identity -------------------------------------------------------
    name: str
    label: str                       # shown in the UI
    species: str

    # --- geometry files (looked up through config.inputs) ---------------
    #  Legacy human names end in `1010`; a new model may name them anything.
    bmask: str
    gaxes: str
    blabel: str
    positions: str

    # --- electrodes -----------------------------------------------------
    n_elec: int                      # electrodes in the pool (reference excluded)
    ref_elec: str                    # grounded electrode; its field is identically zero
    leadfield_dir: str               # relative to config.LEADFIELD_ROOT
    #  "legacy" = the original human pipeline: `M*.npy` columns + `enames*.json` +
    #             `unitnorm.json`, scaled by unitnorm x LEADFIELD_AMP_FIX.
    #  "direct" = one `{electrode}.npy` per electrode, already normalised to unit current.
    #             Everything built after 2026-08 uses this.
    leadfield_style: str = "direct"

    # --- anatomy --------------------------------------------------------
    midline_x: Optional[float] = None       # left/right dividing plane (mm)
    left_is_plus_x: bool = False            # see the module docstring — not cosmetic
    #  An **oblique** midline, for a model whose head is not axis-aligned. NeuroRat lies at
    #  about 55 deg to z, so no single coordinate separates the hemispheres: the plane is
    #  `midline_normal . p == midline_offset` and the normal points towards the **left**.
    #  When these are set they take precedence over `midline_x`.
    midline_normal: Optional[Tuple[float, float, float]] = None
    midline_offset: Optional[float] = None
    labels: Dict[str, Optional[int]] = field(default_factory=dict)
    off_sets: Dict[str, Optional[Tuple[int, ...]]] = field(default_factory=dict)
    off_default: str = "gm_wm"
    neural: Tuple[str, ...] = ("gm",)       # keys into `labels`

    # --- currents -------------------------------------------------------
    #  A mouse head is ~1/30 the diameter of a human one, so the same current gives a far
    #  larger field. These are per-model, and `None` means nobody has established it yet.
    ich_max: Optional[float] = None         # mA held by the larger channel
    #  Is `ich_max` an established convention for this head, or just a working value someone
    #  picked? A filled-in field looks equally authoritative either way, and current scales
    #  every absolute field exactly, so the UI has to be able to say which it is.
    ich_established: bool = False
    imax: Optional[float] = None            # per-electrode cap (mA)
    ecap: Optional[float] = None            # off-target envelope cap (V/m)

    # --- Sim4Life -------------------------------------------------------
    smash_env: str = ""                     # env var naming the .smash file
    notes: str = ""

    # ------------------------------------------------------------------
    def require(self, *fields):
        """Fetch fields that must be known, failing loudly rather than guessing."""
        out = []
        for f in fields:
            v = getattr(self, f, None)
            if v is None:
                raise ValueError(
                    f"model {self.name!r}: {f!r} is not established yet.\n"
                    f"  Fill it in src/tip/models.py once it has been measured. "
                    f"It is deliberately left None so that nothing invents a value.")
            out.append(v)
        return out[0] if len(out) == 1 else tuple(out)

    def label_id(self, key):
        """Tissue label number for a key such as 'gm'. Raises if this model lacks it."""
        if key not in self.labels or self.labels[key] is None:
            raise ValueError(
                f"model {self.name!r} has no tissue label for {key!r} "
                f"(known: {sorted(k for k, v in self.labels.items() if v is not None)})")
        return self.labels[key]

    def off_labels(self, key=None):
        """Label tuple for an off-target pool key. `None` means every brain voxel."""
        key = self.off_default if key is None else key
        if key not in self.off_sets:
            raise ValueError(f"unknown off-target set {key!r} for model {self.name!r} "
                             f"(available: {list(self.off_sets)})")
        return self.off_sets[key]

    def is_left(self, x):
        """True where x lies in the left hemisphere. Uses the stored side convention.

        Axis-aligned models only. A model with an oblique midline (`midline_normal`) has no
        answer from x alone, so it raises here rather than returning a plausible wrong side —
        call `is_left_pts` with full coordinates instead.
        """
        if self.midline_normal is not None:
            raise ValueError(
                f"model {self.name!r} has an oblique midline; x alone cannot decide the "
                f"hemisphere. Use is_left_pts((N,3) coordinates).")
        mid = self.require("midline_x")
        return (x > mid) if self.left_is_plus_x else (x < mid)

    def is_left_pts(self, P):
        """True where each (N,3) point lies in the left hemisphere. Works for both kinds."""
        import numpy as np
        P = np.asarray(P, float).reshape(-1, 3)
        if self.midline_normal is not None:
            n = np.asarray(self.require("midline_normal"), float)
            return P @ n > self.require("midline_offset")
        return self.is_left(P[:, 0])


# ─────────────────────────────────────────────────────────────────────────
# MIDA human head — the model everything was originally written against.
# These values are unchanged; moving them here must not alter a single number.
HUMAN = Model(
    name="human",
    label="Human (MIDA)",
    species="human",
    bmask="bmask1010.npy",
    gaxes="gaxes1010.npz",
    blabel="blabel1010.npy",
    positions="pos1010.json",
    n_elec=60,                       # Cz excluded — it is the reference
    ref_elec="Cz",
    #  ★2026-08-14: 전극을 실제 프로토콜·tip.lite 와 같은 **3 cm²** 로 키워 84전극 재생성.
    #  이전 세트 `leadfield_rebuild` 는 0.5 cm²(r=4mm)로, 원본 `leadfieldF` 에서 물려받은
    #  값이었다. 공개 CSV(모델 무관) 4표적 53몽타주 대조:
    #      M1비 1.139 → 1.118 · **M2비 0.956 → 0.988** · M3비 1.098 → 1.053 · M1 CV 9.3 → 8.9%
    #      몽타주 선정 **후회(WP 손실) 29.4% → 10.1%** · 1위 일치 1/4 → 2/4 표적
    #      (좌해마는 공개 1위 `O1-T7|AF7-P10` 을 정확히 맞힘 — 이전엔 7위/11)
    #  대가: WP 순위상관 0.631 → 0.573(중하위권 순서만 흐트러짐, 상위권은 개선).
    #  ⚠ 전극 **위치는 그대로**이고 반경만 2.4430배 — `positions.json` 은 동일 파일이다.
    #  ⚠ M1 전용 상관이 0.905 로 WP(0.573)보다 여전히 높다 ⇒ **강도 위주 목적함수 처방 유지.**
    #  되돌리려면 `TIP_LEADFIELD_DIR=leadfield_rebuild` (아래 override 참조).
    leadfield_dir=os.environ.get("TIP_LEADFIELD_DIR", "leadfield_rebuild_3cm2"),
    leadfield_style="legacy",        # `TIP_LEADFIELD_SET=rebuild` swaps the pool afterwards
    midline_x=-27.0,
    left_is_plus_x=False,            # MIDA: the left hemisphere is at x < midline
    labels={"gm": 75, "wm": 131, "hippocampus": 81},
    off_sets={
        "gm": (75,),                 # tip.lite reproduction and historical numbers
        "gm_wm": (75, 131),          # ★default — see the OFF_LABEL_SETS note in config.py
        "brain": None,               # every brain voxel; measured identical to gm_wm
    },
    off_default="gm_wm",
    neural=("gm",),
    ich_max=1.0,
    ich_established=True,            # the tip.lite convention, published with their values
    imax=2.0,
    ecap=0.25,
    smash_env="TIP_REBUILD_SMASH",
)

# ─────────────────────────────────────────────────────────────────────────
# IT'IS B6C3F1N_M_3w mouse — read out of `Mouse.smash` on 2026-08-13.
#
# ★★PARKED 2026-08-15 — BLOCKED ON GEOMETRY, NOT ON EFFORT.
#   `Mouse.smash` **does not contain the phantom's tissue geometry.** `CreateVoxels()` fails
#   with "Simulation pre-processing failed" because 68 of its material settings point at
#   entities that are not in the file: of 151 components exactly 76 resolve, and those 76 are
#   the 38 electrodes plus the 38 sensors. `B6C3F1N_M_3w 1 -> Brain` is an empty group.
#
#   This is absence, not a licence lock. The model is one ACIS blob (548 MB of the 557 MB
#   file) and scanning it finds `Elec_0.3mm` 76 times and `CaudoPutamen` 4 times but
#   `Hippocampus`, `Thalamus`, `Cerebral_cortex` and `Skull` **zero** times. The tissue names
#   that do appear elsewhere in the file are the simulation's component descriptions.
#
#   What the file does still hold: the `Brain` mesh (299.618 mm^3) and Striatum Pallidum L/R
#   (13.52 / 13.93 mm^3) as real anatomy, a 470x200x950 @ 0.1 mm greyscale MRI whose voxels
#   the API will not hand over, and the `Targets_PuStriaSpheres` entries — which are literally
#   spheres, not anatomy (14.137 mm^3 is exactly r = 1.5 mm, 3.054 is exactly r = 0.9 mm).
#   No hippocampus, no thalamus, no skull, no skin.
#
#   So the 15-montage reference CSV (`results_mouse_hippocampus_left.csv`) cannot be
#   reproduced from this file at all — its target does not exist here. tip.lite has nothing
#   further to give. Restarting needs a complete labelled mouse head from somewhere else
#   (open atlases, or an IT'IS ViP animal-model licence — the 17 features we hold have none).
#
#   Everything below stays because it is measured and still correct; only the geometry is
#   missing. Do not "fix" this by segmenting the MRI ourselves: the result would no longer be
#   tip.lite's model, which removes the only reason the CSV was worth having.
#
# Established (by reading the project in Sim4Life, not assumed):
#   · 38 electrodes `Elec_0.3mm <10-10 label>`, all the same body rotated onto the skull
#     (volume 0.098175 mm^3 and area 1.668971 mm^2 are bit-identical across all 38)
#   · the simulation is `EmLfElectroQsOhmicSimulation` named "LF": 37 electrodes carry the
#     `Active` boundary with TreatAsPort=True and **PO8 alone is `Passive`** — so PO8 is the
#     reference and one solve yields all 37 leadfields
#   · midline x = -22.49 mm, from four independent observations agreeing within 0.07 mm
#     (CaudoPutamen / VentralStriatum / Striatum Pallidum L-R midpoints and the brain mesh
#     bounding-box centre). **Left sits at the larger x here** — the opposite of MIDA
#   · materials are IT'IS LF database links, anisotropy off for all 44. Thalamus 0.475 S/m
#     matches the tip.lite human canonical value, so both models share a material basis
#   · there is **no white matter** in this phantom, so `gm_wm` cannot mean what it means for
#     the human head
#
# Deliberately unknown until the first solve:
#   · tissue label numbers — the phantom is an `Image`, so tissues are not model entities.
#     The label map only exists after voxelisation, in the solver h5 (same as the human path)
#   · `ich_max` — tip.lite's mouse CSV gives ratios, never an absolute current. Solve first,
#     compare M2/M3 (current-invariant) to validate the model, **then** back out the current
#     from the M1 ratio. Assuming a current first makes an M1 gap unattributable
#   · off-target definition — `Brain_Mask/Brain` (a 299.618 mm^3 mesh tip.lite kept in its own
#     group) and the grey-matter label are both candidates; 15 reference montages are enough
#     to tell them apart on M2/M3
MOUSE = Model(
    name="mouse",
    label="Mouse (IT'IS B6C3F1N_M_3w)",
    species="mouse",
    bmask="bmask_mouse.npy",
    gaxes="gaxes_mouse.npz",
    blabel="blabel_mouse.npy",
    positions="pos_mouse.json",
    n_elec=37,                       # 38 electrodes, PO8 is the reference
    ref_elec="PO8",
    leadfield_dir="leadfield_mouse",
    leadfield_style="direct",
    midline_x=-22.49,
    left_is_plus_x=True,             # ★opposite of MIDA — see the module docstring
    labels={"gm": None, "wm": None, "hippocampus": None, "thalamus": None},
    off_sets={"brain": None},        # resolved after the first solve
    off_default="brain",
    neural=("gm",),
    ich_max=None,                    # ★back-solved from M1; never guessed
    imax=None,
    ecap=None,
    smash_env="TIP_MOUSE_SMASH",
    notes="PARKED — tissue geometry absent from Mouse.smash; see the block above. "
          "Solver grid 387x575x329 = 73.2 MCells, manual discretisation at 0.625 mm with "
          "20% grading (the earlier 'uniform 49.9 um' note was a misreading of one axis).",
)

# ─────────────────────────────────────────────────────────────────────────
# IT'IS NeuroRat V4.0 (male, 150 g, posable) — solved here on 2026-08-18.
#
# This is the animal model the mouse was supposed to be. `Mouse.smash` turned out to ship
# without its tissue geometry (see the MOUSE note above) and the downloader offers no mouse
# phantom, so the branch closed; NeuroRat is in our licence, has all 179 tissues including
# **Hippocampus (31.88 mm^3) and Thalamus (39.73 mm^3)**, skull, CSF, dura and skin, and
# imports in 15 s.
#
# ★★There is **no reference CSV** for the rat. The tip.lite results are for their mouse and
#   cannot be replayed here. Validation is therefore a *consistency* check (leadfield
#   superposition against a direct Sim4Life re-solve), which tests the pipeline, **not the
#   model**. Say so wherever these numbers are reported.
#
# Established by measurement, not assumption:
#   · 38 electrodes, Ø0.25 x 2.0 mm pins normal to the scalp, placed by scaling the tip.lite
#     mouse layout (AP x1.49, ML x2.23) and ray-casting onto the NeuroRat skin. 38/38 hit.
#   · one solve gives 37 leadfields: 37 electrodes are `Active` Dirichlet ports and **PO8
#     alone is `Passive`** — the same convention as the tip.lite mouse.
#   · materials relinked to **IT'IS LF 4.2**, the database the human model and tip.lite use.
#     Nine comparable tissues match their canonical values exactly.
#   · the head is tilted ~55 deg to z, so the hemispheres do not separate along any axis.
#     The midline is the plane `ml_left . p == -68.320`, fitted from 9 midline structures
#     whose spread about it is 0.506 mm.
#
# ⚠**A lateralised rat target carries ~7% of the other side, and that is the phantom.**
#   Scored against the brain mask this plane puts 53.3% of voxels on the left, per structure
#   41% (midbrain) to 57% (hippocampus). Three independent refits — mirror overlap, per-
#   structure volume balance, label-mirror agreement — land on three different planes and
#   **none reduces the imbalance**; the best label-mirror agreement reachable by any plane is
#   0.81, where a symmetric segmentation would exceed 0.90. The interhemispheric fissure
#   shows as a density minimum exactly at this plane. So NeuroRat V4.0 is simply one animal
#   and is lopsided (cortex: 401k voxels left, 344k right). `tools/prep/fit_rat_midplane.py`
#   reproduces all of it. Report the contamination; do not tune the plane to hide it.
#
# ⚠2026-08-18: the first solve was **discarded**. Its grid box had been set by hand and
#   missed the head (z padding -135.7 mm), so the thalamus kept 9.7% of its volume and the
#   cerebellum, pons and medulla had no voxels at all. The rebuild sets the grid lines
#   explicitly and refuses to continue unless every brain structure's voxel volume matches
#   its mesh volume — it now agrees to within 1.6%. Keep that check.
RAT = Model(
    name="rat",
    label="Rat (IT'IS NeuroRat V4.0)",
    species="rat",
    bmask="bmask_rat.npy",
    gaxes="gaxes_rat.npz",
    blabel="blabel_rat.npy",
    positions="pos_rat.json",
    n_elec=37,                       # PO8 excluded — it is the reference
    ref_elec="PO8",
    #  ★2026-08-20: switched from `leadfield_rat` (EM LF **port mode**: electrode k at 1 V
    #  with all 36 others held at 0 V) to a re-solve under the **montage convention** — one
    #  electrode driven, the reference at 0 V, every other electrode **floating**, which is
    #  what a real montage and `s4l_montage.set_pair` do. The old convention shorts 36 PEC
    #  pins together on the scalp, a low-impedance path across the head that no experiment
    #  has, and it is not a small effect: it drew 1.69-3.73x the current at 1 V (median
    #  2.27x), so the field per mA came out that much too small.
    #
    #  Verified against a direct Sim4Life solve of `O1-C5 | PO3-AF3` (left hippocampus,
    #  0.1 mA), which is the only ground truth this head has:
    #                     M1        M2       M3
    #      Sim4Life     1.2494    1.9991   10.477 %
    #      port mode    0.6807    2.1653    9.536     (M1 -45.5 %)
    #      this set     1.2485    1.9992   10.505     (M1  -0.1 %)
    #  `leadfield_rat` is kept on disk to reproduce numbers published before this date; every
    #  absolute rat V/m from it is about 1.8x low. M2, M3 and montage rankings barely moved.
    #  Go back with `TIP_RAT_LEADFIELD_DIR=leadfield_rat`. It is a **separate variable from
    #  the human's** `TIP_LEADFIELD_DIR` on purpose: both are read once at import, so a
    #  shared name would silently point the other head at a set that is not its own.
    leadfield_dir=os.environ.get("TIP_RAT_LEADFIELD_DIR", "leadfield_rat_float"),
    leadfield_style="direct",        # {electrode}.npy, already 1 mA normalised
    midline_x=None,                  # oblique — see below
    midline_normal=(-0.42439439910665616, -0.8855478420281918, -0.18892965221508484),
    midline_offset=-68.320,
    #  Label numbers come from tools/s4l/rat_extract.py:RAT_LABELS and are written to
    #  inputs/labels_rat.json. They are ours, not the solver's voxel ids, so they survive a
    #  re-voxelisation. There is **no white matter** as a separate tissue in this phantom.
    labels={"gm": 1, "wm": None, "hippocampus": 3, "thalamus": 4,
            "caudo_putamen": 5, "cerebellum": 9},
    off_sets={
        "cortex": (1,),                     # Cerebral_cortex alone
        "forebrain": (1, 2, 3, 4, 5, 10),    # cortex, rest_of_brain, hippo, thal, CPu, OB
        "brain": None,                      # every voxel in the mask — the default
    },
    off_default="brain",
    neural=("gm",),
    #  Measured on the solved set (2026-08-18): injected current 0.261-0.717 mA per volt
    #  (median 0.295), median brain |E| 0.725-2.570 V/m per mA (median 1.203), electrode to
    #  nearest brain voxel 2.46-4.83 mm. **C6, CP6 and P6 draw 1.5-2.7x the current of the
    #  rest** — all three are the only electrodes with temporalis muscle within 2 mm
    #  (corr(I, muscle fraction) = +0.80; muscle 0.461 S/m against fat's 0.0776). Physics,
    #  not a defect, and each field is normalised to 1 mA regardless. Note the left-side
    #  counterparts C5/CP5/P5 have no muscle under them: the same lopsidedness as the midline.
    #
    #  ⚠ 0.1 mA is an **operator's protocol choice**, not a measured or published value for
    #  this phantom — set on request 2026-08-19 so the GUI starts with a filled field. It was
    #  deliberately `None` before, because current is first order in Tmax: it scales M1
    #  exactly and leaves M2, M3 and every ranking untouched, so a wrong number does not look
    #  wrong anywhere, it just makes every absolute field quietly wrong. Treat any absolute
    #  rat number as "per 0.1 mA on the larger channel" and say so when reporting it.
    #  `imax` and `ecap` stay unset: nobody has established a per-electrode cap or an
    #  off-target envelope limit for a rat, and those two do change which montage wins.
    ich_max=0.1,
    imax=None,
    ecap=None,
    smash_env="TIP_RAT_SMASH",
)

REGISTRY = {m.name: m for m in (HUMAN, MOUSE, RAT)}
DEFAULT = "human"


def active():
    """The model selected by `TIP_MODEL`."""
    name = (os.environ.get("TIP_MODEL") or DEFAULT).strip().lower()
    if name not in REGISTRY:
        raise ValueError(f"TIP_MODEL={name!r} is not a known model "
                         f"(available: {sorted(REGISTRY)})")
    return REGISTRY[name]
