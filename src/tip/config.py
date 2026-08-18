# -*- coding: utf-8 -*-
"""
config.py — paths and physical constants. **The single source of truth for paths.**
===================================================================================
Nothing anywhere else in the repository assembles a path. Scripts that build their own
(`os.path.join(HERE, "data")` and friends) break silently every time a file moves — that
happened here, repeatedly, which is why this rule exists.

    repo/
      src/tip/        the program (this file lives here)
      tools/          scripts you run yourself (Sim4Life, analysis, input preparation)
      research/       one-off validation and diagnostics (not distributed)
      inputs/         ★read-only, not reproducible, too large for git
        leadfield/{leadfieldF,leadfield_rebuild,leadfield_extra,leadfield_3cm2}
        geometry/     bmask1010.npy · gaxes1010.npz · pos1010.json …
        masks/        target masks
        fibers/       fibre trajectories and fibre leadfields
      outputs/        ★everything reproducible, not in git
        cache/        content-addressed cache (same analysis → same key → instant)
        jobs/ montage/ figures/ logs/ research/
      docs/

Override with the environment variables `TIP_INPUTS` and `TIP_OUTPUTS` — useful when the
data sits on another drive or a teammate has a different layout.
"""
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))             # repo/src/tip
SRC_DIR = os.path.dirname(PKG_DIR)                               # repo/src
ROOT_DIR = os.path.dirname(SRC_DIR)                              # repo/
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")

INPUTS_DIR = os.environ.get("TIP_INPUTS") or os.path.join(ROOT_DIR, "inputs")
OUTPUTS_DIR = os.environ.get("TIP_OUTPUTS") or os.path.join(ROOT_DIR, "outputs")

GEOMETRY_DIR = os.path.join(INPUTS_DIR, "geometry")
MASKS_DIR = os.path.join(INPUTS_DIR, "masks")
FIBERS_DIR = os.path.join(INPUTS_DIR, "fibers")
LEADFIELD_ROOT = os.path.join(INPUTS_DIR, "leadfield")

CACHE_DIR = os.path.join(OUTPUTS_DIR, "cache")
JOBS_DIR = os.path.join(OUTPUTS_DIR, "jobs")
MONTAGE_DIR = os.path.join(OUTPUTS_DIR, "montage")

#  ⚠ `DATA_DIR` is the old name, from when inputs and outputs shared one folder. It stays
#    as an alias because the argument name `LeadField(data_dir=...)` is already widespread.
#    **New code should use `INPUTS_DIR` / `OUTPUTS_DIR`.**
DATA_DIR = INPUTS_DIR
LEADFIELD_DIR = os.path.join(LEADFIELD_ROOT, "leadfieldF")


def inputs(*parts):
    """Resolve an input file. Given a bare filename it searches the subfolders, so callers
    never need to know where a file currently lives — files move, this does not break."""
    if len(parts) == 1 and os.sep not in parts[0] and "/" not in parts[0]:
        for sub in ("", "geometry", "masks", "fibers", "leadfield"):
            p = os.path.join(INPUTS_DIR, sub, parts[0])
            if os.path.exists(p):
                return p
    return os.path.join(INPUTS_DIR, *parts)


def outputs(*parts, mkdir=True):
    """Resolve an output path, creating the parent directory."""
    p = os.path.join(OUTPUTS_DIR, *parts)
    if mkdir:
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


# ── ★Active head model (2026-08-14) ──────────────────────────────────────
# Everything that changes when the head changes lives in `models.py`; this module re-exports
# the selected model's values under their historical names so existing code keeps working.
# Select with `TIP_MODEL=human|mouse` (default human).
#
# ⚠ Read these as `config.N_ELEC`, not `from config import N_ELEC` — attribute access is
#   late-bound, a from-import freezes the human value at import time.
from . import models as _models                                       # noqa: E402

MODEL = _models.active()
MODEL_NAME = MODEL.name

# Geometry file names — these used to be the literals `bmask1010.npy` and friends.
BMASK_FILE = MODEL.bmask
GAXES_FILE = MODEL.gaxes
BLABEL_FILE = MODEL.blabel
POS_FILE = MODEL.positions

# --- Physical / model constants (validated in FOUNDATION.md and migration §4) ---
N_ELEC = MODEL.n_elec           # electrodes in use (the reference is excluded)
REF_ELEC = MODEL.ref_elec
MIDLINE_X = MODEL.midline_x     # left/right midline (mm)
#  ⚠ The plane alone does not say which side is which: MIDA puts the left hemisphere at
#    x < midline, the IT'IS mouse phantom at x > midline. Use `config.MODEL.is_left(x)`.
LEFT_IS_PLUS_X = MODEL.left_is_plus_x

# TI carriers (tip.lite convention: 2000/2100, Δf = 100 Hz).
# The frequency does not affect envelope amplitude — the solve is quasi-static.
F1_HZ, F2_HZ = 2000.0, 2100.0

# ── ★Leadfield amplitude correction (2026-08-05) ─────────────────────────
# `unitnorm.json` is normalised to **half the current amplitude**, so using it as-is makes
# the field exactly 2× too large. This constant undoes that.
#
# **Why half** — this closes arithmetically, no simulation needed:
#   average power of a sinusoid   P_avg = ½ · V_amp · I_amp
#   Integrating Sim4Life's `El. Loss Density` over volume gives exactly that P_avg.
#   But `leadfield_gen.injected_current()` used the result directly as the current:
#       I_inj = P_avg / V_amp = ½ · I_amp        ← half the amplitude
#   Since `unitnorm = 1e-3 / I_inj`, unitnorm is 2× too large and so is the field.
#
#   That function's docstring says `I = ∫σ|E|²dV`, which is **correct** — but the quantity
#   actually integrated, `El. Loss Density`, is **σ|E|²/2** (confirmed via `RmsFactor`:
#   E and J store peak amplitude at 1/√2, loss density uses the time-averaged convention
#   at 1.0). The "independently confirmed" note in `leadfield_gen.py` compared two
#   equally-wrong quantities, so it could not catch this.
#
# **Measured** (four-sphere phantom, `scratchpad_diag/s4l_phantom_recipe.py`):
#   the charge-conservation current ∫Jx·dA against the loss-density integral converges to
#   2.0 as the grid tightens (3 mm 2.1125 · 2 mm 2.1000 · 1.5 mm 2.0455 · 1 mm 2.0348).
#   The remainder is discretisation error.
#
# "Total injected current 2 mA" in TI refers to **amplitude**, so I_amp is the right basis.
#
# Effect: absolute V/m, threshold ratios, µV predictions and safety limits are **all halved**.
# Dimensionless metrics (M2 focality, phase, cross-method comparison, separability) are
# **unaffected**. Set this to 1.0 to reproduce historical numbers exactly.
LEADFIELD_AMP_FIX = 0.5

# The remaining puzzle: of the "4–6× vs literature" in `SCALE_AUDIT.md`, removing the 2×
# above leaves **2–3×**. → **Identified on 2026-08-10.** See `LEADFIELD_SET` below.

# ── ★Leadfield set selection (2026-08-10) ────────────────────────────────
# "rebuild"  = 70 electrodes re-solved from scratch on this machine with headless
#              Sim4Life (**default**)
# "original" = the old `leadfieldF` + `leadfield_extra` (to reproduce historical numbers)
#
# **Why rebuild is the default** — we solved the tip.lite reference model (uniform 0.4 mm,
# 157 M cells) through our own pipeline to create a baseline, then compared all three:
#   · that baseline reproduced tip.lite's published M1 to **0.8%** → the pipeline is correct
#   · against it, the old `leadfieldF` reads **2.3–2.6× high** (single-electrode field,
#     electrode-pair field, and M1 over 52 montages — three independent paths agree).
#     That is the "remaining 2–3×" above.
#   · the rebuild reads 1.10–1.20× the baseline, depth profile within 1.4%, direction
#     cos 0.9959
#   · median metric ratios over 52 montages: M1 1.14 · M2 0.96 · M3 1.09
#     (the old set: 2.26 / 0.81 / 1.50)
#
# The ten lower-ring electrodes (F9/10 · FT9/10 · T9/10 · TP9/10 · P9/10) also differ. The
# old set used rule-of-thumb positions that sat 19.1 mm away (median) from the official
# `ViP.Create1010System(add_outer_ring=True)` vertices. The rebuild solves at the official
# positions.
#
# ⚠ Rebuild files are **already normalised to 1 mA**, so neither `unitnorm` nor
#   `LEADFIELD_AMP_FIX` is applied to them (scale = 1.0). Only the old set needs that.
LEADFIELD_SET = os.environ.get("TIP_LEADFIELD_SET", "rebuild")
LEADFIELD_REBUILD_DIR = os.path.join(LEADFIELD_ROOT, MODEL.leadfield_dir)
#  "legacy" = M*.npy columns + enames/unitnorm (the original human pipeline)
#  "direct" = one {electrode}.npy already normalised to unit current
LEADFIELD_STYLE = MODEL.leadfield_style

# Direction samples for Tmax (ITIS reference implementation: 120 in-plane directions)
N_DIR = 120

# Safety / optimisation defaults — per model, since a mouse head is ~1/30 the diameter and
# the same current produces a far larger field. `None` means nobody has established it yet;
# `MODEL.require("ich_max")` then fails loudly instead of inventing a number.
ECAP_DEFAULT = MODEL.ecap       # off-target envelope cap (V/m)
IMAX = MODEL.imax               # per-electrode current cap (mA) — a skin safety constraint
ITOT = 4.0                      # per-channel total current budget (mA)

# ── Total injected current budget (the basis for fair comparison) ────────
# I_total = Σ(current into + electrodes) = 0.5 · Σ|all electrode currents|. Normalising
# every method (classic, dual, distributed) to this makes M1 directly comparable across
# methods. Per-electrode current still respects IMAX.
#
# ── ★2026-08-05: default changed 2.0 → 1.0 to match the tip.lite convention ──
# The newer tip.lite CSVs give channel currents as `a1, a2` with **a1 + a2 = 1.0**
# (measured: 0.5307854201 + 0.4692145799 = 1.0000000000). Our ITOTAL is defined as exactly
# the same quantity, so a default of 1.0 puts both tools on one scale.
#
# ⚠ **That reasoning turned out to be wrong** — see `CURRENT_NORM` below. `a1, a2` are
#   distribution *ratios*, not absolute currents. The apparent 7% agreement at the time was
#   a normalisation error (÷≈1.9) cancelling the old leadfield's 2.3× overestimate.
#
# ⚠ **All historical absolute numbers halve.** To compare with older values either set
#   ITOTAL = 2.0 or double them. M2 (focality) and M3 (leakage) are dimensionless and
#   **unaffected**.
# ⚠ `classic.py` and others do `from ..config import ITOTAL`, which **binds at import time.**
#   To change it at runtime do not edit this constant — pass it through, e.g.
#   `channel_currents(r, budget=...)`. The GUI slider uses that path.
ITOTAL = 1.0
# Per-channel maximum current (the value the larger channel takes under max-channel norm)
ICH_MAX = MODEL.ich_max

# ── ★Current normalisation convention (2026-08-11) ───────────────────────
# "max_channel" = pin the **larger channel** to ICH_MAX; the smaller one follows the ratio.
#                 → **the tip.lite convention.** Absolute metrics can be compared directly
#                 against the reference and the literature.
# "total"       = pin i1 + i2 = ITOTAL → **for fair cross-method comparison**; comparing
#                 classic, dual and distributed strength at equal dose requires this one.
#
# **Why max_channel is the default** — the 2026-08-05 note above ("a1 + a2 = 1.0, so
# matching the sum puts us on the same scale") was **wrong**. `a1, a2` are distribution
# ratios; the absolute currents are scaled so the larger channel is 1.0. Three checks:
#   · solving the tip.lite **model** directly reproduced their published M1 to **0.8%** —
#     only under max_channel
#   · with electrode coordinates matched, over 52 montages: vs published values,
#     A(max) 1.143 / C(sum = 1.0) 0.589; residual correlation with `max(a1, a2)` is
#     A **−0.076** vs C +0.374 (the correct convention should correlate at 0)
#   · all four regions point the same way (C sits consistently at ≈0.58–0.62)
#
# ⚠ Passing `budget=` explicitly (dual TI and friends) **always** means a total-current
#   budget — there the point is splitting a budget per system, which is a different thing.
CURRENT_NORM = os.environ.get("TIP_CURRENT_NORM", "max_channel")

# Tissue labels (BLABEL_FILE). Per model — the MIDA numbers below mean nothing in another
# phantom, and a model that has not been voxelised yet has none at all (hence None).
LABEL_GM = MODEL.labels.get("gm")
LABEL_WM = MODEL.labels.get("wm")
LABEL_HIPPO = MODEL.labels.get("hippocampus")
# Labels used to restrict a target to neural tissue (`Target.from_sphere(restrict_neural=True)`).
# **This is not the off-target pool definition** — that is OFF_LABEL_SETS below.
# tip.lite agreement checks (`validate_tiplite.py`) depend on this, so it stays GM-only.
NEURAL_LABELS = tuple(MODEL.labels[k] for k in MODEL.neural
                      if MODEL.labels.get(k) is not None)

# ── Off-target pool definition (Target.off_idx) ──────────────────────────
# 2026-08-05: **the default changed from GM-only to GM ∪ WM.** Reasoning below.
# Per model — for the human head the sets are {gm, gm_wm, brain}; the IT'IS mouse phantom has
# **no white matter at all**, so `gm_wm` there would be a different thing wearing the same name.
OFF_LABEL_SETS = MODEL.off_sets
OFF_DEFAULT = MODEL.off_default

# ── ★Why it changed (reproduce with scratchpad_diag/diag_offdef.py) ──────
# GM-only leaves **981,723 white-matter voxels (51% of the brain)** and every deep nucleus
# out of the off-target pool. For deep targets the internal capsule and surrounding white
# matter are exactly where the field peaks outside the target — so the optimiser could pour
# field into them **without any penalty**.
#
# Re-running the search per definition across five targets, **the winner changes in 3 of 5**
# (thalamus L, STN L, amygdala L change; hippocampus L and putamen L stay).
#
# STN L is the clearest. Re-scored on a common yardstick (whole-brain off):
#     GM-only winner  : M1 1.186 · M2 2.35 · M3 **10.8%**  (its own off pool reports 5.7%)
#     GM+WM winner    : M1 0.196 · M2 5.34 · M3 **0.3%**
# GM-only calls a montage with 36× worse real leakage "optimal", and by its own yardstick
# that looks reasonable.
#
# **gm_wm and brain pick the same winner for all five targets** → the missing white matter
# was the entire cause and adding deep nuclei changes nothing. Hence gm_wm, not brain.
#
# To reproduce older numbers use `Target(..., off_labels="gm")`. The definition actually
# used is recorded in `Target.off_def` and must be reported alongside any metric.

# ── Voxel off-target vs fibre off-target ─────────────────────────────────
# Voxel metrics (M1/M2/M3, `targets.Target.off_idx`) : OFF_DEFAULT = **GM ∪ WM**
# Fibre populations (`fiberlead.sample_seeds` default): **GM ∪ WM**
# → the change above brought the two into agreement (previously voxels were GM-only).
#
# Caution: even so, **do not put the two metrics side by side in one table.** A montage
# optimised on fibres once showed 70–80% leakage on the voxel directional metric, yet in
# NEURON it had the lowest target threshold — the two yardsticks measure different things
# (E versus the activating function).
# Details: MIGRATION_STATUS.md §4-1 · SETUP.md §5-4
