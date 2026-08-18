# TIP — transcranial Temporal Interference stimulation Planner

Pick the two electrode pairs and the current ratio that best stimulate a target inside a head
model, then re-solve the chosen montage in Sim4Life to check it across the **whole head**.

- **Planning is instant.** Per-electrode leadfields (1 mA solutions) are precomputed, so an
  exhaustive search over 70–84 electrodes runs in seconds by linear superposition.
- **Verification is exact.** The chosen montage goes back to Sim4Life and is re-solved with FEM.
  Everything outside the brain — scalp, skull, CSF, eye — only exists there, and that is where
  **safety numbers such as scalp current density** come from.

---

## Quick start

```bash
git clone <this repository>
cd tip
pip install -e ".[plot]"          # numpy + scipy (matplotlib for slice images)
# put the data under inputs/ — see "Getting the data" below
python -m tip.gui.app             # or run_gui.bat (Windows) / ./run_gui.sh (macOS, Linux)
```

A browser opens at `http://127.0.0.1:8765`. Pick a target → click electrodes in the 3D view →
press **Compute**. The UI ships in English and Korean; toggle with the button in the header.

### Which platform do I need?

|  | Windows | macOS / Linux |
|---|---|---|
| planner, metrics, 3D view, result export | ✔ | ✔ |
| montage re-solve in Sim4Life | ✔ | ✖ — Sim4Life is Windows-only |
| NEURON axon models | via WSL | ✔ natively |

The planner and the UI depend on nothing beyond numpy, scipy and the standard library, so they
are fully cross-platform. Only the `tools/s4l/` scripts and **▶ Send to Sim4Life** need Windows,
because Sim4Life itself does. A colleague on macOS can plan montages, compute M1/M2/M3, inspect
the field in 3D and export results — and hand a montage to someone on Windows for the whole-head
solve, or read one back from the cache.

---

## Layout

```
tip/
├─ src/tip/            ★the program — this is what gets distributed
│   ├─ leadfield.py      per-electrode leadfields, loading and superposition
│   ├─ ti.py             TI envelope (Grossman closed form) · directional Tmax
│   ├─ metrics.py        M1 strength · M2 focality · M3 leakage
│   ├─ targets.py        target definitions (mask / sphere / coordinates)
│   ├─ optimize/         optimisation methods (table below)
│   ├─ orch/             job management, Sim4Life execution, result cache
│   └─ gui/              web UI (app.py + a single dependency-free index.html)
│
├─ tools/              run these yourself — most need Sim4Life
│   ├─ s4l/              rebuild leadfields · add electrodes · export a montage
│   ├─ analyze/          solver output → per-tissue statistics + slice images
│   └─ prep/             build inputs (rasterise masks, extract fibre tracts, …)
│
├─ inputs/             ★read-only, not reproducible, git-ignored (3.6 GB)
│   ├─ leadfield/        four sets of per-electrode 1 mA solutions
│   ├─ geometry/         brain voxel mask · grid axes · electrode positions
│   ├─ masks/            target masks
│   └─ fibers/           fibre trajectories and fibre leadfields
│
├─ outputs/            ★everything here is reproducible, git-ignored
│   ├─ cache/            content-addressed cache — identical analyses are never redone
│   ├─ jobs/             state of running and finished jobs
│   ├─ montage/          Sim4Life analysis results (numbers + slice PNGs)
│   └─ research/         artefacts from past experiments
│
├─ research/           one-off validation and diagnostic scripts (git-ignored)
└─ docs/               design notes, scale audit, pipeline design
```

**Head models are defined in exactly one place: [src/tip/models.py](src/tip/models.py).**
Grid, masks, electrode pool, tissue labels, midline and current limits all change together
when you swap the head, so they live on one descriptor. Pick one with `TIP_MODEL`:

```bash
TIP_MODEL=human python -m tip.gui.app   # MIDA human head — the default
TIP_MODEL=rat   python -m tip.gui.app   # IT'IS NeuroRat V4.0, 37 electrodes, reference PO8
TIP_MODEL=mouse python -m tip.gui.app   # IT'IS B6C3F1N_M_3w mouse — parked, no geometry
```

Adding a model means adding one `Model(...)` entry. Anything not yet measured for a model is
left `None` and fails loudly when something reaches for it — a plausible placeholder that
quietly produces numbers is worse than a crash. Note that `midline_x` alone does not say
which side is which: MIDA puts the left hemisphere at x < midline, the mouse phantom at
x > midline, so use `config.MODEL.is_left(x)` rather than comparing by hand.

The rat's head is tilted ~55° to z, so **no coordinate separates its hemispheres**: it
carries an oblique `midline_normal` / `midline_offset` instead, and `is_left(x)` refuses to
answer for it. Use `config.MODEL.is_left_pts(coords)`, which handles both kinds. Two things
to state whenever rat numbers are reported: there is **no reference dataset** for this model
(tip.lite's CSVs are for their mouse, whose electrode set does not exist here), so validation
is a consistency check of the pipeline and not of the model; and a lateralised rat target
carries about **7% of the contralateral structure**, because the phantom's own segmentation
is asymmetric — see `tools/prep/fit_rat_midplane.py`, which measures it and changes nothing.

**Paths are defined in exactly one place: [src/tip/config.py](src/tip/config.py).**
Scripts that assemble paths themselves break silently whenever a file moves — that happened here,
so it is now a rule. To keep the data elsewhere, set `TIP_INPUTS` and `TIP_OUTPUTS`.

---

## Getting the data

`inputs/` is 3.6 GB and is deliberately not in the repository. Download it from the team share
and unpack it at the repository root so the tree looks like this:

```
inputs/
  leadfield/leadfield_rebuild_3cm2/  ★default set. 3 cm² electrodes, 84 × (1907678,3)  1.8 GB
  leadfield/leadfield_rebuild/   same model at 0.5 cm² (the previous default)         1.8 GB
  leadfield/leadfieldF/          legacy set (see the warning below)                  1.3 GB
  leadfield/leadfield_extra/     lower-ring electrodes of the legacy set             229 MB
  leadfield/leadfield_3cm2/      4 electrodes, from an earlier machine — kept only
                                 as a historical artefact; do not use                 92 MB
  geometry/                      bmask1010.npy · gaxes1010.npz · pos1010.json …       54 MB
  masks/                         target masks                                         74 MB
  fibers/                        fibre trajectories and fibre leadfields              72 MB
```

> **Team share**:
> https://drive.google.com/drive/folders/1b-5ad3QkNmGa4wB4TmPg7sQqWaOiH34y?usp=sharing
> Ask a maintainer if you cannot reach it. To keep the data elsewhere, set `TIP_INPUTS`
> instead of moving it.

**You do not need all of it.** `leadfield/leadfield_rebuild_3cm2/`, `geometry/` and `masks/` —
about 1.9 GB — are enough to run everything the UI offers. The other sets exist only to
reproduce past comparisons: `leadfieldF` + `leadfield_extra` are the legacy set,
`leadfield_rebuild` is the same model at 0.5 cm² electrodes (the default until 2026-08-14),
and `leadfield_3cm2` is a four-electrode fragment from an earlier machine.

> **Electrode size does matter** — the older claim that it does not was disproved on
> 2026-08-14. Re-solving all 84 electrodes at 3 cm² (the size the actual protocol, tip.lite
> and its published values all use) moved the agreement with those published values from
> M1 1.139 / M2 0.956 / M3 1.098 to **1.118 / 0.988 / 1.053**, and cut the montage-selection
> regret from 29.4% to **10.1%**. The chosen montage itself does not change. Set
> `TIP_LEADFIELD_DIR=leadfield_rebuild` to go back. Numbers:
> `outputs/research/metrics_by_region.md`.

**Regenerating it instead.** The leadfields can be rebuilt from the Sim4Life project with
`tools/s4l/add_electrodes.py` — roughly 2-3 minutes per electrode across 84 electrodes, and it
needs a Sim4Life licence. See [tools/README.md](tools/README.md).

Check it loaded:

```bash
python -c "import sys;sys.path.insert(0,'src');from tip import LeadField;lf=LeadField();print(len(lf.names),'electrodes',lf.set_name)"
# → 84 electrodes rebuild
```

> ⚠ **The two leadfield sets are not on the same scale.** Per unit current the legacy
> `leadfieldF` reads about 2.3–2.5× high; three independent checks agree. The default is
> `rebuild`. You can switch with `TIP_LEADFIELD_SET=original`, but **never mix absolute values
> from the two sets.** See [docs/SCALE_AUDIT.md](docs/SCALE_AUDIT.md).

---

## Optimisation methods

| Mode | What it does | When to use it |
|---|---|---|
| `classic` | exhaustive search over two pairs, then polish the current ratio | default; strongest for superficial targets |
| `nsga` | multi-objective evolution (Pareto front) | when you want to see the strength ↔ focality trade-off |
| `distributed` | non-convex multi-channel interference optimisation | when **directional selectivity** matters |
| `dual` | 4-channel 2+2 dual TI | when deep targets need more strength |
| `timemux` | time-multiplexed slots | to hit several places without widening the focus |

No method dominates — the winner depends on target depth and whether a specific axis is required.

---

## Metrics

| | Meaning | Direction |
|---|---|---|
| **M1** | median TI envelope inside the target (V/m) | higher = stronger |
| **M2** | target vs off-target strength ratio (RMS²) | higher = more focal |
| **M3** | fraction of off-target volume stimulated above the target median (%) | lower = less leakage |

> ⚠ **The yardstick is itself a control.** For one montage these numbers move by 0.59× to 2.16×
> depending on isotropic vs directional evaluation and on the current normalisation convention.
> Always compare with **the same yardstick on both sides**. The convention lives in
> `config.CURRENT_NORM` (default `max_channel`).

---

## Sim4Life integration

> **The planner and the UI do not need Sim4Life.** Everything in this section is optional and
> applies only if you want to re-solve a montage across the whole head.

### What you need beyond `pip install`

| | |
|---|---|
| Sim4Life 9.6 | with a **QS_SOLVER licence seat** — a solve fails without one |
| the head model | `mida1010_rebuild.smash` (276 MB) plus `mida1010_rebuild.smash_Results` (7 MB). **Not in this repository** — take `tip-s4l-project` from the [team share](https://drive.google.com/drive/folders/1b-5ad3QkNmGa4wB4TmPg7sQqWaOiH34y?usp=sharing) |
| a Python with `s4l_v1` | the Sim4Life bundled interpreter, or a venv built from it |

Put the project files in a folder next to the repository:

```
<parent>/
  tip/                      ← this repository
  s4l_projects/
    mida1010_rebuild.smash
    mida1010_rebuild.smash_Results/
```

That layout is the default. If yours differs, set the environment variables below — nothing is
hard-coded any more.

| Variable | Default | What it points at |
|---|---|---|
| `TIP_S4L_PROJECTS` | `../s4l_projects` | folder holding the `.smash` files |
| `TIP_REBUILD_SMASH` | `$TIP_S4L_PROJECTS/mida1010_rebuild.smash` | the head model itself |
| `TIP_ISOLVE` | `C:\Program Files\Sim4Life_9.6\Solvers\iSolve.exe` | the solver executable |
| `TIP_S4L_PYTHON` | auto-detected | interpreter that has `s4l_v1` |
| `TIP_SCRATCH` | `outputs/scratch` | working area for large intermediates |

Check the wiring before running a montage:

```bash
python -c "import sys;sys.path.insert(0,'src');from tip.orch import s4l;import os;\
print('python:', s4l.s4l_python());print('model :', s4l.BASE_SMASH(), os.path.exists(s4l.BASE_SMASH()))"
```

Both lines must resolve and the model must report `True`.

### Using it

Press **▶ Send to Sim4Life** in the UI. The montage becomes a Sim4Life project, both channels are
solved, and the results come back for the whole head (about 7 minutes).

```
pick electrodes (leadfield, instant)  →  Sim4Life solve  →  per-tissue stats + 3 slices
```

Only available here: **scalp current density** (safety limit), fields inside skull, CSF and eye,
head resistance, and the injected current per channel.

If a target is selected, M1/M2/M3 are recomputed from the Sim4Life solution and shown next to the
leadfield values, together with a per-structure breakdown of **where** the stimulation actually
lands — the same threshold M3 uses, split by anatomical structure.

**Consistency check.** For the same montage, leadfield superposition and the Sim4Life re-solve
agree to a spatial correlation of **0.99997**. M2 and M3 match within **1%**; only M1 differs
(0.975–1.022, sign varies by montage). Since M2 and M3 are dimensionless ratios, the residual is
a pure scale factor — a discretisation bias in the current normalisation, not physics. Note that
this validates superposition and the extraction chain, **not the head model**: both paths solve
the same model.

### Result cache

**The same analysis is never computed twice.** The key is not a file name but a hash of every
input:

```
electrode pairs and polarity · current ratio and budget · leadfield set ·
fingerprint of the model file · algorithm version
```

Change any one of them and it re-solves. Adding electrodes or changing the grid alters the
project file's size and mtime, so the cache notices — **it will not quietly hand back a stale
result.** A cache hit is labelled in the UI with the date of the original run. Pass `force: true`
to bypass it.

The target is deliberately **not** part of the key: the field does not depend on it. Brain-voxel E
is kept in the cache instead, so switching targets recomputes the metrics in about a second
without re-solving.

```
GET /api/cache            list previously computed analyses
```

⚠ There is a **single Sim4Life licence seat**, so one job runs at a time (concurrent requests get
a 409). Running jobs can be cancelled; a cancelled job never writes to the cache.

---

## Requirements

| | |
|---|---|
| optimisation + UI | Python ≥ 3.10, numpy, scipy. **Sim4Life not required** |
| slice images | matplotlib |
| Sim4Life integration | Sim4Life 9.6 + licence; run `tools/s4l/` with the bundled Python |
| NEURON axon models | WSL + NEURON 9 (see [docs/PIPELINE.md](docs/PIPELINE.md)) |

---

## Documentation

| | |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | environment setup and where the data came from |
| [docs/PIPELINE.md](docs/PIPELINE.md) | integrated pipeline design (Sim4Life and NEURON backends) |
| [docs/SCALE_AUDIT.md](docs/SCALE_AUDIT.md) | ★absolute leadfield scale audit — read before quoting any absolute number |
| [research/README.md](research/README.md) | which validation script settled which question |

> Most documents under `docs/` are written in Korean and predate the current layout; the mapping
> from old to new paths is at the bottom of [docs/README.md](docs/README.md).
