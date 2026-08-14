# tools/ — scripts you run yourself

`src/tip/` is a library. These are the things a person invokes from the command line.
Most need Sim4Life, so run them with the **bundled Python**:

```bash
"C:\Program Files\Sim4Life_9.6\Python\python.exe" tools/s4l/<script>.py
```

> ⚠ **Never pipe their output** (`| tail` and friends). These scripts end with `os._exit()`
> because a clean shutdown takes minutes in DLL detach — and at that moment a Sim4Life child
> process still holds the pipe, so **EOF never arrives**. Redirect to a file instead.

> ⚠ There is a **single Sim4Life licence seat**. Start two of these at once and both will stall
> or die.

## s4l/ — Sim4Life

| File | What it does |
|---|---|
| `rebuild_solve_batch.py` | ★shared library: swap the driven electrode, extract the result h5, normalise the leadfield |
| `electrode_add.py` | place an electrode body at an arbitrary scalp coordinate (`ViP.PlaceElectrodes` cannot be used — vertices carry no surface parameterisation) |
| `add_electrodes.py` | unattended fill of the 10-10 electrodes that have not been solved yet (`--list` / `--dry`) |
| `s4l_montage.py` | ★export a montage as a project, solve it, and extract E over the **full grid** |
| `tiplite_batch.py` · `tiplite_solve_one.py` · `extract_tiplite_solve.py` | solve the tip.lite reference model |
| `rebuild_batch_run.py` · `identify_electrode.py` · `s4l_fibers.py` | helpers |

## analyze/ — interpreting results

| File | What it does | Environment |
|---|---|---|
| `montage_analyze.py` | TI envelope, per-tissue statistics, safety numbers, three slice images, target metrics | `tip` (needs matplotlib) |

`h5py` only exists in the Sim4Life Python and `matplotlib` only in the `tip` environment, which is
why the pipeline is split in two. Pressing "Send to Sim4Life" in the UI makes `orch/s4l.py` chain
the two stages automatically.

## prep/ — building inputs

| File | Produces |
|---|---|
| `make_mida_masks.py` · `make_stn_mask.py` | target masks → `inputs/masks/` |
| `rasterize_tiplite_targets.py` | tip.lite targets transplanted onto our grid |
| `make_angle_sweep.py` | NEURON angle-sweep cases |
| `scan_tiplite_target_match.py` | compare target definitions |

## Paths

Each file defines `REPO` / `INPUTS` / `OUTPUTS` at the top and resolves input files with
`IN("filename")`. `IN()` searches the `inputs/` subfolders (geometry, masks, fibers, leadfield),
so a script never needs to know the folder structure. The source of truth is
[src/tip/config.py](../src/tip/config.py).
