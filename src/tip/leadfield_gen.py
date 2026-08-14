# -*- coding: utf-8 -*-
"""
leadfield_gen.py -- Sim4Life leadfield extraction pipeline (VERIFIED bit-exact)
===============================================================================
Reverse-engineered + empirically verified reproduction of the existing
`leadfieldF/M{jj}.npy` ground-truth data from a solved Sim4Life
ElectroQsOhmicSimulation (electro-quasistatic, solves div(sigma grad phi)=0).

VERIFICATION (2026-07-23, stored `LDF` result):
  The `LDF` simulation had a stored result (electrode TP8, enames index 60).
  Extracting it via the mapping below and comparing to `leadfieldF/M60.npy`
  gave  max_abs_err = 0.0 ,  rel_err = 0.0  (BIT-EXACT, no new solve needed).
  Boundary assignment independently confirmed the electrode:
    src (Dirichlet 1 V) -> TP8_ElectrodeTemplate (id 9f12243e-...)
    ref (Dirichlet 0 V) -> Cz_ElectrodeTemplate  (id 1d828164-...)
  Injected current (for unitnorm) independently confirmed:
    integral(El. Loss Density) dV = 6.425016878572375e-4  vs
    inj_current["60"]            = 6.425016878572374e-4   (exact to float64).

END-TO-END SOLVE validation (2026-07-23): a clone of LDF driven via
  drive_electrode(clone,"TP8") + solve() (localhost, ~112 s wall-clock,
  CreateVoxels ~18 s) reproduced M60.npy with max_abs=0.0, rel_err=0.0
  (bit-exact -- deterministic solver/mesh). License OK.

FLOATING-ELECTRODE MECHANISM (empirical, from the solver input .h5):
  The head is voxeled purely as MIDA tissues. Electrode bodies add NOTHING to
  the conductivity grid (even the TP8/Cz "materials" occupy 0 voxels). There is
  NO PEC/metal material (max sigma = 1.79 S/m = CSF). The driven electrode and
  Cz are realized ONLY as Dirichlet BCs on their body footprints (1 V / 0 V);
  6 more Dirichlet BCs ground the outer domain-box faces at 0 V. The other 59
  electrodes carry no BC and no material -> TRANSPARENT (their cells are plain
  scalp). "60 basis: i=1V, Cz=0V, other 59 floating" is really "other 59 do
  nothing"; only the DRIVEN electrode's footprint matters. See drive_electrode.

----------------------------------------------------------------------------
PROVEN grid -> bmask1010 mapping (the whole point -- verify, never assume)
----------------------------------------------------------------------------
* The sim uses a NON-UNIFORM (graded) rectilinear grid:
    grid POINTS  : 186 x 255 x 229   (edges, in METERS)
    grid CELLS   : 185 x 254 x 228 = 10,713,720   <- E-field lives HERE
  The E-field output `EM E(x,y,z,f0)` has ValueLocation = 1 (CELL CENTERS),
  NumberOfTuples = 10,713,720, NumberOfComponents = 3 (Ex,Ey,Ez), complex64.

* Cell-center coords (meters) == gaxes1010 axes (mm) EXACTLY:
    cx[i] = 1000 * (XAxis[i] + XAxis[i+1]) / 2   (max abs diff vs gaxes cx = 0.0)
    cy[j] = 1000 * (YAxis[j] + YAxis[j+1]) / 2
    cz[k] = 1000 * (ZAxis[k] + ZAxis[k+1]) / 2
  So gaxes1010.cx/cy/cz ARE the sim's cell-center coordinates in mm.

* Cell flattening order is X-FASTEST (verified via Grid.ComputeCellIndex):
        cell_index(i,j,k) = i + NX*j + NX*NY*k        (NX=185, NY=254)
    ComputeCellIndex(1,0,0)=1 ; (0,1,0)=185 ; (0,0,1)=46990=185*254.

* bmask1010.npy[r] = (i, j, k) cell indices of brain voxel r (N=1,907,678).
  Therefore:
        M{jj}[r] = Re( E_flat[ i + 185*j + 185*254*k ] )    with (i,j,k)=bmask[r]
  i.e. gather the full-grid cell-center E-field at each bmask row's cell index,
  take the REAL part (imag is exactly 0 for a real Dirichlet EQS solve),
  cast to float32. Cells outside the domain are NaN but every bmask cell is
  finite (brain voxels are all inside the solved region).

  `M{jj}.npy` is the RAW 1 V-Dirichlet basis field (V/m). The 1 mA-normalized
  field used by tip.LeadField is  M{jj} * unitnorm[jj],  unitnorm[jj] =
  1e-3 / inj_current[jj].

----------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------
Run INSIDE the Sim4Life python session (s4l_v1 importable). Typical flow:

    import s4l_v1.document as document
    from tip import leadfield_gen as G

    sim = G.get_simulation("LDF")          # or "LD"

    # (A) extract an already-solved result (cheapest -- no solve):
    M, inj = G.extract_leadfield(sim)      # M:(1907678,3) f32, inj: float (A)
    unitnorm = 1e-3 / inj

    # (B) drive + solve one electrode, then extract:
    G.drive_electrode(sim, "PO7")          # PO7=1V, Cz=0V, others untouched
    G.solve(sim)                           # blocks until done
    M, inj = G.extract_leadfield(sim)

The pure-numpy mapping (`ijk_to_flat`, `sample_field_to_bmask`) has no Sim4Life
dependency and is unit-testable standalone.
"""
from __future__ import annotations
import os
import numpy as np

# ----------------------------------------------------------------------------
# Grid constants (verified against the LDF simulation grid)
# ----------------------------------------------------------------------------
CELL_DIMS  = (185, 254, 228)                 # NX, NY, NZ  (E-field cell grid)
POINT_DIMS = (186, 255, 229)                 # grid points (edges)
N_CELLS    = CELL_DIMS[0] * CELL_DIMS[1] * CELL_DIMS[2]   # 10,713,720
N_BRAIN    = 1_907_678                        # bmask rows (informational)

# Names of the two fixed Dirichlet boundaries in the template sim.
SRC_NAME = "src"      # driven electrode, Dirichlet 1 V
REF_NAME = "ref"      # reference electrode (Cz),  Dirichlet 0 V
REF_ELEC = "Cz"
ELECTRODE_SUFFIX = "_ElectrodeTemplate"

# Sensor / output keys inside the SimulationExtractor result tree.
OVERALL_FIELD_KEY = "Overall Field"
EFIELD_KEY        = "EM E(x,y,z,f0)"
LOSS_KEY          = "El. Loss Density(x,y,z,f0)"


# ============================================================================
# 1. PURE MAPPING  (numpy only -- the proven grid<->bmask correspondence)
# ============================================================================
def ijk_to_flat(bmask: np.ndarray, dims=CELL_DIMS) -> np.ndarray:
    """(i,j,k) cell indices -> flat cell index, X-fastest.

    flat = i + NX*j + NX*NY*k.  `bmask` is (N,3) int; returns (N,) int64.
    This is exactly Sim4Life RectilinearGrid.ComputeCellIndex(i,j,k).
    """
    nx, ny, _ = dims
    b = np.asarray(bmask, dtype=np.int64)
    return b[:, 0] + nx * b[:, 1] + (nx * ny) * b[:, 2]


def flat_to_ijk(flat: np.ndarray, dims=CELL_DIMS) -> np.ndarray:
    """Inverse of `ijk_to_flat`. flat (M,) -> (M,3) int64 (i,j,k)."""
    nx, ny, _ = dims
    f = np.asarray(flat, dtype=np.int64)
    k = f // (nx * ny)
    rem = f - k * (nx * ny)
    j = rem // nx
    i = rem - j * nx
    return np.stack([i, j, k], axis=1)


def sample_field_to_bmask(field_flat: np.ndarray, bmask: np.ndarray,
                          dims=CELL_DIMS) -> np.ndarray:
    """Gather full-grid cell field onto bmask row order.

    field_flat : (N_CELLS, C) complex or real -- cell-center field, X-fastest.
    bmask      : (N, 3) int cell indices.
    returns    : (N, C) float32  (real part; imag must be ~0).
    Raises if any selected cell is non-finite (would indicate a bad mapping).
    """
    idx = ijk_to_flat(bmask, dims)
    sub = np.asarray(field_flat)[idx]
    if np.iscomplexobj(sub):
        imag_max = np.abs(sub.imag).max() if sub.size else 0.0
        if imag_max > 1e-6 * (np.abs(sub.real).max() + 1e-30):
            raise ValueError(f"non-negligible imaginary part: {imag_max}")
        sub = sub.real
    if not np.isfinite(sub).all():
        raise ValueError("selected cells contain NaN/Inf -- mapping mismatch")
    return np.ascontiguousarray(sub, dtype=np.float32)


def cell_volumes(x_axis_m, y_axis_m, z_axis_m, flat_idx=None,
                 dims=CELL_DIMS) -> np.ndarray:
    """Cell volumes (m^3) in flat X-fastest order.

    x/y/z_axis_m : grid POINT coordinates in METERS (len NX+1, NY+1, NZ+1).
    flat_idx     : optional (M,) flat cell indices to restrict to; None=all.
    """
    dx = np.diff(np.asarray(x_axis_m, dtype=np.float64))   # (NX,)
    dy = np.diff(np.asarray(y_axis_m, dtype=np.float64))   # (NY,)
    dz = np.diff(np.asarray(z_axis_m, dtype=np.float64))   # (NZ,)
    if flat_idx is None:
        flat_idx = np.arange(dims[0] * dims[1] * dims[2], dtype=np.int64)
    ijk = flat_to_ijk(flat_idx, dims)
    return dx[ijk[:, 0]] * dy[ijk[:, 1]] * dz[ijk[:, 2]]


# ============================================================================
# 2. SIM4LIFE ACCESS  (imported lazily so the mapping code above stays pure)
# ============================================================================
def get_simulation(name: str):
    """Return the simulation object called `name` from the open document."""
    import s4l_v1.document as document
    for s in document.AllSimulations:
        if s.Name == name:
            return s
    raise KeyError(f"simulation {name!r} not found "
                   f"({[s.Name for s in document.AllSimulations]})")


def _overall_field(sim):
    """The 'Overall Field' EmSensorExtractor from a solved simulation."""
    if not sim.HasResults():
        raise RuntimeError(f"simulation {sim.Name!r} has no results -- solve first")
    res = sim.Results()
    of = res[OVERALL_FIELD_KEY]
    of.Update()
    return of


def get_efield(sim):
    """Solved E-field of `sim`.

    returns (E_flat, grid) where
      E_flat : (N_CELLS, 3) complex64 cell-center field (may contain NaN outside
               the solved domain),
      grid   : the RectilinearGrid (for axes / cell volumes).
    """
    of = _overall_field(sim)
    out = of.Outputs[EFIELD_KEY]
    out.Update()
    data = out.Data
    E = np.array(data.Field(0), copy=True)          # (N_CELLS, 3) complex64
    return E, data.Grid


def get_loss_density(sim):
    """Solved ohmic loss density sigma|E|^2 (W/m^3), (N_CELLS,1) float32 real."""
    of = _overall_field(sim)
    out = of.Outputs[LOSS_KEY]
    out.Update()
    return np.array(out.Data.Field(0), copy=True).real.astype(np.float64).ravel()


def grid_axes_m(grid):
    """(x,y,z) grid POINT axes in METERS from a RectilinearGrid."""
    return (np.array(grid.XAxis), np.array(grid.YAxis), np.array(grid.ZAxis))


def injected_current(sim) -> float:
    """Injected current (A) at the 1 V-driven electrode.

    For a 1 V Dirichlet EQS solve the total ohmic power equals I*V = I*1V,
    so  I = integral( sigma|E|^2 ) dV  over the whole domain. Verified to match
    inj_current.json to float64 precision.  unitnorm = 1e-3 / I.
    """
    loss = get_loss_density(sim)                    # (N_CELLS,)
    _, grid = None, _overall_field(sim).Outputs[EFIELD_KEY]
    grid.Update()
    g = grid.Data.Grid
    xa, ya, za = grid_axes_m(g)
    finite = np.isfinite(loss)
    vol = cell_volumes(xa, ya, za)                  # (N_CELLS,) m^3
    return float(np.nansum(loss[finite] * vol[finite]))


def load_bmask(data_dir=None) -> np.ndarray:
    """bmask1010.npy (N,3) int cell indices."""
    if data_dir is None:
        from . import config as C
        data_dir = C.DATA_DIR
    return np.load(C.inputs("bmask1010.npy") if data_dir is None else os.path.join(data_dir, "bmask1010.npy"))


def extract_leadfield(sim, bmask=None, data_dir=None):
    """Full extraction: solved sim -> (M, inj_current).

    M   : (N,3) float32 raw 1 V-Dirichlet basis E-field in bmask row order,
          identical convention to leadfieldF/M{jj}.npy.
    inj : injected current (A); unitnorm = 1e-3 / inj.
    """
    if bmask is None:
        bmask = load_bmask(data_dir)
    E, grid = get_efield(sim)
    M = sample_field_to_bmask(E, bmask)
    inj = injected_current(sim)
    return M, inj


# ============================================================================
# 3. DRIVE + SOLVE  (set boundary conditions for one electrode)
# ============================================================================
def _find_entity(name: str):
    import XCoreModeling as xm
    for e in xm.GetActiveModel().GetEntities():
        if e.Name == name:
            return e
    raise KeyError(f"entity {name!r} not found")


def electrode_entity(ename: str):
    """Body entity for electrode `ename` (e.g. 'PO7' -> PO7_ElectrodeTemplate)."""
    return _find_entity(ename + ELECTRODE_SUFFIX)


def _remove_boundaries_named(sim, name):
    for c in list(sim.AllSettings):
        if type(c).__name__ == "BoundarySettings" and c.Name == name:
            try:
                sim.RemoveSettings(c)
            except Exception:
                pass


def set_dirichlet(sim, entity, value: float, name: str):
    """Create a Dirichlet boundary of `value` volts on `entity`, labelled `name`."""
    import s4l_v1.simulation.emlf as emlf
    b = emlf.BoundarySettings()          # BoundaryType defaults to Dirichlet
    b.Name = name
    try:
        b.BoundaryType = emlf.EnumBoundaryType.kDirichlet   # explicit, if supported
    except Exception:
        pass                              # default is already Dirichlet
    b.DirichletValue = float(value)
    sim.Add(b, [entity])
    return b


def drive_electrode(sim, ename: str, ref: str = REF_ELEC):
    """Configure `sim` to drive electrode `ename` at 1 V, `ref` (Cz) at 0 V.

    Only the src/ref Dirichlet boundaries are (re)written; everything else --
    the 6 outer domain-box faces grounded at 0 V ('Boundary Settings'), the
    tissue materials, grid, voxeler and sensor -- is inherited unchanged from
    the template simulation. Does NOT solve.

    VERIFIED end-to-end (2026-07-23): a clone driven with drive_electrode(clone,
    "TP8") + solve() reproduced leadfieldF/M60.npy with max_abs=0.0, rel_err=0.0
    (bit-exact -- same solver/mesh is deterministic). Confirmed BC pattern:
        src = Dirichlet 1 V on {ename}_ElectrodeTemplate  (body footprint)
        ref = Dirichlet 0 V on {ref}_ElectrodeTemplate
        + 6 outer box faces (X+/-,Y+/-,Z+/-) Dirichlet 0 V (untouched here).

    FLOATING ELECTRODES: the other 59 electrode bodies need NO material and NO
    boundary. Empirically they are TRANSPARENT -- voxeled but assigned no
    material, so their cells fall through to the underlying scalp tissue
    (Epidermis_Dermis / Adipose) exactly as bare scalp. They contribute NO
    distinct conductor and do NOT redistribute current. (There is no PEC/metal
    material anywhere; max sigma in the model = 1.79 S/m = CSF.) Consequently a
    non-driven electrode's size has zero effect; only the DRIVEN electrode's
    1 V footprint matters. To keep a new/resized electrode consistent: add its
    body to the voxeler, assign NO material, and give it the src Dirichlet 1 V
    only when it is the one being driven.
    """
    _remove_boundaries_named(sim, SRC_NAME)
    _remove_boundaries_named(sim, REF_NAME)
    set_dirichlet(sim, electrode_entity(ename), 1.0, SRC_NAME)
    set_dirichlet(sim, electrode_entity(ref),   0.0, REF_NAME)


def solve(sim, wait: bool = True, server_id=None, create_voxels: bool = True):
    """Solve `sim` (blocks until finished when wait=True). Returns the run future.

    IMPORTANT: a freshly-configured / cloned simulation MUST have its voxels
    (re)generated before RunSimulation, otherwise the run aborts instantly with
    "Unable to run simulation. Please check your simulation settings." Hence
    create_voxels=True by default (CreateVoxels ~18 s for this 10.7 MCell grid).

    RunSimulation signature is (wait, server_id, wait_for_submission); server_id
    is a server UUID string (e.g. GetAvailableServers()[0].Id, 'localhost').
    A full TP8 solve on localhost took ~112 s wall-clock.
    """
    if create_voxels:
        sim.CreateVoxels()
    sim.WriteInputFile()
    if server_id is None:
        servers = sim.GetAvailableServers()
        server_id = servers[0].Id if servers else None
    return sim.RunSimulation(wait, server_id)


# ============================================================================
# 4. VERIFICATION HELPER
# ============================================================================
def compare_to_reference(M: np.ndarray, ref_path: str):
    """Return (max_abs_err, rel_err) of reconstructed M vs a reference M{jj}.npy."""
    ref = np.load(ref_path)
    num = float(np.abs(M - ref).max())
    den = float(np.abs(ref).max())
    return num, num / den if den else float("inf")


def identify_result(sim, data_dir=None):
    """Match a stored result against every leadfieldF/M{jj}.npy.

    Returns (best_jj, rel_err, max_abs). rel_err ~ 0 pins down which electrode
    the stored result is. Used during verification (found jj=60 = TP8, rel=0).
    """
    import glob, re
    if data_dir is None:
        from . import config as C
        data_dir = C.DATA_DIR
    M, _ = extract_leadfield(sim, data_dir=data_dir)
    best = None
    for f in sorted(glob.glob(os.path.join(C.LEADFIELD_DIR, "M*.npy"))):
        jj = int(re.findall(r"M(\d+)\.npy$", os.path.basename(f))[0])
        num, rel = compare_to_reference(M, f)
        if best is None or rel < best[1]:
            best = (jj, rel, num)
    return best
