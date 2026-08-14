# -*- coding: utf-8 -*-
"""tiplite_solve_one.py — solve the tip.lite reference model one electrode at a time, using our conventions

Runs inside a Sim4Life worker via `exec`. A worker restart loses the document state, so the
entire configuration procedure lives here: `prepare(path)` once, then `set_active(name)`.

Settings changed from the reference project's defaults to match our conventions:
  · `Active`   → exactly one electrode, Dirichlet **1 V**, `TreatAsPort=False`
                 (the reference default is all 70 electrodes at 0 V with a port sweep)
  · `Passive`  → **Cz** at 0 V (the reference default is TP9)
  · outer boundary → left as the reference has it, **Flux 0 (insulating)**
  · white matter and midbrain → **downgraded to isotropic**. The reference only sets the
                 aniso+inhomo flags with all-zero tensors, which makes WriteInputFile refuse:
                 `Material ... has incorrectly defined conductivity (expected: inhomogeneous tensor)`

⚠ Material flags are **silently ignored** on plain assignment because of `ValueLocked` — no
  exception is raised. See `unlock()`.
⚠ At 156.8 MCells: CreateVoxels 87 s, iSolve about 1 h 40 min, peak memory 73 GB (it swaps).
"""
import os
import time

#  Sim4Life solver. Override with TIP_ISOLVE if installed elsewhere.
ISOLVE = os.environ.get("TIP_ISOLVE") or \
    r"C:\Program Files\Sim4Life_9.6\Solvers\iSolve.exe"


def unlock(prop_raw):
    """`ValueLocked` is inherited from the parent, so collect the chain and release from the
    root down.

    `prop.raw.ReadOnly` reports False, which is easy to be fooled by. The real lock is
    `ValueLocked`, and calling `ReleaseValueLock()` on a single property does not clear it.
    """
    chain, n = [], prop_raw
    while n is not None:
        chain.append(n)
        n = n.Parent
    for x in reversed(chain):
        x.ReleaseValueLock()


def prepare(path=None):
    """Open the document (if a path is given), downgrade to isotropic, and return
    (sim, electrode dict)."""
    import s4l_v1 as s4l
    import XCoreModeling as xm
    if path:
        s4l.document.Open(path)
    doc = s4l.document
    sim = list(doc.AllSimulations)[0]

    ents = []
    def walk(g):
        for e in g.Entities:
            ents.append(e)
            if hasattr(e, "Entities"):
                walk(e)
    walk(xm.GetActiveModel().RootGroup)
    EL = {e.Name: e for e in ents if e.Name.startswith("Elec_round")}

    for c in sim.AllSettings:
        if type(c).__name__ != "MaterialSettings":
            continue
        ep = c.ElectricProps
        if ep.ConductivityAnisotropic or ep.ConductivityInhomogeneous:
            unlock(ep.ConductivityAnisotropicProp.raw)
            unlock(ep.ConductivityInhomogeneousProp.raw)
            ep.ConductivityAnisotropic = False
            ep.ConductivityInhomogeneous = False
            assert not ep.ConductivityAnisotropic, f"{c.Name}: failed to disable anisotropy"
    return sim, EL


def _assign(sim, setting, ent, entname):
    import XSimulator
    comp = sim.raw.AcquireComponent(ent)
    if not comp.AssignSettings(setting.raw):
        raise RuntimeError(f"failed to assign {entname}")
    for i in range(setting.raw.SizeAssignedComponents() - 1, -1, -1):
        c = setting.raw.AssignedComponent(i)
        if str(c.ModelEntity.Name) != entname:
            XSimulator.RemoveSettingsFromComponent(setting.raw, c)
    got = [str(setting.raw.AssignedComponent(i).ModelEntity.Name)
           for i in range(setting.raw.SizeAssignedComponents())]
    if got != [entname]:
        raise RuntimeError(f"{setting.Name} ended up as {got}")


def set_electrodes(sim, EL, active, passive="Cz"):
    """Drive `active` at 1 V and `passive` at 0 V. Names use 10-10 notation (TP8, Fp2, Cz)."""
    def key(n):
        cand = f"Elec_round {n}"
        if cand in EL:
            return cand
        # The reference model puts spaces in the names: 'Elec_round TP 8', 'Elec_round Fp 2'
        for i in range(1, len(n)):
            cand = f"Elec_round {n[:i]} {n[i:]}"
            if cand in EL:
                return cand
        raise KeyError(n)
    A = [c for c in sim.AllSettings if c.Name == "Active"][0]
    P = [c for c in sim.AllSettings if c.Name == "Passive"][0]
    ka, kp = key(active), key(passive)
    _assign(sim, A, EL[ka], ka)
    _assign(sim, P, EL[kp], kp)
    A.TreatAsPort = False
    A.DirichletValue = 1.0
    P.DirichletValue = 0.0
    return ka, kp


def write_input(sim):
    t0 = time.time(); sim.CreateVoxels(); v = time.time() - t0
    t0 = time.time(); sim.WriteInputFile(); w = time.time() - t0
    inp = sim.InputFilename
    return inp, f"CreateVoxels {v:.0f}s · WriteInputFile {w:.0f}s · {os.path.getsize(inp)/1e6:.1f} MB"
