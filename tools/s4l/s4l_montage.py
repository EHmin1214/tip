# -*- coding: utf-8 -*-
"""s4l_montage.py — export a TIP montage as a Sim4Life project

Why this exists
---------------
E inside the brain is already **exact** from leadfield superposition (superposition check
6.9e-15); re-solving gives the same numbers. The value of solving again in Sim4Life is
elsewhere:

- **The whole head.** Our leadfield stores only the 1,907,678 brain voxels (18% of the grid).
  Scalp, skull, CSF, eye and neck are absent, so **safety numbers such as scalp current
  density are simply not available.**
- **J, potential and loss density** — we only store E.
- **Sim4Life's own TI post-processing** — an independent implementation, so it cross-checks
  our metrics. (Convention mismatches have produced metric bugs here three times.)

Structure
---------
TI has two carriers. The problem is quasi-static, so the channels are **solved separately**
and the envelope is combined in post-processing. Hence two simulations, `ch1` and `ch2`.
The rebuilt project is copied, so materials, grid and voxel settings are inherited exactly
(grid 185 x 254 x 228 = 10.7 MCells).

**Current convention**: the simulation is solved at 1 V Dirichlet. The real current in mA is
applied afterwards as `E · (i_k / I_inj)` — the same convention as the leadfield. The intended
currents are recorded in a sidecar JSON.

Usage (from the Sim4Life Python):
    python s4l_montage.py export <out.smash> A B C D --ratio 1.0 --itotal 2.0
"""
import json
import os
import shutil
import sys

os.environ.setdefault("S4L_API_AUTO_INIT", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
from tip.config import inputs as IN                # input-file resolver
OUTPUTS = os.path.join(REPO, "outputs")

import rebuild_solve_batch as R  # noqa: E402


from tip import config as C          # noqa: E402  read C.<name> late — the head can change

#  ★How each head's project is put together. These are **not** interchangeable: the two
#  projects were built years and species apart.
#
#      human   bodies `<name>_ElectrodeTemplate` in the group "Cloned Templates";
#              two BoundarySettings already exist, `src` (1 V) and `ref` (0 V), and the
#              electrodes that are not driven carry no material at all — they are absent.
#      rat     bodies `Elec_0.25mm <name>` in "Electrodes_0.25mm"; the project came out of a
#              **leadfield port solve**, so its settings are `Active` (37 electrodes,
#              TreatAsPort=True) and `Passive` (PO8). Driving a montage means turning the
#              port mode off, putting 1 V on one electrode and 0 V on one other, and leaving
#              the remaining 36 attached to nothing.
#
#  ⚠ That difference is not cosmetic — it changes the physics. See `rat_montage_check.py`.
CONV = {
    "human": dict(group="Cloned Templates", fmt="{}_ElectrodeTemplate",
                  src="src", ref="ref", port=False),
    "rat":   dict(group="Electrodes_0.25mm", fmt="Elec_0.25mm {}",
                  src="Active", ref="Passive", port=True),
}


def conv():
    c = CONV.get(C.MODEL_NAME)
    if c is None:
        raise KeyError(f"no Sim4Life project convention for model {C.MODEL_NAME!r} "
                       f"(known: {sorted(CONV)})")
    return c


def _ename(name):
    return conv()["fmt"].format(name)


def _grp(xm):
    g = conv()["group"]
    m = xm.GetActiveModel()
    for e in m.RootGroup.Entities:
        if e.Name == g:
            return e
    raise KeyError(f"no {g!r} group in this project")


def _elec(xm, name):
    want = _ename(name)
    #  The rat groups its electrodes one level down, so search the whole model rather than
    #  only the group's direct children.
    m = xm.GetActiveModel()
    hit = m.FindEntities(lambda e: str(e.Name) == want)
    if hit:
        return hit[0]
    raise KeyError(f"{want} not found")


def base_smash():
    """The project a montage is copied from, for the head currently selected.

    ⚠ The rat's is the leadfield project itself — there is no separate montage base yet, so
    `set_pair` has to undo its port mode. Overridable per head with TIP_REBUILD_SMASH /
    TIP_RAT_SMASH so a re-solved model can be swapped in without editing code.
    """
    proj = os.environ.get("TIP_S4L_PROJECTS") or \
        os.path.join(os.path.dirname(REPO), "s4l_projects")
    if C.MODEL_NAME == "rat":
        return os.environ.get("TIP_RAT_SMASH") or os.path.join(proj, "rat_lf.smash")
    return R.SMASH


def set_pair(sim, xm, plus, minus):
    """Move `src` (1 V) to the `plus` electrode and `ref` (0 V) to `minus`.

    `rebuild_solve_batch.set_src` hard-codes Cz as the reference, so it cannot drive a montage.
    This follows the same component-level convention — `sim.Add` / `sim.Remove` are blocked in
    Sim4Life 9.6.
    ⚠ `RemoveSettingsFromComponent` returns False even on success.
      **Attach before detaching**, otherwise there is a moment with nothing assigned.
    """
    import XSimulator                                     # noqa: N813

    def _nm(c):
        me = c.ModelEntity
        return None if me is None else str(me.Name)

    cv = conv()

    def _bset(role):
        bs = [c for c in sim.AllSettings
              if type(c).__name__ == "BoundarySettings" and c.Name == role]
        if not bs:
            raise RuntimeError(f"no boundary setting named {role!r}")
        return bs[0]

    def _assigned(bs):
        return [bs.raw.AssignedComponent(i)
                for i in range(bs.raw.SizeAssignedComponents())]

    #  ★Break any cross-role collision **first**. A simulation is cloned from channel 1 and
    #  then reassigned, so it arrives holding channel 1's electrodes. If the electrode now
    #  wanted as the anode is the one it inherited as the cathode, assigning it to the anode
    #  while it still sits on the cathode leaves it in both, and the later removal does not
    #  take — `Passive` ended up holding two electrodes and the export aborted.
    #  Only time multiplexing hits this: reusing electrodes across slots is the point of that
    #  mode, so the same name turns up in a different role in a later channel.
    for role, who in ((cv["src"], plus), (cv["ref"], minus)):
        other = _bset(cv["ref"] if role == cv["src"] else cv["src"])
        for c in _assigned(other):
            if _nm(c) == _ename(who):
                XSimulator.RemoveSettingsFromComponent(other.raw, c)

    for role, who in ((cv["src"], plus), (cv["ref"], minus)):
        bs = _bset(role)
        if role == cv["src"]:
            #  A leadfield project drives its electrodes as **ports**; a montage does not.
            #  Leaving TreatAsPort on would solve 37 separate port cases again instead of the
            #  one two-terminal problem asked for.
            if cv["port"]:
                bs.TreatAsPort = False
            bs.DirichletValue = 1.0
        tgt = _ename(who)
        old = [c for c in _assigned(bs) if _nm(c) != tgt]
        if not sim.raw.AcquireComponent(_elec(xm, who)).AssignSettings(bs.raw):
            raise RuntimeError(f"failed to assign {who} to {role}")
        for o in old:
            XSimulator.RemoveSettingsFromComponent(bs.raw, o)
        got = [_nm(c) for c in _assigned(bs)]
        if got != [tgt]:
            raise RuntimeError(
                f"{role} ended up as {got}, expected [{tgt!r}] — an electrode is probably "
                f"still held by the other role (see the cross-role purge above)")
    return True


def export(out_smash, pairs, currents=None, ratio=1.0, itotal=2.0, combine=None,
           compose="sum", duties=None, verbose=True):
    """Create a new project containing the montage. **Does not solve.**

    pairs    : list of (anode, cathode) electrode-name pairs — one Sim4Life simulation each,
               named ch1, ch2, ... Two pairs = classic TI; four = dual TI (2+2).
    currents : mA per pair, same order. Omit for the two-pair case to derive them from
               `ratio` and `itotal`, which is what the classic path has always done.
    combine  : how the channels form envelopes, e.g. [["ch1","ch2"]] for classic or
               [["ch1","ch2"],["ch3","ch4"]] for dual TI. Defaults to a single group of the
               first two channels.
    compose  : "sum" (default) — the groups run at once and their envelopes add, which is
               classic and dual TI (`optimize_dual_ti` scores `ct = etA + etB`);
               "timeavg" — time multiplexing: one slot at a time, so the analysis takes the
               duty-weighted average per direction and maximises afterwards.
    duties   : per-group duty cycle, required for "timeavg".
    returns  : {"smash": path, "meta": path, "sims": ["ch1", ...]}

    ★A channel is **one electrode pair**, because that is what a two-terminal Dirichlet solve
    can express (one electrode at 1 V, one at 0 V, the rest floating). Dual TI fits because
    its four channels are still four pairs. Distributed TI does **not** — there a channel is
    a current distribution over many electrodes, which this cannot represent at all.
    """
    import s4l_v1 as s4l
    import s4l_v1.document as doc
    import XCoreModeling as xm

    pairs = [tuple(p) for p in pairs]
    if not pairs or any(len(p) != 2 for p in pairs):
        raise ValueError("every channel must be exactly two electrodes: %r" % (pairs,))
    if currents is None:
        if len(pairs) != 2:
            raise ValueError("currents= is required for anything but a two-pair montage")
        from tip.optimize.classic import channel_currents
        currents = list(channel_currents(float(ratio), budget=float(itotal)))
    currents = [float(c) for c in currents]
    if len(currents) != len(pairs):
        raise ValueError("got %d pairs but %d currents" % (len(pairs), len(currents)))
    names = ["ch%d" % (i + 1) for i in range(len(pairs))]
    combine = [list(g) for g in (combine or [names[:2]])]
    if compose not in ("sum", "timeavg"):
        raise ValueError("compose must be 'sum' or 'timeavg', got %r" % compose)
    if compose == "timeavg":
        if not duties or len(duties) != len(combine):
            raise ValueError("timeavg needs one duty per group (%d groups, %r)"
                             % (len(combine), duties))
        #  A duty is a fraction of the period; they have to account for all of it or the
        #  reported envelope silently belongs to a different schedule than the one solved.
        if abs(sum(duties) - 1.0) > 1e-6:
            raise ValueError("duties must sum to 1, got %r (sum %.6f)"
                             % (duties, sum(duties)))
    duties = [float(w) for w in duties] if duties else None
    os.makedirs(os.path.dirname(out_smash), exist_ok=True)

    #  ★Copy the rebuilt project **at the file level first**. Using doc.SaveAs alone leaves
    #    `_Results` behind, which forces the voxels to be rebuilt (87 s).
    if os.path.exists(out_smash):
        os.remove(out_smash)
    base_path = base_smash()
    shutil.copy2(base_path, out_smash)
    src_res = base_path + "_Results"
    dst_res = out_smash + "_Results"
    if os.path.isdir(src_res):
        #  ⚠ `rmtree(..., ignore_errors=True)` swallows a locked file and leaves the directory
        #  behind, and the plain `copytree` that followed then died with
        #  `FileExistsError: montage_gui.smash_Results`. Sim4Life or a viewer holding one h5
        #  is enough to trigger it. Copy **into** whatever survives instead of demanding a
        #  clean slate — the files are overwritten either way.
        shutil.rmtree(dst_res, ignore_errors=True)
        shutil.copytree(src_res, dst_res, dirs_exist_ok=True)

    doc.Open(out_smash)
    sims = list(doc.AllSimulations)
    #  ⚠ Not `base` — that name holds the base project **path** above, and reusing it here
    #  silently turned `meta["base_project"]` into a live simulation object, which killed the
    #  whole export on `json.dump` ("ElectroQsOhmicSimulation is not JSON serializable")
    #  after the model had already been copied and opened.
    one = sims[0]

    #  Channel 1 reuses the original simulation; the rest are clones of it.
    one.Name = names[0]
    set_pair(one, xm, pairs[0][0], pairs[0][1])
    for nm, pr in zip(names[1:], pairs[1:]):
        sim = one.Clone()
        sim.Name = nm
        if sim not in list(doc.AllSimulations):
            doc.AllSimulations.Add(sim)
        set_pair(sim, xm, pr[0], pr[1])

    doc.Save()

    mont = {nm: list(pr) for nm, pr in zip(names, pairs)}
    mont["pairs"] = [list(pr) for pr in pairs]
    mont["combine"] = combine
    mont["compose"] = compose
    if duties is not None:
        mont["duties"] = duties
    mont["ratio"] = float(ratio)
    meta = {
        "montage": mont,
        "currents_mA": dict({nm: c for nm, c in zip(names, currents)},
                            itotal=float(sum(currents))),
        "convention": ("Solved at 1 V Dirichlet. Real field = E_1V * (i_k / I_inj), with "
                       "I_inj = integral of sigma|E|^2 dV. Same convention as the leadfield."),
        "grid": "inherited from the base project",
        "base_project": base_path,
        "model": C.MODEL_NAME,
    }
    mpath = out_smash.replace(".smash", "_montage.json")
    json.dump(meta, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if verbose:
        print(f"[s4l_montage] project {out_smash}")
        for nm, pr, c in zip(names, pairs, currents):
            print(f"  {nm} {pr[0]}(+) → {pr[1]}(−)  {c:.4f} mA")
        print(f"  envelope groups: {combine} · compose={compose}"
              + (f" · duties={duties}" if duties else ""))
        print(f"  simulations: {[s.Name for s in doc.AllSimulations]}")
        print(f"  metadata {mpath}")
    return {"smash": out_smash, "meta": mpath,
            "sims": [s.Name for s in doc.AllSimulations]}


def extract_full(out_h5):
    """Extract E over the **full grid**. `R.extract` returns brain voxels only, and the whole
    head is the point here, so this extends it.

    Returns: E (NX,NY,NZ,3) float32 in V/m at 1 V drive, sigma (NX,NY,NZ) float32,
    the axes in metres, and I in amperes.
    """
    import h5py
    import numpy as np
    with h5py.File(out_h5, "r") as f:
        lut, vox, ax = R._sigma_and_axes(f)
        fg = f["FieldGroups"]
        key = [x for x in fg if "EM E(x,y,z,f0)" in fg[x]["AllFields"]][0]
        sn = fg[key]["AllFields"]["EM E(x,y,z,f0)"]["_Object"]["Snapshots"]["0"]
        e0 = sn["comp0"][..., 0].astype(np.float32)
        e1 = sn["comp1"][..., 0].astype(np.float32)
        e2 = sn["comp2"][..., 0].astype(np.float32)
    # Staggered-grid edges -> cell centres, averaging the four parallel edges. Same convention
    # as `R.extract`.
    Ex = .25 * (e0[:, :-1, :-1] + e0[:, 1:, :-1] + e0[:, :-1, 1:] + e0[:, 1:, 1:]); del e0
    Ey = .25 * (e1[:-1, :, :-1] + e1[1:, :, :-1] + e1[:-1, :, 1:] + e1[1:, :, 1:]); del e1
    Ez = .25 * (e2[:-1, :-1, :] + e2[1:, :-1, :] + e2[:-1, 1:, :] + e2[1:, 1:, :]); del e2
    sig = lut[vox].astype(np.float32)
    d = [np.diff(a) for a in ax]
    E2 = (Ex.astype(np.float64) ** 2 + Ey.astype(np.float64) ** 2
          + Ez.astype(np.float64) ** 2)
    E2 = np.where(np.isfinite(E2), E2, 0.0)
    I = float(np.sum(lut[vox] * E2 *
                     (d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :])))
    del E2
    return np.stack([Ex, Ey, Ez], -1), sig, ax, I


def solve_project(smash, out_dir, verbose=True):
    """Solve the project's simulations (ch1, ch2) in turn and save E over the **full grid**.

    ⚠ `CreateVoxels` deletes `_Results`, so extraction must happen **before** the next channel
      is solved. Hence the sequential loop that extracts immediately into `out_dir`.
    """
    import subprocess
    import time
    import numpy as np
    import s4l_v1.document as doc

    os.makedirs(out_dir, exist_ok=True)
    doc.Open(smash)
    inj = {}
    for sim in list(doc.AllSimulations):
        t0 = time.time()
        sim.CreateVoxels()
        sim.WriteInputFile()
        inp = sim.InputFilename
        r = subprocess.run([R.ISOLVE, inp], cwd=os.path.dirname(inp),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{sim.Name} iSolve failed with {r.returncode}: {r.stdout[-400:]}")
        E, sig, ax, I = extract_full(inp.replace("_Input.h5", "_Output.h5"))
        np.save(os.path.join(out_dir, f"{sim.Name}_E1V.npy"), E)
        #  ★Also save the brain-voxel subset (22.9 MB per channel).
        #    The metrics (M1/M2/M3) need only this, and the full grid (128 MB) is deleted after
        #    the analysis. This file is what makes re-deriving metrics for a different target
        #    possible **without re-solving**.
        #  ⚠ Per head — `C.BMASK_FILE`, never the literal "bmask1010.npy". The human file
        #  exists whatever model is loaded, so hard-coding it does not fail on the rat: it
        #  indexes the rat's grid with human voxel indices and writes a plausible-looking
        #  `_Ebrain.npy` full of the wrong voxels. Silent, and every later metric inherits it.
        bm = np.load(IN(C.BMASK_FILE)).astype(np.int64)
        np.save(os.path.join(out_dir, f"{sim.Name}_Ebrain.npy"),
                E[bm[:, 0], bm[:, 1], bm[:, 2]].astype(np.float32))
        if not os.path.exists(os.path.join(out_dir, "sigma.npy")):
            np.save(os.path.join(out_dir, "sigma.npy"), sig)
            np.savez(os.path.join(out_dir, "axes.npz"),
                     **{f"axis_{c}": a for c, a in zip("xyz", ax)})
        inj[sim.Name] = I
        if verbose:
            print(f"[s4l_montage] {sim.Name} done · {time.time()-t0:.0f}s · "
                  f"I={I*1e3:.4f} mA · E {E.shape}")
        del E, sig
    json.dump(inj, open(os.path.join(out_dir, "inj.json"), "w"), indent=1)
    return inj


def cleanup_shadows(smash):
    """Delete Sim4Life autosave shadow files — about 250 MB accumulates per montage."""
    d, base = os.path.dirname(smash), os.path.basename(smash)
    n = 0
    for f in os.listdir(d):
        if f != base and f.startswith((base + ".", "." + base + ".")):
            try:
                os.remove(os.path.join(d, f)); n += 1
            except OSError:
                pass
    return n


def main(argv):
    if not argv:
        print(__doc__); return 1
    if argv[0] == "export":
        out = argv[1]
        ratio = float(argv[argv.index("--ratio") + 1]) if "--ratio" in argv else 1.0
        itot = float(argv[argv.index("--itotal") + 1]) if "--itotal" in argv else 2.0
        #  Two ways in. `--pairs` is the general one (any number of channels, explicit
        #  currents); the four positional electrodes are the classic two-pair form and are
        #  kept working unchanged.
        if "--pairs" in argv:
            pairs = json.loads(argv[argv.index("--pairs") + 1])
            cur = json.loads(argv[argv.index("--currents") + 1])                 if "--currents" in argv else None
            comb = json.loads(argv[argv.index("--combine") + 1])                 if "--combine" in argv else None
            comp = argv[argv.index("--compose") + 1] if "--compose" in argv else "sum"
            dut = json.loads(argv[argv.index("--duties") + 1]) if "--duties" in argv else None
            export(out, pairs, currents=cur, ratio=ratio, itotal=itot, combine=comb,
                   compose=comp, duties=dut)
        else:
            a, b, c, d = argv[2:6]
            export(out, [(a, b), (c, d)], ratio=ratio, itotal=itot)
        if "--solve" in argv:
            odir = argv[argv.index("--solve") + 1]
            solve_project(out, odir)
        print(f"cleaned up {cleanup_shadows(out)} shadow file(s)")
        print("=== END · export complete")    # orch completion marker
    elif argv[0] == "solve":
        solve_project(argv[1], argv[2])
        print(f"cleaned up {cleanup_shadows(argv[1])} shadow file(s)")
        print("=== END · solve complete")
    else:
        print(__doc__); return 1
    os._exit(0)                            # a clean shutdown takes minutes in DLL detach


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
