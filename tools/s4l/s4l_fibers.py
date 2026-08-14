# -*- coding: utf-8 -*-
"""
s4l_fibers.py — fibre trajectory generation on the Sim4Life side (run via `s4l_run_python`)
==========================================================================
Covers stage 1 (trajectory generation) and stage 5 (GAF evaluation) of the pipeline in
SETUP.md §8. It exchanges `.npz` files with the local side (`tip/fiberlead.py`).

    1. [S4L]   ViP.GenerateSplines → trajectories (F,N,3)  make_fibers / fibers_around  ★once
    2. [local] per-electrode Ve → fibre leadfield          build_fiber_leadfield        ★once
    3. [local] montage → envelope → exhaustive search      FiberLeadField.envelope      every time
    4. [local] export the top N as Ve                      export_candidates
    5. [S4L]   GAF thresholds and recruitment              evaluate_candidates    (licence needed)

Usage (from an MCP session):
import os
import sys
# ── repo paths ── derived from __file__ so moving the file does not break them
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))   # the tip package
INPUTS = os.path.join(REPO, "inputs")
OUTPUTS = os.path.join(REPO, "outputs")

    import sys; sys.path.insert(0, os.path.join(REPO, "src"))
    from s4l_fibers import check_coordinate_frame, fibers_around
    check_coordinate_frame()
    fibers_around("data/fibers_hippoL.npz", center=[-51.7, 259.7, 28.3],
                  direction=[0.17, -0.486, -0.857], length_mm=20, radius_mm=6, num_lines=300)

Coordinate convention: the Sim4Life frame equals the leadfield frame (confirmed 2026-08-03,
MIGRATION_STATUS.md §2-2c).
**MIDA's NIfTI (.nii) is in a different frame** — always use `.sab` for comparison and import.
"""
import os
import numpy as np


MIDA_SAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "MIDA (Static) - 1.0", "MIDA_v1.0.sab")
# Leadfield coordinate extent, computed from the input data — the reference for the
# frame-agreement check
LF_BBOX = (np.array([-95.0, 230.5, -55.3]), np.array([38.1, 343.6, 115.8]))


# ────────────────────────── model ──────────────────────────
def import_mida(sab_path=None, verbose=True):
    """Import the MIDA `.sab` into the current document and return {name: Entity}, reusing it
    if already present."""
    import XCoreModeling as xm
    import s4l_v1
    sab_path = sab_path or MIDA_SAB
    ents = {}
    try:
        for e in xm.GetActiveModel().GetEntities():
            if e.Name.lower().startswith("mida"):
                ents = {c.Name: c for c in e.Entities}
                break
    except Exception:
        pass
    if not ents:
        if not os.path.exists(sab_path):
            raise FileNotFoundError(f"MIDA model not found: {sab_path}")
        try:
            xm.GetActiveModel()
        except Exception:
            s4l_v1.document.New()
        root = xm.Import(sab_path)[0]
        ents = {c.Name: c for c in root.Entities}
    if verbose:
        print(f"[s4l_fibers] {len(ents)} MIDA tissues")
    return ents


def check_coordinate_frame(sab_path=None, lf_bbox=None, verbose=True):
    """★Step 1 — is the MIDA coordinate system the same space as the leadfield? If this fails,
    everything downstream is meaningless."""
    import XCoreModeling as xm
    ents = import_mida(sab_path, verbose=False)
    bb = xm.GetBoundingBox(list(ents.values()))
    lo, hi = (np.array([bb[0][i] for i in range(3)]), np.array([bb[1][i] for i in range(3)]))
    llo, lhi = lf_bbox or LF_BBOX
    ov = np.minimum(hi, lhi) - np.maximum(lo, llo)
    ok = bool((ov > 0).all())
    if verbose:
        print(f"[s4l_fibers] MIDA  bbox: {lo.round(1)} ~ {hi.round(1)}")
        print(f"[s4l_fibers] leadfield : {llo.round(1)} ~ {lhi.round(1)}")
        print(f"[s4l_fibers] overlap per axis: {ov.round(1)} mm  → "
              f"{'pass' if ok else 'mismatch (a transform is needed)'}")
    return ok, (lo, hi)


# ────────────────────────── trajectories ──────────────────────────
def _resample_uniform(pts, n_nodes):
    """(M,3) polyline → (n_nodes,3) resampled at equal arc length, so every trajectory has the
    same node count."""
    pts = np.asarray(pts, float)
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if d[-1] <= 0:
        return np.repeat(pts[:1], n_nodes, axis=0)
    q = np.linspace(0.0, d[-1], n_nodes)
    return np.stack([np.interp(q, d, pts[:, k]) for k in range(3)], axis=1)


def make_fibers(centers, normals, radii, num_lines=300, n_nodes=31,
                spline_sampling_distance=0.5, target_length=-1.0,
                target_length_tol_percent=20.0, mask=None, max_failures=10000,
                out_path=None, seed_note="", verbose=True):
    """Generate a bundle of trajectories with `ViP.GenerateSplines` and normalise to (F,N,3).

    centers/normals/radii : the sequence of discs that defines the bundle, laid along its axis
    mask                  : an XCoreModeling.LabelField (white matter, say) — if given, the
                            trajectories are confined to it
    n_nodes               : every trajectory is resampled to this many equally spaced nodes,
                            which the fibre leadfield assumes
    """
    import ViP
    import XCoreModeling as xm
    import s4l_v1

    def _v3(a):
        return [s4l_v1.Vec3(float(x[0]), float(x[1]), float(x[2])) for x in np.asarray(a, float)]

    centers = np.asarray(centers, float)
    normals = np.asarray(normals, float)
    radii = [float(r) for r in np.atleast_1d(radii)]
    if len(radii) == 1:
        radii = radii * len(centers)

    kw = dict(num_lines=int(num_lines),
              spline_sampling_distance=float(spline_sampling_distance),
              target_length=float(target_length),
              target_length_tol_percent=float(target_length_tol_percent),
              max_failures=int(max_failures))
    if mask is not None:
        kw["mask"] = mask
    splines = ViP.GenerateSplines(_v3(centers), _v3(normals), radii, **kw)
    if verbose:
        print(f"[s4l_fibers] GenerateSplines → {len(splines)} trajectories"
              f"{' (mask applied)' if mask is not None else ''}")
    if not splines:
        raise RuntimeError("no trajectories generated - check the disc radius, mask and max_failures")

    polys = xm.ConvertToPolyLine(list(splines))
    if not isinstance(polys, (list, tuple)):
        polys = [polys]
    trajs, lens = [], []
    for p in polys:
        pts = np.array([[q[0], q[1], q[2]] for q in xm.GetPolylinePoints(p)], float)
        if len(pts) < 2:
            continue
        lens.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
        trajs.append(_resample_uniform(pts, n_nodes))
    trajs = np.stack(trajs)
    if verbose:
        L = np.array(lens)
        print(f"[s4l_fibers] trajectories {trajs.shape} · length {L.mean():.1f} ± {L.std():.1f} mm "
              f"(min {L.min():.1f} · max {L.max():.1f})")
        print(f"[s4l_fibers] bbox {trajs.reshape(-1,3).min(0).round(1)} ~ "
              f"{trajs.reshape(-1,3).max(0).round(1)}")
    if out_path:
        np.savez_compressed(out_path, trajs=trajs, lengths=np.array(lens),
                            centers=centers, normals=normals, radii=np.array(radii),
                            note=np.array(seed_note))
        if verbose:
            print(f"[s4l_fibers] saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return trajs


def fibers_around(out_path, center, direction, length_mm=20.0, radius_mm=6.0,
                  num_lines=300, n_disks=5, n_nodes=31, mask_name=None,
                  sab_path=None, verbose=True, **kw):
    """A bundle around a target: lay out discs from a centre, direction, length and radius,
    then call `make_fibers`.

    mask_name : a MIDA tissue name (e.g. "Brain White Matter"). This needs a LabelField, but
                `.sab` only provides TriangleMesh, so it is currently **unsupported and we
                proceed without a mask** (the route SETUP.md §8-3 Step 2 permits). Building a
                LabelField would require the Voxeler.
    """
    n = np.asarray(direction, float); n /= np.linalg.norm(n)
    c = np.asarray(center, float)
    t = np.linspace(-length_mm / 2, length_mm / 2, n_disks)
    centers = c[None, :] + t[:, None] * n[None, :]
    normals = np.repeat(n[None, :], n_disks, axis=0)

    mask = None
    if mask_name:
        ents = import_mida(sab_path, verbose=False)
        ent = ents.get(mask_name)
        import XCoreModeling as xm
        if ent is not None and xm.IsLabelField(ent):
            mask = ent
        elif verbose:
            print(f"[s4l_fibers] '{mask_name}' is not a LabelField "
                  f"({type(ent).__name__ if ent is not None else 'missing'}) → proceeding without a mask")

    if verbose:
        print(f"[s4l_fibers] bundle: centre {c.round(1)} · direction {n.round(3)} · "
              f"length {length_mm} mm · radius {radius_mm} mm · {num_lines} lines")
    return make_fibers(centers, normals, [radius_mm] * n_disks, num_lines=num_lines,
                       n_nodes=n_nodes, target_length=length_mm, mask=mask,
                       out_path=out_path, verbose=verbose, **kw)


def fibers_scatter(out_path, centers, directions, length_mm=20.0, radius_mm=3.0,
                   lines_per_seed=12, n_disks=3, n_nodes=31, verbose=True, **kw):
    """Build a small bundle at each of several seeds and merge them — this produces the
    **off-target population**.

    centers/directions : the two (K,3) arrays returned by `fiberlead.sample_seeds()`.
    A seed that fails to produce a bundle is skipped rather than fatal — this happens fairly
    often near the brain boundary.
    """
    centers = np.asarray(centers, float)
    directions = np.asarray(directions, float)
    allt, owner, fails = [], [], 0
    for k, (c, n) in enumerate(zip(centers, directions)):
        try:
            t = fibers_around(None, center=c, direction=n, length_mm=length_mm,
                              radius_mm=radius_mm, num_lines=lines_per_seed,
                              n_disks=n_disks, n_nodes=n_nodes, verbose=False, **kw)
            allt.append(t)
            owner.append(np.full(len(t), k))
        except Exception as e:
            fails += 1
            if verbose and fails <= 3:
                print(f"[s4l_fibers] seed {k} failed: {type(e).__name__} {str(e)[:80]}")
        if verbose and (k + 1) % 20 == 0:
            print(f"[s4l_fibers] seed {k+1}/{len(centers)} · fibres so far "
                  f"{sum(len(x) for x in allt)}")
    if not allt:
        raise RuntimeError("trajectory generation failed at every seed")
    trajs = np.concatenate(allt)
    owner = np.concatenate(owner)
    if verbose:
        print(f"[s4l_fibers] off population {trajs.shape} · "
              f"{len(centers)-fails}/{len(centers)} seeds succeeded")
    if out_path:
        np.savez_compressed(out_path, trajs=trajs, owner=owner,
                            centers=centers, directions=directions)
        if verbose:
            print(f"[s4l_fibers] saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return trajs


# ────────────────────── stage 5: GAF evaluation (TODO) ──────────────────────
def evaluate_candidates(cand_npz, out_path=None, verbose=True):
    """Evaluate the candidate Ve exported in stage 4 with a Sim4Life neuron model.

    TODO — **not implemented: there is no T-Neuro licence.** (MIGRATION_STATUS.md §2-2)
    None of the 17 available licence features is a NEURO one, so the `NeuronYale` and
    `neuron_s4l` components were never installed. Once a licence exists, fill in:

      1. build Ve(t) = g·[Ve1·cos(2*pi*f1*t) + Ve2·cos(2*pi*f2*t)]
         → `NeuronSimulator.NeuronSetupSettings` + `NeuronLineSensorSettings`
      2. titration → a per-fibre threshold g*
         → `PerformTitration=True`, `TitrationStrategy=kEstimator` (estimated via the GAF path)
         → the result is `NeuronPostPro.TitrationEvaluator.TitrationFactor`
      3. amplitude sweep → recruitment curve
         → `NeuronPostPro.RecruitmentEvaluator` (the Recruitment Curve Evaluator announced in 9.6)
      Axons are created with `NeuronModeling.CreateAxonNeuron(spline, MotorMrgNeuronProperties(), ...)`.

    Then check whether the NEURON results in SETUP.md §7 reproduce (hippocampus 24/24 af_opt,
    cortex field_opt).
    """
    raise NotImplementedError(
        "Stage 5 (GAF / titration) requires a T-Neuro licence. The current licence has no NEURO "
        "feature, so NeuronYale and neuron_s4l are not installed. "
        "See MIGRATION_STATUS.md §2-2.")
