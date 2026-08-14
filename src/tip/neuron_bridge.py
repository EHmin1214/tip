# -*- coding: utf-8 -*-
"""
neuron_bridge.py — bridge from TI fields to NEURON
=================================================
Exports the two carrier fields computed from our leadfield and montage as the
**extracellular potential V_e** that NEURON needs in order to drive TI stimulation.

The physics:
  · V_e,k(x) is the quasi-static extracellular potential of carrier k. Our leadfield only
    provides E (= -grad V), so we integrate along the axon trajectory:
    V_e(s) = -integral_0^s E·dl. The result is path-independent and only the axial component
    contributes.
  · NEURON: e_extracellular_i(t) = g·[V_e1(xᵢ)·cos(2πf₁t) + V_e2(xᵢ)·cos(2πf₂t)]
    (g is an overall gain, scaled up to threshold). The neuron's sodium nonlinearity
    rectifies and thereby demodulates the low-frequency envelope.

Units: coordinates are mm and E is V/m, so V_e comes out in volts; multiply by 1e3 for
NEURON's millivolts.
(Absolute thresholds depend on units and the axon model, but the **relative comparison of a
field-optimal versus an AF-optimal montage** is robust.)
"""
import numpy as np
from .report import _montage_fields


def carrier_fields_at(lf, best, pts):
    """The two carrier fields E1, E2 (N,3) at arbitrary 3D points pts (N,3), by **trilinear
    interpolation** (no staircase; see §6).
    `best` is either a classic-style montage (ch1/ch2/ratio) or a distributed one (currents)."""
    from .fieldsample import interp_apply
    return interp_apply(lf, lambda ix: _montage_fields(lf, best, ix), np.asarray(pts, float))


def axon_potentials(lf, best, traj):
    """Extracellular potentials V_e1, V_e2 (N,) in mV for each carrier, along the ordered axon
    trajectory traj (N,3).
    V_e(s) = -integral E·dl, accumulated. The trajectory spacing sets the potential's spatial
    resolution."""
    traj = np.asarray(traj, float)
    E1, E2 = carrier_fields_at(lf, best, traj)
    dl = np.zeros_like(traj); dl[1:] = np.diff(traj, axis=0)          # step vectors, mm

    def cum(E):
        edl = np.sum(E * dl, axis=1) * 1e-3                            # V (E[V/m]·dl[m])
        v = -np.cumsum(edl)
        return (v - v.mean()) * 1e3                                    # mV, zero-mean reference

    return cum(E1), cum(E2)


def straight_axon(center, direction, length_mm=20.0, ds_mm=0.25):
    """Coordinates (N,3) and arc length s (N,) of a straight axon centred at `center` and
    running along `direction`."""
    d = np.asarray(direction, float); d = d / (np.linalg.norm(d) + 1e-30)
    n = int(round(length_mm / ds_mm)) + 1
    s = (np.arange(n) - n // 2) * ds_mm
    return np.asarray(center, float)[None, :] + s[:, None] * d[None, :], s


def _orthonormal(n):
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u) + 1e-30
    w = np.cross(n, u)
    return u, w


def export_ti_case(lf, target, montages, out_path, axis=None,
                   length_mm=20.0, ds_mm=0.25, f1=2000.0, f2=2100.0,
                   off_point=None, verbose=True):
    """Build a NEURON validation case (.npz).

    montages : {"name": best_dict, ...} (e.g. {"field_opt": ..., "af_opt": ...})
    axis     : the principal axon direction; without it, the principal 3D-GEVD component is
               used. Axons are generated at the target along that axis and two orthogonal
               directions. Passing `off_point` also generates them there, for a selectivity
               comparison.
    Saves: coords (A,N,3), arclen (N), Ve1/Ve2 (A,N) in mV per montage, direction labels,
    f1 and f2.
    """
    import os
    from .benchmark import principal_direction
    cen = np.asarray(target.target_idx); C = lf.coords()
    tc = C[cen].mean(0)
    n = np.asarray(axis, float) if axis is not None else principal_direction(lf, target)
    n = n / (np.linalg.norm(n) + 1e-30)
    u, w = _orthonormal(n)
    sites = [("target", tc)]
    if off_point is not None:
        sites.append(("off", np.asarray(off_point, float)))
    dirs = [("axis", n), ("orthoU", u), ("orthoW", w)]

    trajs = []; labels = []; coords = []
    arclen = None
    for sname, sc in sites:
        for dname, dv in dirs:
            tr, s = straight_axon(sc, dv, length_mm, ds_mm)
            trajs.append(tr); labels.append(f"{sname}:{dname}"); coords.append(tr)
            arclen = s
    coords = np.stack(coords)                                          # (A, N, 3)

    out = dict(coords=coords, arclen=arclen, labels=np.array(labels),
               f1=float(f1), f2=float(f2), target_center=tc, axis=n)
    for mname, best in montages.items():
        Ve1 = np.zeros((len(trajs), coords.shape[1])); Ve2 = np.zeros_like(Ve1)
        for i, tr in enumerate(trajs):
            v1, v2 = axon_potentials(lf, best, tr); Ve1[i] = v1; Ve2[i] = v2
        out[f"{mname}__Ve1"] = Ve1; out[f"{mname}__Ve2"] = Ve2
        out[f"{mname}__montage"] = np.array(str(best.get("ch1", "")) + " x " + str(best.get("ch2", "")))
    np.savez(out_path, **out)
    if verbose:
        print(f"[neuron_bridge] saved {out_path}")
        print(f"  {len(trajs)} trajectories ({', '.join(labels)}) · {coords.shape[1]} segments each "
              f"· spacing {ds_mm} mm · length {length_mm} mm")
        for mname in montages:
            v = out[f"{mname}__Ve1"]; print(f"  {mname}: |Ve1| max {np.abs(v).max():.3f} mV · Ve2 max {np.abs(out[mname+'__Ve2']).max():.3f} mV")
    return out_path
