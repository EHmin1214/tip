# -*- coding: utf-8 -*-
"""
fiberlead.py — fibre leadfield (per-electrode extracellular potential Ve, precomputed)
=====================================================================================
Just as the E-field leadfield solves "E per electrode" once and gets any montage by
superposition, precomputing **Ve per electrode along a trajectory** gives the Ve of any
montage instantly.

Why this is exact — the whole Ve path is **linear** in current:
    trilinear interpolation -> E·dl dot product -> cumulative sum -> zero mean -> scale
    Every step is linear, so  Ve(sum I_e·E_e) = sum I_e·Ve(E_e)  holds **exactly**.
-> we can bolt a physiology layer (AF/GAF) onto an exhaustive search over tens of thousands
   of montages without giving up the search.

The reference implementation is `neuron_bridge.axon_potentials()`. This module decomposes it
per electrode; equivalence is checked by the superposition test in §8-3 Step 3
(relative error < 1e-5) -> `verify_superposition()`.

Three drives (SETUP.md §5-3, §7):
    ve    |Ve|                     the extracellular potential itself
    field E_par = -dVe/ds          axial field — targets dominated by **terminations**
    af    dE_par/ds = -d2Ve/ds2    Rattay activating function — **fibre-of-passage** targets
    gaf   sum_j W(|j|)·af(x+j·L)   cable-kernel weighting (Peterson & Grill MDF2), same
                                   constants as `fieldsample`
The envelope is always **element-wise** 2·min(|D1|, |D2|); reduction over a fibre (max) comes
**after** the envelope. Reducing first would let the two carriers peak at different places,
which robs the min of its physical meaning.

Units: coordinates mm, E V/m, Ve mV (same as `neuron_bridge`).
"""
import os
import json
import numpy as np

from .fieldsample import grid, _gaf_kernel, GAF_L, GAF_LAMBDA, GAF_K


# ────────────────────────────── construction ──────────────────────────────
def _arclen(trajs):
    """Cumulative arc length s per trajectory, shape (F,N), in mm."""
    d = np.zeros(trajs.shape[:2])
    d[:, 1:] = np.linalg.norm(np.diff(trajs, axis=1), axis=2)
    return np.cumsum(d, axis=1)


def _grad(V, s):
    """dV/ds for V of shape (...,F,N) and s of shape (F,N). Second-order central differences
    on a non-uniform spacing, first-order at the ends.

    Gives the same values as `np.gradient(..., edge_order=1)` but vectorised over all the
    leading axes."""
    hs = s[..., 1:-1] - s[..., :-2]
    hd = s[..., 2:] - s[..., 1:-1]
    out = np.empty(np.broadcast_shapes(V.shape, s.shape), float)
    out[..., 1:-1] = (hs ** 2 * V[..., 2:] + (hd ** 2 - hs ** 2) * V[..., 1:-1]
                      - hd ** 2 * V[..., :-2]) / (hs * hd * (hd + hs))
    out[..., 0] = (V[..., 1] - V[..., 0]) / (s[..., 1] - s[..., 0])
    out[..., -1] = (V[..., -1] - V[..., -2]) / (s[..., -1] - s[..., -2])
    return out


def build_fiber_leadfield(lf, trajs, out_path=None, electrodes=None,
                          dtype=np.float64, verbose=True):
    """Build a fibre leadfield: Ve **per electrode** along the trajectories `trajs` (F,N,3).

    Returns/saves: Ve (E,F,N) in mV, names (E,), trajs (F,N,3), arclen (F,N).
    Same conventions as `axon_potentials`: right-endpoint Riemann sum with dl[0] = 0,
    zero mean per trajectory, millivolts.

    dtype: the storage precision for Ve. **float64 by default** — a montage uses the
    **difference** between an electrode pair (Ve,a - Ve,b), and when the two terms are close
    in magnitude the cancellation amplifies the storage precision directly. Stored as
    float32, the superposition check drifts to ~1e-5: harmless for optimisation ranking, but
    it fails the §8-3 Step 3 gate. The cost is E·F·N·8 B — even 70 electrodes × 1000 fibres ×
    200 nodes is 112 MB, so float64 is usually fine. Pass float32 only if memory is tight.
    """
    trajs = np.asarray(trajs, float)
    if trajs.ndim != 3 or trajs.shape[2] != 3:
        raise ValueError(f"trajs must be (F,N,3) - got shape {trajs.shape}")
    F, N = trajs.shape[:2]
    names = list(electrodes) if electrodes is not None else [e for e in lf.names if lf.has(e)]

    # Interpolation weights do not depend on the electrode, so compute them once and reuse
    # for every electrode (same pattern as `af_proj_elec`)
    pts = trajs.reshape(-1, 3)
    rows, w = grid(lf).weights(pts)                       # (F·N, 8)
    uniq, inv = np.unique(rows, return_inverse=True)
    inv = inv.reshape(rows.shape)

    dl = np.zeros_like(trajs)
    dl[:, 1:] = np.diff(trajs, axis=1)                    # (F,N,3) mm, dl[:,0]=0

    Ve = np.empty((len(names), F, N), dtype)
    for i, e in enumerate(names):
        Ee = lf.elec_field(e, uniq)                       # (U,3) V/m @1mA
        Efib = (w[..., None] * Ee[inv]).sum(1).reshape(F, N, 3)   # trilinear interpolation
        edl = (Efib * dl).sum(-1) * 1e-3                  # (F,N) V  (E[V/m]·dl[mm])
        v = -np.cumsum(edl, axis=1)                       # Vₑ = −∫E·dl
        Ve[i] = ((v - v.mean(axis=1, keepdims=True)) * 1e3).astype(dtype)   # mV, zero mean

    # trajs and arclen **must stay float64** — they define the coordinates Ve was sampled at.
    # Dropping to float32 (~3e-5 mm error) shifts the interpolation points on recomputation
    # and breaks the superposition check at the 1e-5 level. At F·N·3·8 B the cost is
    # negligible next to Ve.
    out = dict(Ve=Ve, names=np.array(names), trajs=trajs, arclen=_arclen(trajs))
    if out_path:
        np.savez_compressed(out_path, **out)
        if verbose:
            mb = os.path.getsize(out_path) / 1e6
            print(f"[fiberlead] saved {out_path}")
            print(f"  electrodes {len(names)} · fibres {F} · nodes {N} · "
                  f"{np.dtype(dtype).name} · {mb:.1f} MB")
    return out


def label_fibers(lf, trajs, target_idx, tol_mm=1.0):
    """Does each trajectory pass through the target voxel set? Returns bool (F,).
    `tol_mm` is the allowed node-to-target nearest distance."""
    from scipy.spatial import cKDTree
    trajs = np.asarray(trajs, float)
    tgt = lf.coords()[np.asarray(target_idx)]
    tree = cKDTree(tgt)
    d, _ = tree.query(trajs.reshape(-1, 3))
    return (d.reshape(trajs.shape[:2]) <= tol_mm).any(axis=1)


def sample_seeds(lf, n_seeds, exclude_idx=None, margin_mm=12.0, labels=None,
                 edge_mm=12.0, seed=11):
    """Sample seed positions and directions for an off-target fibre population from brain
    tissue. Returns (centers (K,3), dirs (K,3)).

    labels : tissue labels. Default **GM ∪ WM** — fibres are a white-matter structure, but
             cortical neurons are also being evaluated, so both go in. (`config.NEURAL_LABELS`
             is GM-only and meant for voxel metrics; do not use it here. Comparing a voxel
             off-pool against a fibre off-pool built on a different basis is invalid —
             MIGRATION_STATUS §4-1.)
    dirs   : **isotropic random**. Without DTI the real fibre orientations are unknown, so
             instead of committing to a direction we build an off population that is
             direction-agnostic, i.e. includes the worst case. Same stance as the
             "worst direction" convention in `optimize/selective.py`.

    Seeds are excluded within `margin_mm` of `exclude_idx` and within `edge_mm` of the brain
    boundary — a bundle running off the grid would drop to nearest-neighbour interpolation.
    """
    from scipy.spatial import cKDTree
    from . import config as C
    labels = (C.LABEL_GM, C.LABEL_WM) if labels is None else tuple(labels)
    lab = np.load(C.inputs("blabel1010.npy"))
    pool = np.where(np.isin(lab, labels))[0]
    P = lf.coords(pool)

    lo, hi = lf.coords().min(0), lf.coords().max(0)          # brain-boundary margin
    keep = ((P >= lo + edge_mm) & (P <= hi - edge_mm)).all(1)
    pool, P = pool[keep], P[keep]

    if exclude_idx is not None and len(exclude_idx):
        d, _ = cKDTree(lf.coords()[np.asarray(exclude_idx)]).query(P)
        pool, P = pool[d > margin_mm], P[d > margin_mm]
    if len(pool) < n_seeds:
        raise ValueError(f"only {len(pool)} seed candidates - reduce margin_mm / edge_mm")

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pool), n_seeds, replace=False)
    v = rng.normal(size=(n_seeds, 3))
    return P[sel], v / np.linalg.norm(v, axis=1, keepdims=True)


# ──────────────────────────── use ────────────────────────────
class FiberLeadField:
    """Evaluate montages instantly on top of a precomputed fibre leadfield."""

    def __init__(self, path_or_dict, target_mask=None):
        d = np.load(path_or_dict) if isinstance(path_or_dict, str) else path_or_dict
        self.Ve = np.asarray(d["Ve"])                          # (E,F,N) mV, keeping the stored dtype
        self.names = [str(x) for x in d["names"]]
        self.trajs = np.asarray(d["trajs"], float)
        self.arclen = np.asarray(d["arclen"], float)           # (F,N) mm
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.n_elec, self.n_fibers, self.n_nodes = self.Ve.shape
        self.target = (np.zeros(self.n_fibers, bool) if target_mask is None
                       else np.asarray(target_mask, bool))
        self._cache = {}                                       # per-electrode drives and the GAF interpolation operator

    # ---- carriers ----
    def _elec(self, name):
        if name not in self.idx:
            raise KeyError(f"electrode {name!r} is not in the fibre leadfield; "
                           f"it holds {len(self.names)} (e.g. {self.names[:5]})")
        return self.Ve[self.idx[name]]

    def carriers(self, best):
        """Montage → the two carrier extracellular potentials (V1, V2), each (F,N) in mV,
        by local **linear superposition**."""
        if "currents" in best:                                  # distributed form
            c = best["currents"]
            V1 = sum(self._elec(e) * I for e, I in c["ch0"].items())
            V2 = sum(self._elec(e) * I for e, I in c["ch1"].items())
            return np.asarray(V1, float), np.asarray(V2, float)
        a, b = best["ch1"]; cc, d = best["ch2"]
        from .optimize.classic import channel_currents
        i1, i2 = channel_currents(best.get("ratio", 1.0))
        return (i1 * (self._elec(a) - self._elec(b)).astype(float),
                i2 * (self._elec(cc) - self._elec(d)).astype(float))

    # ---- drives ----
    # Every drive (d/ds, GAF kernel sum) is a **linear** operation on Ve. So computing the
    # drive once per electrode — a drive leadfield — lets any montage be recovered by
    # superposition again.
    # Re-differentiating for every montage would make exhaustive search impossible
    # (~34 ms per montage for gaf, versus ~0.1 ms this way).
    def drive(self, V, kind="gaf", L=None, lam=None, K=None):
        """Ve (...,F,N) → drive (...,F,N). `kind` is one of ve|field|af|gaf."""
        if kind == "ve":
            return V
        Epar = -_grad(V, self.arclen)                           # E∥ = −∂Vₑ/∂s
        if kind == "field":
            return Epar
        AF = _grad(Epar, self.arclen)                     # dE_par/ds (Rattay activating fn)
        if kind == "af":
            return AF
        if kind != "gaf":
            raise ValueError(f"kind must be one of ve|field|af|gaf - got {kind!r}")
        out = np.zeros_like(AF)
        for wj, lo, t in self._gaf_ops(GAF_L if L is None else L, lam, K):
            lo_b = np.broadcast_to(lo, AF.shape); t_b = np.broadcast_to(t, AF.shape)
            a = np.take_along_axis(AF, lo_b, -1)
            b = np.take_along_axis(AF, np.minimum(lo_b + 1, AF.shape[-1] - 1), -1)
            out += wj * ((1.0 - t_b) * a + t_b * b)
        return out

    def _gaf_ops(self, L, lam, K):
        """Linear-interpolation operator per node offset (weights, lower index, fraction).
        Depends only on `arclen`, so it is cached."""
        key = ("ops", L, lam, K)
        if key not in self._cache:
            nodes, W = _gaf_kernel(lam, K)
            ops = []
            for j, wj in zip(nodes, W):
                lo = np.empty(self.arclen.shape, np.intp)
                t = np.empty(self.arclen.shape, float)
                for i in range(self.n_fibers):                  # once per fibre, montage-independent
                    s = self.arclen[i]
                    q = np.clip(s + j * L, s[0], s[-1])
                    ix = np.clip(np.searchsorted(s, q) - 1, 0, len(s) - 2)
                    lo[i] = ix
                    t[i] = (q - s[ix]) / np.maximum(s[ix + 1] - s[ix], 1e-30)
                ops.append((wj, lo, t))
            self._cache[key] = ops
        return self._cache[key]

    def elec_drives(self, kind="gaf", **kw):
        """Per-electrode drive (E,F,N), computed once and cached — the linear basis every
        montage evaluation is built from."""
        key = (kind, kw.get("L"), kw.get("lam"), kw.get("K"))
        if key not in self._cache:
            self._cache[key] = self.drive(np.asarray(self.Ve, float), kind, **kw)
        return self._cache[key]

    def carrier_drives(self, best, kind="gaf", **kw):
        """Montage → the two carrier drives (D1, D2), each (F,N), by superposing the
        per-electrode drives."""
        D = self.elec_drives(kind, **kw)
        if "currents" in best:
            c = best["currents"]
            D1 = sum(D[self.idx[e]] * I for e, I in c["ch0"].items())
            D2 = sum(D[self.idx[e]] * I for e, I in c["ch1"].items())
            return np.asarray(D1, float), np.asarray(D2, float)
        a, b = best["ch1"]; cc, d = best["ch2"]
        from .optimize.classic import channel_currents
        i1, i2 = channel_currents(best.get("ratio", 1.0))
        for nm in (a, b, cc, d):
            if nm not in self.idx:
                raise KeyError(f"electrode {nm!r} is not in the fibre leadfield")
        return (i1 * (D[self.idx[a]] - D[self.idx[b]]),
                i2 * (D[self.idx[cc]] - D[self.idx[d]]))

    def envelope(self, best, kind="gaf", **kw):
        """TI envelope 2·min(|D1|, |D2|), shape (F,N). **Element-wise** — reduction over a
        fibre happens after this, never before."""
        D1, D2 = self.carrier_drives(best, kind, **kw)
        return 2.0 * np.minimum(np.abs(D1), np.abs(D2))

    def fiber_drive(self, best, kind="gaf", **kw):
        """One scalar per fibre: the maximum of the envelope along the trajectory (F,).
        Firing starts at that maximum."""
        return self.envelope(best, kind, **kw).max(axis=1)

    def metrics(self, best, kind="gaf", pctl=50, **kw):
        """M1 strength, M2 focality, M3 leakage at the fibre-population level
        (definitions in SETUP.md §5-4)."""
        fd = self.fiber_drive(best, kind, **kw)
        t, o = self.target, ~self.target
        if not t.any():
            raise ValueError("no target fibres - check centre, direction and tol_mm in label_fibers")
        et, eo = fd[t], fd[o]
        M1 = float(np.median(et))
        if not o.any():
            return {"M1": M1, "M2": float("inf"), "M3": 0.0, "kind": kind,
                    "n_target": int(t.sum()), "n_off": 0}
        rms_t = np.sqrt((et ** 2).mean()); rms_o = np.sqrt((eo ** 2).mean())
        return {"M1": M1,
                "M2": float((rms_t / max(rms_o, 1e-12)) ** 2),
                "M3": float(100.0 * (eo > np.percentile(et, pctl)).mean()),
                "kind": kind, "n_target": int(t.sum()), "n_off": int(o.sum())}

    # ---- export ----
    def export_candidates(self, montages, out_path, f1=2000.0, f2=2100.0, verbose=True):
        """Export Ve for the top candidates so a neuron simulator can evaluate them
        (pipeline stage 4).

        montages : {"name": best_dict, ...}

        Written with **the same key convention as `neuron_bridge.export_ti_case()`**, because
        the existing standalone NEURON harness has to read this file unmodified. (The NEURON
        validation in §7 went through that path. Sim4Life T-Neuro does the same job faster but
        is not an irreplaceable route.)
            coords (F,N,3) [alias of trajs] · arclen (F,N) · labels (F,) · f1 · f2 · target (F,)
            {name}__Ve1 / __Ve2, each (F,N) in mV · {name}__montage
        """
        lab = np.array([f"fiber{i:04d}:{'target' if t else 'off'}"
                        for i, t in enumerate(self.target)])
        tc = (self.trajs[self.target].reshape(-1, 3).mean(0) if self.target.any()
              else self.trajs.reshape(-1, 3).mean(0))
        ax = self.trajs[:, -1] - self.trajs[:, 0]           # mean end-to-end direction per fibre
        ax = ax.mean(0); ax = ax / (np.linalg.norm(ax) + 1e-30)
        out = dict(coords=self.trajs, trajs=self.trajs,   # coordinates lossless (same reason as in build)
                   arclen=self.arclen, labels=lab,
                   f1=float(f1), f2=float(f2), target=self.target,
                   target_center=tc, axis=ax)             # same keys as export_ti_case
        for nm, best in montages.items():
            V1, V2 = self.carriers(best)
            out[f"{nm}__Ve1"] = V1.astype(np.float32)      # float32 is enough for NEURON input
            out[f"{nm}__Ve2"] = V2.astype(np.float32)
            # same string format as the existing harness; the full information goes in a
            # separate key
            out[f"{nm}__montage"] = np.array(
                str(best.get("ch1", "")) + " x " + str(best.get("ch2", "")))
            out[f"{nm}__montage_json"] = np.array(json.dumps(
                {k: v for k, v in best.items() if isinstance(v, (str, float, int, list, tuple))},
                ensure_ascii=False))
        np.savez_compressed(out_path, **out)
        if verbose:
            print(f"[fiberlead] exported {len(montages)} candidates → {out_path} "
                  f"(fibres {self.n_fibers} · nodes {self.n_nodes} · "
                  f"target {int(self.target.sum())})")
        return out_path


# ──────────────────── verification (SETUP.md §8-3 Step 3) ────────────────────
def verify_superposition(lf, fl, best, fibers=(0,), tol=1e-5, verbose=True):
    """**Required check** — does linear superposition agree with the direct computation
    (`neuron_bridge.axon_potentials`)?

    If this premise fails, the whole fibre-leadfield construction is invalid. Passes when the
    relative error is below `tol`.
    """
    from .neuron_bridge import axon_potentials
    V1, V2 = fl.carriers(best)
    worst = 0.0
    for i in fibers:
        v1d, v2d = axon_potentials(lf, best, fl.trajs[i])
        for got, ref in ((V1[i], v1d), (V2[i], v2d)):
            rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
            worst = max(worst, float(rel))
    ok = worst < tol
    if verbose:
        print(f"[fiberlead] superposition check, relative error {worst:.2e} "
              f"({'pass' if ok else 'FAIL'} · tol {tol:.0e}) · fibres {list(fibers)}")
    return ok, worst
