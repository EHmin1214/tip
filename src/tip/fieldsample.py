# -*- coding: utf-8 -*-
"""
fieldsample.py — trilinear interpolation on a graded rectilinear grid, plus windowed
smoothing along the axon
=====================================================================
The leadfield stores brain-voxel E on a **graded (non-uniform) rectilinear grid** (cx, cy, cz;
0.4-0.9 mm inside the brain). Snapping an off-grid point — an axon coordinate, say — **to the
nearest voxel** (what neuron_bridge, activating and app used to do) turns Ve(s) along the axon
into a **staircase**, and its derivative, the activating function, amplifies grid-scale noise.
NEURON RESULTS §6 flagged this: 43-94% of the AF was a component a 5th-order polynomial could
not capture, and diagnostics showed 4-6x more point-to-point roughness and roughly 2x inflated
AF magnitude.

The fix:
  · **Trilinear interpolation** — weight the eight corner brain voxels of the enclosing grid
    cell by the non-uniform spacing. The staircase disappears. (Corners outside the brain get
    weight 0 and the rest are renormalised; if every corner is invalid, fall back to nearest.)
  · **Smoothing at the physical scale** — compute AF = d(n·E)/ds not from a two-point finite
    difference but as the **least-squares slope** over `nsamp` interpolated samples in a window
    [-L, L], with L = AF_SMOOTH_MM (internodal scale, ~1 mm). A real axon low-pass filters the
    field over its internodal length, so this smoothing is physically the right thing to do.
"""
import numpy as np
from scipy.spatial import cKDTree

AF_SMOOTH_MM = 1.0        # AF window half-width (internodal scale, ~1 mm) = smoothing scale
AF_NSAMP = 5              # interpolated samples inside the window (for the LSQ slope)

_GRIDS = {}


class _Grid:
    """Trilinear interpolation weights for the leadfield's non-uniform rectilinear grid."""
    def __init__(self, lf):
        self.lf = lf
        self.cx, self.cy, self.cz = lf.cx, lf.cy, lf.cz
        b = lf.bmask
        self.gr = np.full((len(self.cx), len(self.cy), len(self.cz)), -1, np.int64)
        self.gr[b[:, 0], b[:, 1], b[:, 2]] = np.arange(b.shape[0])
        self._tree = None

    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.lf.coords())
        return self._tree

    def weights(self, pts):
        """pts (M,3) → (rows (M,8), w (M,8)): brain-voxel row indices and trilinear weights.
        Invalid corners (outside the brain) get weight 0 and the valid ones are renormalised.
        If every corner is invalid, fall back to the nearest voxel with weight 1."""
        pts = np.asarray(pts, float)

        def loc(a, x):
            i = np.clip(np.searchsorted(a, x) - 1, 0, len(a) - 2)
            t = np.clip((x - a[i]) / (a[i + 1] - a[i]), 0.0, 1.0)
            return i, t

        i0, tx = loc(self.cx, pts[:, 0]); j0, ty = loc(self.cy, pts[:, 1]); k0, tz = loc(self.cz, pts[:, 2])
        M = len(pts); rows = np.empty((M, 8), np.int64); w = np.empty((M, 8)); c = 0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    rows[:, c] = self.gr[i0 + di, j0 + dj, k0 + dk]
                    w[:, c] = (tx if di else 1 - tx) * (ty if dj else 1 - ty) * (tz if dk else 1 - tz)
                    c += 1
        valid = rows >= 0
        w = np.where(valid, w, 0.0)
        ssum = w.sum(1)
        bad = ssum <= 1e-12
        if bad.any():                                    # all corners invalid → nearest
            _, nn = self.tree().query(pts[bad])
            rows[bad] = nn[:, None]; w[bad] = 0.0; w[bad, 0] = 1.0
            valid[bad] = True; ssum = w.sum(1)
        rows = np.where(valid, rows, 0)                  # make invalid rows safe to index
                                                         # (weight 0, so they contribute nothing)
        return rows, w / ssum[:, None]


def grid(lf):
    k = id(lf)
    g = _GRIDS.get(k)
    if g is None:
        g = _GRIDS[k] = _Grid(lf)
    return g


def _unit(n):
    n = np.asarray(n, float)
    return n / (np.linalg.norm(n) + 1e-30)


def interp_apply(lf, field_at, pts):
    """The two carrier fields (E1, E2), shape (M,3), trilinearly interpolated at pts (M,3).
    `field_at(idx) -> (E1, E2)` supplies the montage's two fields at brain-voxel rows `idx`
    and works for any montage type."""
    rows, w = grid(lf).weights(pts)
    uniq, inv = np.unique(rows, return_inverse=True); inv = inv.reshape(rows.shape)  # (M,8)
    E1u, E2u = field_at(uniq); E1u = np.asarray(E1u); E2u = np.asarray(E2u)
    return (w[..., None] * E1u[inv]).sum(1), (w[..., None] * E2u[inv]).sum(1)


def af_env(lf, field_at, n, pts, smooth_mm=None, nsamp=AF_NSAMP):
    """AF envelope 2·min(|AF1|, |AF2|) at pts (M,3), where AF_k is the least-squares slope of
    the interpolated samples of n·E_k(s) inside the window [-L, L] — this removes the staircase
    and smooths at the physical scale. `field_at` is the same as in `interp_apply`."""
    if smooth_mm is None:
        smooth_mm = AF_SMOOTH_MM
    n = _unit(n); pts = np.asarray(pts, float)
    s = np.linspace(-smooth_mm, smooth_mm, nsamp); sc = s - s.mean(); denom = (sc ** 2).sum()
    allpts = (pts[None] + s[:, None, None] * n).reshape(-1, 3)      # (nsamp·M, 3)
    E1, E2 = interp_apply(lf, field_at, allpts); M = len(pts)
    P1 = E1.reshape(nsamp, M, 3) @ n; P2 = E2.reshape(nsamp, M, 3) @ n
    AF1 = (sc[:, None] * (P1 - P1.mean(0))).sum(0) / denom
    AF2 = (sc[:, None] * (P2 - P2.mean(0))).sum(0) / denom
    return 2.0 * np.minimum(np.abs(AF1), np.abs(AF2))


def af_proj_elec(lf, electrodes, n, idx, smooth_mm=None, nsamp=AF_NSAMP):
    """Per-electrode AF_e = d(n·E_e)/ds at voxels `idx`. Returns (K,N).
    Interpolation plus the windowed least-squares slope."""
    if smooth_mm is None:
        smooth_mm = AF_SMOOTH_MM
    n = _unit(n); pts = lf.coords()[np.asarray(idx)]
    s = np.linspace(-smooth_mm, smooth_mm, nsamp); sc = s - s.mean(); denom = (sc ** 2).sum()
    g = grid(lf); N = len(pts)
    rows = np.empty((nsamp, N, 8), np.int64); wts = np.empty((nsamp, N, 8))
    for a, sa in enumerate(s):
        rows[a], wts[a] = g.weights(pts + sa * n)
    uniq, inv = np.unique(rows, return_inverse=True); inv = inv.reshape(rows.shape)   # (nsamp,N,8)
    K = len(electrodes); AF = np.empty((K, N))
    for ei, e in enumerate(electrodes):
        Fe = lf.elec_field(e, uniq) @ n                          # (U,) projection
        Pe = (wts * Fe[inv]).sum(-1)                             # (nsamp,N)
        AF[ei] = (sc[:, None] * (Pe - Pe.mean(0))).sum(0) / denom
    return AF


# ============ GAF (generalised activating function, the MDF2 form) ============
# Peterson & Grill, "Predicting myelinated axon activation..." (PMC3197268): convolve the
# second difference at the nodes with the passive-cable response kernel, i.e.
# GAF(x) = sum_j W(|j|)·AF(x + jL). Against MRG this has 5.7% error, versus 40% for the
# classic AF.
# W is the passive-cable step response (~exp(-|j|/lambda), lambda = length constant in
# internodal units) — the concrete form of the "Green's function" idea.
# In short: spatially low-pass our AF (Rattay's d2Ve/ds2) with the cable kernel at the
# internodal spacing to obtain GAF.
GAF_L = 1.0          # internodal spacing (mm); ~1.15 for a 10 um MRG axon
GAF_LAMBDA = 3.0     # passive-cable length constant (internodal units); ~2-5 for myelinated MRG
GAF_K = 4            # kernel half-width (number of internodes)


def _gaf_kernel(lam=None, K=None):
    lam = GAF_LAMBDA if lam is None else lam; K = GAF_K if K is None else K
    nodes = np.arange(-K, K + 1)
    W = np.exp(-np.abs(nodes) / lam)
    return nodes, W / W.sum()


def gaf_env(lf, field_at, n, pts, L=None, lam=None, K=None, dl=None):
    """GAF envelope 2·min(|GAF1|, |GAF2|) at pts, with
    GAF_k(x) = sum_j W(|j|)·AF_k(x + j·L), AF = d(n·E)/ds at node j and W the passive-cable
    kernel. `field_at` is the same as in `interp_apply`.
    With K = 0 this reduces to `af_env` (a single node). Models the spatial filtering a
    myelinated axon performs."""
    L = GAF_L if L is None else L; dl = (0.5 * L) if dl is None else dl
    n = _unit(n); pts = np.asarray(pts, float)
    nodes, W = _gaf_kernel(lam, K)
    offs = np.concatenate([[k * L - dl, k * L + dl] for k in nodes])    # (2·(2K+1),)
    E1, E2 = interp_apply(lf, field_at, (pts[None] + offs[:, None, None] * n).reshape(-1, 3))
    M = len(pts); no = len(offs)
    P1 = E1.reshape(no, M, 3) @ n; P2 = E2.reshape(no, M, 3) @ n        # E∥ at offsets
    AF1 = (P1[1::2] - P1[0::2]) / (2 * dl); AF2 = (P2[1::2] - P2[0::2]) / (2 * dl)   # (2K+1,M) per-node AF
    GAF1 = (W[:, None] * AF1).sum(0); GAF2 = (W[:, None] * AF2).sum(0)
    return 2.0 * np.minimum(np.abs(GAF1), np.abs(GAF2))


def gaf_proj_elec(lf, electrodes, n, idx, L=None, lam=None, K=None, dl=None):
    """Per-electrode GAF_e at voxels `idx`. Returns (Kel,N): AF_e weighted by the cable kernel
    at the node spacing."""
    L = GAF_L if L is None else L; dl = (0.5 * L) if dl is None else dl
    n = _unit(n); pts = lf.coords()[np.asarray(idx)]
    nodes, W = _gaf_kernel(lam, K)
    offs = np.concatenate([[k * L - dl, k * L + dl] for k in nodes]); no = len(offs)
    g = grid(lf); N = len(pts)
    rows = np.empty((no, N, 8), np.int64); wts = np.empty((no, N, 8))
    for a, sa in enumerate(offs):
        rows[a], wts[a] = g.weights(pts + sa * n)
    uniq, inv = np.unique(rows, return_inverse=True); inv = inv.reshape(rows.shape)
    Kel = len(electrodes); GAF = np.empty((Kel, N))
    for ei, e in enumerate(electrodes):
        Fe = lf.elec_field(e, uniq) @ n; Pe = (wts * Fe[inv]).sum(-1)   # (no,N) E∥
        AFn = (Pe[1::2] - Pe[0::2]) / (2 * dl)                          # (2K+1,N) per-node AF
        GAF[ei] = (W[:, None] * AFn).sum(0)
    return GAF
