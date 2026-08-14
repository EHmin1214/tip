# -*- coding: utf-8 -*-
"""
activating.py — the activating-function objective
=====================================================
What drives a neuron is not the field *magnitude* but the axial gradient of the field along
the axon direction n:
    AF(x) = d(n·E)/ds   (Rattay's activating function, the drive term for a long straight fibre)
The TI drive for a long fibre is the AF envelope 2·min(|AF1|, |AF2|). Terminations and bends
are driven by the field projection n·E instead.

The grid is irregular, so a uniform-grid derivative is not available. We use **trilinear
interpolation plus a windowed least-squares slope** (`fieldsample.py`):
AF_e(x) is the least-squares slope of the interpolated samples of n·E_e(x + s·n) over the
window [-L, L]. That removes the nearest-neighbour staircase artefact (NEURON RESULTS §6).
AF_e is still a per-electrode scalar, so the direction-optimisation machinery is reused as-is.

L, the smoothing half-width, is the internodal length scale of a real axon (~1 mm when
myelinated); the default is `fieldsample.AF_SMOOTH_MM`.
(Previously this used a KDTree nearest-neighbour finite difference with h = 2.0, whose
grid-snap jitter inflated AF roughly 2x and made the optimiser fit noise.)
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from .optimize.classic import _ratio_grid, channel_currents

_TREE = {}


def _tree(lf):
    k = id(lf)
    if k not in _TREE:
        _TREE[k] = cKDTree(lf.coords())
    return _TREE[k]


def af_proj(lf, electrodes, n, idx, h=None):
    """Activating function AF_e = d(n·E_e)/ds at voxels `idx`. Returns (K, N).
    **Trilinear interpolation plus a least-squares slope over the window [-h, h]** — removes
    the staircase artefact and smooths at the internodal scale.
    `h` is the smoothing half-width (~1 mm internodal); None uses
    `fieldsample.AF_SMOOTH_MM`. See `fieldsample` §6."""
    from .fieldsample import af_proj_elec
    return af_proj_elec(lf, electrodes, n, idx, smooth_mm=h)


def field_proj(lf, electrodes, n, idx):
    """Directional field projection n·E_e at voxels `idx`. Returns (K, N)."""
    n = np.asarray(n, float); n = n / (np.linalg.norm(n) + 1e-30)
    return np.stack([lf.elec_field(e, idx) @ n for e in electrodes])


def dir_envelope(P, meta, r):
    """Directional envelope 2·min(|i1·d1|, |i2·d2|) of a two-pair montage, from the projection
    scalars P (K,N)."""
    (a, b), (c, d) = meta
    i1, i2 = channel_currents(r)
    return 2.0 * np.minimum(np.abs(i1 * (P[a] - P[b])), np.abs(i2 * (P[c] - P[d])))


def optimize_projected(Pt, Po, names, weights=(0.5, 0.5, 0.5), pctl=50, ratio_n=7):
    """Exhaustive search for the best two-pair montage from the projection scalars
    (Pt = (K,Nt), Po = (K,No)).

    It does not matter whether the projection is the field (n·E) or the AF — either way this
    maximises the WP of the directional envelope 2·min(|ch1|, |ch2|).
    Returns a classic-style dict(ch1, ch2, ratio)."""
    K = Pt.shape[0]; ratios = _ratio_grid(ratio_n)
    metas = []
    for a, b, c, d in itertools.combinations(range(K), 4):
        metas += [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
    p1a = np.fromiter((m[0][0] for m in metas), int); p1b = np.fromiter((m[0][1] for m in metas), int)
    p2a = np.fromiter((m[1][0] for m in metas), int); p2b = np.fromiter((m[1][1] for m in metas), int)
    C1t = Pt[p1a] - Pt[p1b]; C2t = Pt[p2a] - Pt[p2b]      # (M, Nt)
    C1o = Po[p1a] - Po[p1b]; C2o = Po[p2a] - Po[p2b]
    M = len(metas); aM1 = []; aM2 = []; aM3 = []; aR = []
    for r in ratios:
        i1, i2 = channel_currents(r)
        et = 2.0 * np.minimum(np.abs(i1 * C1t), np.abs(i2 * C2t))   # (M, Nt)
        eo = 2.0 * np.minimum(np.abs(i1 * C1o), np.abs(i2 * C2o))
        aM1.append(np.median(et, 1))
        rt = np.sqrt((et ** 2).mean(1)); ro = np.sqrt((eo ** 2).mean(1))
        aM2.append((rt / np.maximum(ro, 1e-12)) ** 2)
        thr = np.percentile(et, pctl, axis=1); aM3.append(100.0 * (eo > thr[:, None]).mean(1))
        aR.append(np.full(M, r))
    M1 = np.concatenate(aM1); M2 = np.concatenate(aM2); M3 = np.concatenate(aM3); RR = np.concatenate(aR)
    mx1 = M1.max() or 1.0; mx2 = M2.max() or 1.0; mx3 = M3.max() or 1.0
    WP = weights[0] * M1 / mx1 + weights[1] * M2 / mx2 - weights[2] * M3 / mx3
    bi = int(np.argmax(WP)); mi = bi % M; r = float(RR[bi])
    (a, b), (c, d) = metas[mi]
    return dict(ch1=(names[a], names[b]), ch2=(names[c], names[d]), ratio=r)
