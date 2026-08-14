# -*- coding: utf-8 -*-
"""
geometry.py — cortical surface normals (radial orientation) = pyramidal cell direction
==============================================================
Pyramidal cells in a cortical target align perpendicular to the surface (radially). MIDA does
not ship DTI orientations, but the outward surface normal can be computed geometrically from
**the gradient of the brain occupancy field** — which gives the cortical radial orientation
without any DTI.
Radial is meaningless for deep targets, so those use an anatomical axis or GEVD instead
(`is_cortical` decides automatically).
Validation: outwardness 0.76 at M1_R, i.e. mostly radial. In 3D, field / AF / GAF remain
spatially distinguishable (they yield different montages).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from .fieldsample import grid

_NRM = {}


def surface_normals(lf, sigma=2.5):
    """Outward brain-surface normals (N,3): the negative gradient of a Gaussian-smoothed brain
    occupancy field. On the non-uniform grid the median spacing per axis is used as an
    approximation."""
    k = id(lf)
    if k in _NRM:
        return _NRM[k]
    g = grid(lf); occ = (g.gr >= 0).astype(np.float32)
    occs = gaussian_filter(occ, sigma=sigma)
    sp = np.array([np.median(np.diff(lf.cx)), np.median(np.diff(lf.cy)), np.median(np.diff(lf.cz))])
    G = np.gradient(occs); b = lf.bmask
    nrm = np.stack([-(G[j][b[:, 0], b[:, 1], b[:, 2]]) / sp[j] for j in range(3)], axis=1)
    _NRM[k] = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    return _NRM[k]


def _idx(target):
    return np.asarray(target.target_idx if hasattr(target, "target_idx") else target)


def radial_direction(lf, target):
    """Mean radial orientation (surface normal) of a cortical target."""
    n = surface_normals(lf)[_idx(target)].mean(0)
    return n / (np.linalg.norm(n) + 1e-9)


def outwardness(lf, target):
    """How well the mean normal aligns with the brain-centre-to-target direction, i.e. how
    radial it is (+1 = fully radial, typical of cortex). Deep targets score low."""
    idx = _idx(target); C = lf.coords(); bc = C.mean(0)
    r = C[idx] - bc; r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    return float((surface_normals(lf)[idx] * r).sum(1).mean())


def is_cortical(lf, target, thresh=0.45):
    """Is the target cortical (superficial and radial)? Decides whether to use the radial
    orientation."""
    return outwardness(lf, target) > thresh
