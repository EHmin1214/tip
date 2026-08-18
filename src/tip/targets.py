# -*- coding: utf-8 -*-
"""
targets.py — target / ROI definitions
========================================================
This is where the "what do you want to stimulate" half of the problem enters. It replaces the
hard-coded left hippocampus with arbitrary targets. Coordinate-based targets **must** go
through bmask decoding (`leadfield.coords`) — see the note in `leadfield.py`.

Target:
  target_idx : row indices of the target voxels
  off_idx    : row indices of the off-target voxels (target plus margin removed); may be
               subsampled
  off_def    : name of the off-pool definition ("gm_wm" by default).
               **Always report this alongside any metric.**
  direction  : optional axis n for fixed-direction stimulation (e.g. the hippocampal nL)

★The off-pool default changed from GM-only to **GM ∪ WM** on 2026-08-05.
  With GM only, white matter (51% of the brain) is absent from the off pool, so optimising a
  deep target could pour field into it without penalty. The winning montage actually changes
  for 3 of 5 targets.
  The reasoning and numbers are in the OFF_LABEL_SETS comment in config.py.
  To reproduce older numbers use `Target(..., off_labels="gm")`.
"""
import numpy as np
from . import config as C


class Target:
    def __init__(self, lf, target_idx, name="target", direction=None,
                 off_margin_mm=0.0, off_subsample=30000, seed=42, off_labels=None):
        """off_labels : None = config.OFF_DEFAULT, or "gm" / "gm_wm" / "brain", or a tuple of
        labels. Whichever was used is recorded in `self.off_def` — always report it with the
        metrics."""
        self.lf = lf
        self.name = name
        self.target_idx = np.asarray(target_idx, dtype=np.int64)
        self.direction = None if direction is None else np.asarray(direction, float)

        # off-target pool = the off_labels tissues, minus the target, minus the margin
        # (GM ∪ WM by default; see config.py)
        key = C.OFF_DEFAULT if off_labels is None else off_labels
        if isinstance(key, str):
            if key not in C.OFF_LABEL_SETS:
                raise ValueError(f"unsupported off_labels: {key!r} "
                                 f"(available: {list(C.OFF_LABEL_SETS)})")
            labels = C.OFF_LABEL_SETS[key]; self.off_def = key
        else:
            labels = tuple(key); self.off_def = str(labels)
        bl = np.load(_p(lf, C.BLABEL_FILE))
        neural = np.ones(len(bl), bool) if labels is None else np.isin(bl, labels)
        off_mask = neural.copy()
        off_mask[self.target_idx] = False
        if off_margin_mm > 0:
            tc = lf.coords(self.target_idx)
            allc = lf.coords()
            # exclude voxels inside the margin (approximated by the target bounding box
            # expanded by the margin)
            lo = tc.min(0) - off_margin_mm; hi = tc.max(0) + off_margin_mm
            near = np.all((allc >= lo) & (allc <= hi), axis=1)
            off_mask[near] = False
        off_all = np.where(off_mask)[0]

        rng = np.random.default_rng(seed)
        if off_subsample and len(off_all) > off_subsample:
            self.off_idx = np.sort(rng.choice(off_all, off_subsample, replace=False))
        else:
            self.off_idx = off_all
        self.off_full_n = len(off_all)

    # ---------- factories ----------
    @classmethod
    def from_mask(cls, lf, mask, **kw):
        """Define a target from a boolean mask (N,) or an array of row indices."""
        idx = np.where(mask)[0] if mask.dtype == bool else np.asarray(mask)
        return cls(lf, idx, **kw)

    @classmethod
    def from_label(cls, lf, label, **kw):
        """Define a target from a tissue label in `blabel` (e.g. hippocampus = 81)."""
        bl = np.load(_p(lf, C.BLABEL_FILE))
        return cls(lf, np.where(bl == label)[0], **kw)

    @classmethod
    def from_sphere(cls, lf, center, radius_mm, restrict_neural=True, **kw):
        """Spherical target at `center` = (x, y, z) in mm. Selection uses bmask-decoded
        coordinates only."""
        c = np.asarray(center, float)
        coords = lf.coords()
        d2 = ((coords - c) ** 2).sum(1)
        sel = d2 <= radius_mm ** 2
        if restrict_neural:
            bl = np.load(_p(lf, C.BLABEL_FILE))
            sel &= np.isin(bl, C.NEURAL_LABELS)
        return cls(lf, np.where(sel)[0], **kw)

    def summary(self):
        c = self.lf.coords(self.target_idx)
        return dict(name=self.name, n_target=len(self.target_idx),
                    center=c.mean(0).round(1).tolist(),
                    radius=float(np.sqrt(((c - c.mean(0)) ** 2).sum(1)).max().round(1)),
                    off_full=self.off_full_n, off_sub=len(self.off_idx),
                    off_def=self.off_def)


def _p(lf, fn):
    import os
    from . import config as C
    return C.inputs(fn)
