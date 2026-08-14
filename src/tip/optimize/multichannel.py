# -*- coding: utf-8 -*-
"""
multichannel.py — multi-channel (distributed) TI current optimisation
=================================================================
Given a fixed electrode set, optimise the two channel currents per electrode — the inner half
of a two-level decomposition.
Maximise the modulation envelope along a direction n (e.g. the hippocampal axis nL), subject to
the off-target envelope staying below Ecap.

Method (a generalisation of the validated legacy `seq_opt`):
  E1 = sum c0_i·Mn_i,  E2 = sum c1_i·Mn_i,  with sum(c) = 0 per channel and |c_i| <= Imax
  directional envelope = 2·min(|n·E1|, |n·E2|)
  The min and abs make this non-convex, so we freeze the signs and channel assignment at the
  current solution and iterate an LP (active-set on the off-target constraints).

The "two frequency groups" of multi-channel TIP are represented by c0 (f1) and c1 (f2). An
electrode may appear in both groups.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, minimize
from scipy.linalg import eigh
from .. import ti, metrics
from ..config import ITOTAL


# ===================== GEVD focality (distributed TI) =====================
def _proj(lf, electrodes, idx, n):
    """Projection of the electrode fields onto direction n, shape (N, K)."""
    return np.stack([lf.elec_field(e, idx) @ n for e in electrodes], axis=1)


def focality_bound(at, ao):
    """Focality upper bound lambda_max for a single directional field, plus the optimal current
    distribution c*.
       max c^T A_t c / c^T A_o c, with A_t the target covariance and A_o the off covariance."""
    K = at.shape[1]
    At = at.T @ at / at.shape[0]
    Ao = ao.T @ ao / ao.shape[0] + 1e-9 * np.eye(K)
    w, V = eigh(At, Ao)
    return float(w[-1]), V[:, -1]


def optimize_gevd(lf, target, electrodes, direction, Imax=2.0, off_subsample=8000,
                  single_freq=False, select_k=None, seeds=6, iters=20, seed=42, verbose=True):
    """
    Distributed TI: maximise the focality M2 of the modulation envelope along n (a
    generalisation of the legacy `ti_gevd`).
    The single-field GEVD bound is then realised with two TI channels under a single-frequency
    constraint (one channel per electrode).
    return dict(focality_bound, M2_dir, ..., currents)
    """
    K = len(electrodes)
    n = np.asarray(direction, float); n = n / (np.linalg.norm(n) + 1e-30)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng0 = np.random.default_rng(seed)
    if off_subsample and len(off) > off_subsample:
        off = np.sort(rng0.choice(off, off_subsample, replace=False))
    at = _proj(lf, electrodes, tgt, n)   # (Nt, K)
    ao = _proj(lf, electrodes, off, n)   # (No, K)
    lam, cstar = focality_bound(at, ao)

    def m2_ti(c0, c1):
        Tt = np.minimum(np.abs(at @ c0), np.abs(at @ c1))
        To = np.minimum(np.abs(ao @ c0), np.abs(ao @ c1))
        return float((Tt ** 2).mean() / ((To ** 2).mean() + 1e-12))

    # Default: realise the focality bound with c0 = c1 = c* (M2 = lambda_max). Note that this
    # puts both frequencies on every electrode, which is a safety consideration.
    c0, c1, m = cstar.copy(), cstar.copy(), lam
    if single_freq:   # (experimental) iterated GEVD with one channel per electrode; falls back
                      # to the bound if it collapses
        best = None
        for sd in range(seeds):
            grp = (cstar >= 0).astype(int) if sd == 0 else rng0.integers(0, 2, K)
            if grp.sum() == 0 or grp.sum() == K:
                continue
            g0 = np.where(grp == 0, cstar, 0.0); g1 = np.where(grp == 1, cstar, 0.0)
            for _ in range(iters):
                st = (np.abs(at @ g0) <= np.abs(at @ g1)).astype(int)
                so = (np.abs(ao @ g0) <= np.abs(ao @ g1)).astype(int)
                def build(a, sig):
                    X = np.zeros((a.shape[0], 2 * K))
                    X[:, :K] = a * (sig == 0)[:, None]; X[:, K:] = a * (sig == 1)[:, None]
                    return X
                Xt = build(at, st); Xo = build(ao, so)
                At = Xt.T @ Xt / Xt.shape[0]; Ao = Xo.T @ Xo / Xo.shape[0] + 1e-9 * np.eye(2 * K)
                keep = np.ones(2 * K, bool)
                keep[np.where(grp == 1)[0]] = False
                keep[K + np.where(grp == 0)[0]] = False
                idx = np.where(keep)[0]
                w, V = eigh(At[np.ix_(idx, idx)], Ao[np.ix_(idx, idx)])
                full = np.zeros(2 * K); full[idx] = V[:, -1]
                g0, g1 = full[:K], full[K:]
                mm = m2_ti(g0, g1)
                if best is None or mm > best[0]:
                    best = (mm, g0.copy(), g1.copy())
        if best is not None and best[0] > 1e-6 and np.abs(best[2]).max() > 1e-9:
            m, c0, c1 = best   # a valid single-frequency solution
        elif verbose:
            print("  (single-frequency iteration collapsed - using the bound c0 = c1 = c*)", flush=True)
    # K-selection: keep the top-K electrodes by importance and re-run (outer selection with an
    # inner GEVD)
    if select_k and select_k < K:
        imp = np.abs(c0) + np.abs(c1)
        keep = np.sort(np.argsort(-imp)[:select_k])
        return optimize_gevd(lf, target, [electrodes[i] for i in keep], direction, Imax=Imax,
                             off_subsample=off_subsample, single_freq=single_freq,
                             seeds=seeds, iters=iters, seed=seed, verbose=verbose)
    # Normalise to physical currents. Focality is scale-invariant; this is for the absolute M1.
    sc = Imax / max(np.abs(c0).max(), np.abs(c1).max(), 1e-9)
    c0 *= sc; c1 *= sc
    Ft3 = np.stack([lf.elec_field(e, tgt) for e in electrodes])
    Fo3 = np.stack([lf.elec_field(e, off) for e in electrodes])
    if verbose:
        print(f"  focality bound lambda_max = {lam:.3f} | realised TI M2(n) = {m:.3f}", flush=True)
    out = _finalize(electrodes, c0, c1, Ft3, Fo3, n, verbose)
    out["focality_bound"] = lam
    return out
# ===========================================================================


def _gevd3d(Ft3, Fo3):
    """Isotropic 3D energy focality GEVD: max c^T Mt c / c^T Mo c, with Mt the target 3D field
    covariance.
    Returns (lambda_max, c*, n), where n is the normalised mean target field direction of c*."""
    K = Ft3.shape[0]
    Ftm = Ft3.reshape(K, -1); Fom = Fo3.reshape(K, -1)   # (K, N*3)
    Mt = Ftm @ Ftm.T / Ft3.shape[1]
    Mo = Fom @ Fom.T / Fo3.shape[1] + 1e-9 * np.eye(K)
    w, V = eigh(Mt, Mo)
    cstar = V[:, -1]
    Et = np.einsum("k,knd->nd", cstar, Ft3)             # target field
    n = Et.mean(0); nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-12 else np.array([0, 0, 1.0])
    return float(w[-1]), cstar, n


def optimize_distributed(lf, target, electrodes, direction=None, select_k=None,
                         pmax_rel=0.1, seed=42, verbose=False, **_legacy):
    """Distributed (directional array) optimisation — **routed through the Huang, Datta & Parra
    2020 backend** since 2026-07-28.

    Why: the previous GEVD/WP objective with an **isotropic off-target term** collapsed the
    modulation whenever focality was pushed — it could not balance strength against focality,
    and at iso-strength it produced no focal solution for the hippocampus. Huang's formulation
    (maximise directional target modulation **subject to off-target modulation power <= Pmax**)
    is structurally better at iso-strength, so it now fills the distributed role. Lower
    `pmax_rel` means more focal and weaker.

    The return shape is unchanged (distributed format, currents = ch0/ch1). The legacy GEVD
    implementation is preserved as `_distributed_legacy`.
    (Old arguments such as weights/pctl/Imax/restarts are ignored — Huang is a Pmax-constrained
    formulation.)"""
    from .huang import optimize_huang
    n = direction
    if n is None:
        from ..benchmark import principal_direction
        n = principal_direction(lf, target)
    sk = int(select_k) if select_k else 32
    best = optimize_huang(lf, target, electrodes, n, pmax_rel=pmax_rel, select_k=sk, seed=seed, verbose=verbose)
    for _k in ("s1", "s2", "els"):
        best.pop(_k, None)                     # drop numpy so it serialises for the GUI, and
                                               # expose only the distributed format
    return best


def _distributed_legacy(lf, target, electrodes, direction=None, Imax=2.0,
                        off_subsample=3000, tgt_lp=400, off_lp=800, restarts=5, maxiter=60,
                        select_k=None, weights=(0.5, 0.5, 0.5), pctl=50, seed=42, verbose=True):
    """
    (Legacy, preserved.) Distributed multi-electrode TI with free two-channel current
    distributions. Each carrier is realised by c0, c1 in R^K:
      E1 = sum c0·L, E2 = sum c1·L, with sum(c0) = sum(c1) = 0 and |c| <= Imax. The number and
      ratio of + and - electrodes is free and electrodes may appear in both channels.

    **The target term is directional**: maximise the envelope 2·min(|n·E1|, |n·E2|) along n
    (an anatomical axis, or the principal 3D-GEVD component).
    **The off-target term is isotropic Tmax** (all directions), minimised. This keeps the
    directional drive while blocking leakage regardless of orientation — the earlier version
    also looked only along n off-target, which was blind to leakage in other directions.

    Objective = directional WP: w1·M1_dir + w2·M2(target directional / off isotropic) - w3·M3,
    solved with multi-restart SLSQP (a smooth strength + selectivity inner objective; restart
    selection uses the true WP and M3 after Imax normalisation).
    It never returns worse than the GEVD directional solution c0 = c1 = c*, which serves as
    both warm start and fallback.
    """
    K = len(electrodes)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)
    if off_subsample and len(off) > off_subsample:
        off = np.sort(rng.choice(off, off_subsample, replace=False))
    Ft3 = np.stack([lf.elec_field(e, tgt) for e in electrodes])   # (K, Nt, 3)
    Fo3 = np.stack([lf.elec_field(e, off) for e in electrodes])   # (K, No, 3)
    _, _, n3 = _gevd3d(Ft3, Fo3)
    n = np.asarray(direction, float) if direction is not None else n3
    n = n / (np.linalg.norm(n) + 1e-30)
    At = (Ft3 @ n).T   # (Nt, K)  target projection along n
    Bo = (Fo3 @ n).T   # (No, K)  off-target projection along n
    lam_dir, cstar = focality_bound(At, Bo)   # directional GEVD: warm start and fallback

    # K-selection: reduce to the top-K by GEVD importance first, so the non-convex solve runs
    # once on the reduced set
    if select_k and select_k < K:
        keep = np.sort(np.argsort(-np.abs(cstar))[:select_k])
        return optimize_distributed(lf, target, [electrodes[i] for i in keep], direction=n, Imax=Imax,
                                    off_subsample=off_subsample, tgt_lp=tgt_lp, off_lp=off_lp,
                                    restarts=restarts, maxiter=maxiter, select_k=None,
                                    weights=weights, pctl=pctl, seed=seed, verbose=verbose)

    w1, w2, w3 = weights; eps = 1e-9
    Nt = At.shape[0]; No = Bo.shape[0]
    til = np.sort(rng.choice(Nt, tgt_lp, replace=False)) if (tgt_lp and Nt > tgt_lp) else np.arange(Nt)
    oil = np.sort(rng.choice(No, off_lp, replace=False)) if (off_lp and No > off_lp) else np.arange(No)
    Atl = At[til]                 # (til, K)  target directional projection (subsample)
    Fo3l = Fo3[:, oil, :]         # (K, Nol, 3)  off-target 3D (subsample), for isotropic Tmax

    def _iso_off(c0, c1, F):
        E0 = np.einsum("k,knd->nd", c0, F); E1 = np.einsum("k,knd->nd", c1, F)
        return ti.tmax(E0, E1)    # isotropic (all-direction) envelope

    def _scale(c0, c1):
        """Fix the total injected current I_total = 0.5·sum|c| = ITOTAL, with <= Imax per
        electrode — the same budget as classic and dual.
        This bounds M1, so the objective solves "where to place a limited current" rather than
        "pour in more current"."""
        itot = 0.5 * (np.abs(c0).sum() + np.abs(c1).sum())
        mx = max(np.abs(c0).max(), np.abs(c1).max(), 1e-9)
        s = min(ITOTAL / max(itot, 1e-12), Imax / mx)
        return c0 * s, c1 * s

    # Inner objective: strength (directional M1) plus selectivity (target directional over
    # off isotropic), normalised at fixed total current so strength stays bounded and the
    # objective stays smooth
    def _neg_obj(x):
        c0, c1 = _scale(x[:K], x[K:])
        dt = 2.0 * np.minimum(np.abs(Atl @ c0), np.abs(Atl @ c1))   # target directional envelope
        oo = _iso_off(c0, c1, Fo3l)                                  # off isotropic envelope
        strength = np.mean(dt) + eps
        sel = (np.mean(dt ** 2) + eps) / (np.mean(oo ** 2) + eps)
        return -(w1 * np.log(strength) + w2 * np.log(sel))

    # The true directional WP over the full target and off pools (isotropic off). c is
    # evaluated after fixing the total current to ITOTAL, which keeps methods comparable.
    def _true(c0, c1):
        C0, C1 = _scale(c0, c1)
        E0t = np.einsum("k,knd->nd", C0, Ft3); E1t = np.einsum("k,knd->nd", C1, Ft3)
        dt = ti.directional_env(E0t, E1t, n)         # 2·min(|n·E0|, |n·E1|), target
        oo = _iso_off(C0, C1, Fo3)                    # off isotropic
        m1 = float(np.median(dt)); m2 = metrics.M2(dt, oo)
        m3 = 100.0 * float(np.mean(oo > np.percentile(dt, pctl)))
        return (m1, m2, m3), C0, C1

    cons = [{"type": "eq", "fun": (lambda x, i=i: x[i * K:(i + 1) * K].sum())} for i in (0, 1)]
    bnds = [(-Imax, Imax)] * (2 * K)
    cand = [(cstar.copy(), cstar.copy())]   # fallback = the directional GEVD solution
    for s in range(restarts):
        x0 = np.concatenate([cstar, cstar])
        if s:
            x0 = np.clip(x0 + rng.normal(0, 0.6, 2 * K), -Imax, Imax)
        res = minimize(_neg_obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-7})
        cand.append((res.x[:K].copy(), res.x[K:].copy()))
    # Evaluate the true metrics for every candidate (~restarts + 1) and pick by WP normalised
    # against the best candidate — this keeps the choice balanced and stable
    cm = [_true(c0, c1) for c0, c1 in cand]
    M1ref = max(m[0] for m, _, _ in cm) or 1.0
    M2ref = max(m[1] for m, _, _ in cm) or 1.0
    def wp(m): return w1 * m[0] / M1ref + w2 * m[1] / M2ref - w3 * m[2] / 100.0
    scored = sorted(cm, key=lambda t: -wp(t[0]))
    mbest, c0, c1 = scored[0]
    if verbose:
        print(f"  directional WP {wp(mbest):.3f} | M1_dir {mbest[0]:.3f} "
              f"M2(iso off) {mbest[1]:.2f} M3 {mbest[2]:.1f}%  "
              f"({len(cm)} candidates, M1ref {M1ref:.3f} M2ref {M2ref:.2f})", flush=True)

    out = _finalize(electrodes, c0, c1, Ft3, Fo3, n, verbose)   # c0, c1 are already Imax-normalised
    out["focality_bound"] = lam_dir
    out["direction"] = [float(x) for x in n]     # for the directional view and reproduction
    out["c0_raw"] = c0.copy(); out["c1_raw"] = c1.copy()      # warm start for the complex extension
    out["electrodes_used"] = list(electrodes)
    return out


def _sphere_dirs(D=24):
    """Uniform directions on the unit sphere (Fibonacci), for the direction scan of the
    isotropic Tmax of a complex (elliptically polarised) solution."""
    i = np.arange(D) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / D)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], 1)


def optimize_distributed_complex(lf, target, electrodes, direction=None, Imax=2.0,
                                 off_subsample=3000, tgt_lp=400, off_lp=600, n_dirs=12,
                                 restarts=3, maxiter=50, select_k=None, weights=(0.5, 0.5, 0.5),
                                 pctl=50, seed=42, verbose=True):
    """Distributed TI with complex (phase) currents: c0, c1 in C^K, i.e. amplitude and phase
    per electrode. Phase makes each carrier elliptically polarised.

    The real optimum (phi = 0) is included as both warm start and candidate, which **guarantees
    the result is no worse than real-valued distributed** and isolates the pure phase gain.
    Target = modulation depth 2·min(|n·E0|, |n·E1|) along n using complex projections;
    off-target = isotropic via a direction scan. Objective = directional WP.
    Returns best = dict(complex=True, electrodes, c0, c1, direction, focality_bound)."""
    # 1) the real distributed optimum, used as the warm start (same reduced electrode set)
    real = optimize_distributed(lf, target, electrodes, direction=direction, Imax=Imax,
                                off_subsample=off_subsample, off_lp=off_lp, select_k=select_k,
                                weights=weights, pctl=pctl, seed=seed, verbose=False)
    els = real["electrodes_used"]
    n = np.asarray(real["direction"], float); n = n / (np.linalg.norm(n) + 1e-30)
    c0r = np.asarray(real["c0_raw"], float); c1r = np.asarray(real["c1_raw"], float)
    K = len(els); eps = 1e-9; w1, w2, _ = weights
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    if off_subsample and len(off) > off_subsample:
        off = np.sort(rng.choice(off, off_subsample, replace=False))
    Ft3 = np.stack([lf.elec_field(e, tgt) for e in els])   # (K, Nt, 3)
    Fo3 = np.stack([lf.elec_field(e, off) for e in els])   # (K, No, 3)
    At = (Ft3 @ n).T                                        # (Nt, K) target projection along n (real)
    Nt = At.shape[0]; No = Fo3.shape[1]
    til = np.sort(rng.choice(Nt, tgt_lp, replace=False)) if (tgt_lp and Nt > tgt_lp) else np.arange(Nt)
    oil = np.sort(rng.choice(No, off_lp, replace=False)) if (off_lp and No > off_lp) else np.arange(No)
    Atl = At[til]
    dirs = _sphere_dirs(n_dirs)
    Bo = np.einsum("knd,md->mnk", Fo3[:, oil, :], dirs)    # (D, off_lp, K) off projections per
                                                           # direction (real)
    Bo_f = np.einsum("knd,md->mnk", Fo3, dirs)             # (D, No, K) full off pool, for
                                                           # selection and the final score

    def _scale(c0, c1):   # fix total current to ITOTAL, with |c0| + |c1| <= Imax per electrode
        itot = 0.5 * (np.abs(c0).sum() + np.abs(c1).sum())
        peak = float(np.max(np.abs(c0) + np.abs(c1))) + 1e-12
        s = min(ITOTAL / max(itot, 1e-12), Imax / peak)
        return c0 * s, c1 * s

    def _unpack(x):
        return x[:K] + 1j * x[K:2 * K], x[2 * K:3 * K] + 1j * x[3 * K:]

    def _neg_obj(x):
        c0, c1 = _scale(*_unpack(x))
        dt = 2.0 * np.minimum(np.abs(Atl @ c0), np.abs(Atl @ c1))
        p0 = np.abs(np.einsum("mnk,k->mn", Bo, c0)); p1 = np.abs(np.einsum("mnk,k->mn", Bo, c1))
        oo = 2.0 * np.max(np.minimum(p0, p1), axis=0)
        strength = np.mean(dt) + eps
        sel = (np.mean(dt ** 2) + eps) / (np.mean(oo ** 2) + eps)
        return -(w1 * np.log(strength) + w2 * np.log(sel))

    def _true(c0, c1):
        c0, c1 = _scale(c0, c1)
        dt = 2.0 * np.minimum(np.abs(At @ c0), np.abs(At @ c1))
        p0 = np.abs(np.einsum("mnk,k->mn", Bo_f, c0)); p1 = np.abs(np.einsum("mnk,k->mn", Bo_f, c1))
        oo = 2.0 * np.max(np.minimum(p0, p1), axis=0)
        m1 = float(np.median(dt)); m2 = float(metrics.M2(dt, oo))
        m3 = 100.0 * float(np.mean(oo > np.percentile(dt, pctl)))
        return (m1, m2, m3), c0, c1

    cons = [{"type": "eq", "fun": (lambda x, a=a: x[a:a + K].sum())} for a in (0, K, 2 * K, 3 * K)]
    bnds = [(-Imax, Imax)] * (4 * K)
    # (a) re-optimise restricted to real values — the fair baseline: same objective, same warm
    #     start, only the phase degrees of freedom removed
    consR = [{"type": "eq", "fun": (lambda xr, a=a: xr[a:a + K].sum())} for a in (0, K)]
    def _neg_real(xr):
        return _neg_obj(np.concatenate([xr[:K], np.zeros(K), xr[K:], np.zeros(K)]))
    rr = minimize(_neg_real, np.concatenate([c0r, c1r]), method="SLSQP",
                  bounds=[(-Imax, Imax)] * (2 * K), constraints=consR,
                  options={"maxiter": maxiter, "ftol": 1e-7})
    x_real = np.concatenate([rr.x[:K], np.zeros(K), rr.x[K:], np.zeros(K)])   # real* (φ=0)
    scale_im0 = 0.4 * (np.abs(rr.x).mean() + 1e-9)
    # (b) complex: free the phases starting from real*, plus perturbed restarts. real* stays a
    #     candidate, which guarantees complex >= real*.
    cand = [_unpack(x_real)]
    for _s in range(restarts):
        x0 = x_real.copy()
        x0[K:2 * K] += rng.normal(0, scale_im0, K); x0[3 * K:] += rng.normal(0, scale_im0, K)
        res = minimize(_neg_obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-7})
        cand.append(_unpack(res.x))
    scored = [_true(c0, c1) for c0, c1 in cand]
    M1r = max(m[0] for m, _, _ in scored) or 1.0
    M2r = max(m[1] for m, _, _ in scored) or 1.0
    def wp(m): return w1 * m[0] / M1r + w2 * m[1] / M2r - weights[2] * m[2] / 100.0
    wps = [wp(m) for m, _, _ in scored]
    best_i = int(np.argmax(wps))
    (m1, m2, m3), c0, c1 = scored[best_i]
    (rm1, rm2, rm3), _, _ = scored[0]     # real* baseline (no phase)
    phase_util = float((np.abs(c0.imag).sum() + np.abs(c1.imag).sum()) /
                       (np.abs(c0).sum() + np.abs(c1).sum() + 1e-12))
    phase_wp_gain = wps[best_i] - wps[0]
    if verbose:
        print(f"  [complex] real* M1 {rm1:.3f} M2 {rm2:.2f} → complex* M1 {m1:.3f} M2 {m2:.2f} "
              f"| pure phase WP gain {phase_wp_gain:+.3f} | phase utilisation {phase_util:.2f}",
              flush=True)
    return dict(complex=True, electrodes=list(els), c0=c0, c1=c1,
                direction=[float(x) for x in n], focality_bound=real.get("focality_bound", 0.0),
                M1_dir=m1, M2_dir=m2, M3_dir=m3)


def optimize_currents(lf, target, electrodes, direction, Ecap=0.25, Imax=2.0,
                      off_subsample=8000, tgt_lp=600, outer=15, add=400, cap=1500,
                      tlim=60, seed=42, verbose=True):
    """
    electrodes : names of the electrodes to use (the K selected by the outer stage)
    direction  : target stimulation direction n (3,), e.g. the hippocampal nL
    return     : dict(currents={ch0:{e:I}, ch1:{e:I}}, M1_dir, M2_dir, M3_dir,
                      M1, M2, M3, electrodes)   ("dir" entries are directional, the rest are
                      isotropic Tmax)
    """
    K = len(electrodes)
    n = np.asarray(direction, float); n = n / (np.linalg.norm(n) + 1e-30)
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)
    rng = np.random.default_rng(seed)
    if off_subsample and len(off) > off_subsample:
        off = np.sort(rng.choice(off, off_subsample, replace=False))

    # 3D fields (K, N, 3) plus the directional projection (N, K)
    Ft3 = np.stack([lf.elec_field(e, tgt) for e in electrodes])   # (K, Nt, 3)
    Fo3 = np.stack([lf.elec_field(e, off) for e in electrodes])   # (K, No, 3)
    At = (Ft3 @ n).T    # (Nt, K)
    Bo = (Fo3 @ n).T    # (No, K)
    Nt = At.shape[0]
    til = np.sort(rng.choice(Nt, tgt_lp, replace=False)) if (tgt_lp and Nt > tgt_lp) else np.arange(Nt)
    At_lp = At[til]; Ntl = len(til)   # target subsample used in the LP constraints

    NV = 2 * K + 1
    c = np.zeros(NV); c[-1] = -1.0
    A_eq = np.zeros((2, NV)); A_eq[0, :K] = 1.0; A_eq[1, K:2 * K] = 1.0
    bounds = [(-Imax, Imax)] * (2 * K) + [(0, None)]

    s0 = np.ones(Ntl); s1 = np.ones(Ntl)
    active = np.array([], int)
    x = None
    for it in range(outer):
        rows = []; rhs = []
        # target max-min: v - s·(At·c) <= 0, for both channels
        R0 = np.zeros((Ntl, NV)); R0[:, :K] = -s0[:, None] * At_lp; R0[:, -1] = 1.0
        R1 = np.zeros((Ntl, NV)); R1[:, K:2 * K] = -s1[:, None] * At_lp; R1[:, -1] = 1.0
        rows += [R0, R1]; rhs += [np.zeros(Ntl), np.zeros(Ntl)]
        # off-target: |Bo·c| <= Ecap/2 on the assigned channel (the envelope is 2·min <= Ecap)
        if len(active):
            op0 = Bo[active] @ x[:K]; op1 = Bo[active] @ x[K:2 * K]
            g = (np.abs(op1) < np.abs(op0)).astype(int)   # whichever channel is smaller
            for sgn in (1.0, -1.0):
                Ro = np.zeros((len(active), NV))
                for ai, (y, gy) in enumerate(zip(active, g)):
                    Ro[ai, gy * K:(gy + 1) * K] = sgn * Bo[y]
                rows.append(Ro); rhs.append(np.full(len(active), Ecap / 2))
        A_ub = sp.csr_matrix(np.vstack(rows)); b_ub = np.concatenate(rhs)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=sp.csr_matrix(A_eq),
                      b_eq=np.zeros(2), bounds=bounds, method="highs",
                      options={"time_limit": tlim})
        if not res.success:
            if verbose: print(f"  iter{it}: LP failed ({res.message[:40]})")
            break
        x = res.x; c0 = x[:K]; c1 = x[K:2 * K]
        tp0 = At_lp @ c0; tp1 = At_lp @ c1
        s0 = np.sign(tp0); s0[s0 == 0] = 1; s1 = np.sign(tp1); s1[s1 == 0] = 1
        op0 = Bo @ c0; op1 = Bo @ c1
        envoff = 2 * np.minimum(np.abs(op0), np.abs(op1))
        viol = np.where(envoff > Ecap * 1.001)[0]
        if verbose:
            print(f"  iter{it}: target min {2*x[-1]:.3f} | off violations {len(viol)} "
                  f"| active {len(active)}", flush=True)
        if len(viol) == 0:
            break
        newy = viol[np.argsort(-envoff[viol])[:add]]
        active = np.union1d(active, newy)
        if len(active) > cap:
            active = active[np.argsort(-envoff[active])[:cap]]

    if x is None:
        raise RuntimeError("the LP found no solution")
    c0 = x[:K]; c1 = x[K:2 * K]
    return _finalize(electrodes, c0, c1, Ft3, Fo3, n, verbose)


def _finalize(electrodes, c0, c1, Ft3, Fo3, n, verbose):
    E1t = np.einsum("k,knd->nd", c0, Ft3); E2t = np.einsum("k,knd->nd", c1, Ft3)
    E1o = np.einsum("k,knd->nd", c0, Fo3); E2o = np.einsum("k,knd->nd", c1, Fo3)
    # isotropic Tmax
    tt = ti.tmax(E1t, E2t); to = ti.tmax(E1o, E2o); thr = np.median(tt)
    iso = dict(M1=float(np.median(tt)), M2=metrics.M2(tt, to),
               M3=100.0 * float(np.mean(to > thr)))
    # directional (along n)
    dt = ti.directional_env(E1t, E2t, n); do = ti.directional_env(E1o, E2o, n)
    thrd = np.median(dt)
    dr = dict(M1=float(np.median(dt)), M2=metrics.M2(dt, do),
              M3=100.0 * float(np.mean(do > thrd)))
    cur = {"ch0": {e: round(float(c0[i]), 3) for i, e in enumerate(electrodes) if abs(c0[i]) > 0.02},
           "ch1": {e: round(float(c1[i]), 3) for i, e in enumerate(electrodes) if abs(c1[i]) > 0.02}}
    if verbose:
        print(f"  → directional (n): M1 {dr['M1']:.3f} M2 {dr['M2']:.2f} M3 {dr['M3']:.1f}%"
              f"  | isotropic: M1 {iso['M1']:.3f} M2 {iso['M2']:.2f} M3 {iso['M3']:.1f}%", flush=True)
    return dict(currents=cur, electrodes=list(electrodes), n_active=len(cur["ch0"]) + len(cur["ch1"]),
                M1_dir=dr["M1"], M2_dir=dr["M2"], M3_dir=dr["M3"],
                M1=iso["M1"], M2=iso["M2"], M3=iso["M3"])
