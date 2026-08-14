# -*- coding: utf-8 -*-
"""
nsga.py — a lightweight in-house NSGA-II multi-objective optimiser (Phase 2, reproducing the
SuMo methodology)
=====================================================================
Finds the classic two-channel Pareto front when the allowed set is too large for an
exhaustive search. No dependencies beyond numpy.
Mirrors SuMo (V5.2): NSGA-II with native constraints, six parallel seeds merged, and
hypervolume-based adaptive convergence.

Encoding: an individual is [4 electrodes (indices into `allowed`), pairing in {0,1,2}, ratio].
Evaluation: montage → M1 (up) / M2 (up) / M3 (down). Verifiable against the Phase 1
exhaustive-search oracle.
"""
import numpy as np
from .classic import pareto_front, weighted_performance, _metrics, format_montage, _cnorm

SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
_PRESET = {  # precision → (pop, max_gen, patience)
    "low":    (40, 40, 8),
    "medium": (60, 80, 12),
    "high":   (100, 150, 20),
}


# ---------- NSGA-II primitives ----------
def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def _nondominated_sort(F):
    n = len(F); S = [[] for _ in range(n)]; nd = np.zeros(n, int); fronts = [[]]
    for p in range(n):
        for q in range(n):
            if _dominates(F[p], F[q]): S[p].append(q)
            elif _dominates(F[q], F[p]): nd[p] += 1
        if nd[p] == 0: fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0: nxt.append(q)
        i += 1; fronts.append(nxt)
    return [f for f in fronts if f]


def _crowding(F, front):
    l = len(front)
    d = np.zeros(l)
    if l <= 2:
        return np.full(l, np.inf)
    Ff = F[front]
    for m in range(F.shape[1]):
        o = np.argsort(Ff[:, m])
        d[o[0]] = d[o[-1]] = np.inf
        span = (Ff[o[-1], m] - Ff[o[0], m]) or 1.0
        d[o[1:-1]] += (Ff[o[2:], m] - Ff[o[:-2], m]) / span
    return d


def _hv_mc(Fmin, ideal, ref, rng, n=15000):
    """Monte-Carlo hypervolume (minimisation). A scalar for monitoring convergence."""
    S = ideal + (ref - ideal) * rng.random((n, Fmin.shape[1]))
    dom = np.zeros(n, bool)
    for f in Fmin:
        dom |= np.all(f <= S, axis=1)
    return dom.mean() * float(np.prod(ref - ideal))


# ---------- individual-level operators ----------
def _repair(e, K, rng):
    e = list(dict.fromkeys(int(x) for x in e))
    while len(e) < 4:
        x = int(rng.integers(K))
        if x not in e: e.append(x)
    return np.array(e[:4], int)


def _rand(K, rng):
    return (rng.choice(K, 4, replace=False),
            int(rng.integers(3)),
            float(np.exp(rng.uniform(np.log(0.3), np.log(3.3)))))


def _cross(pa, pb, K, rng):
    ea = np.where(rng.random(4) < 0.5, pa[0], pb[0])
    e = _repair(ea, K, rng)
    p = pa[1] if rng.random() < 0.5 else pb[1]
    r = float(np.sqrt(pa[2] * pb[2]))
    return [e, p, r]


def _mutate(ind, K, rng, pm=0.3):
    e, p, r = ind
    if rng.random() < pm:
        e = e.copy(); e[rng.integers(4)] = int(rng.integers(K)); e = _repair(e, K, rng)
    if rng.random() < pm:
        p = int(rng.integers(3))
    r = float(np.clip(r * np.exp(rng.normal(0, 0.15)), 0.2, 5.0))
    return [e, p, r]


# ---------- evaluation ----------
def _eval(ind, Ta, Oa, pctl=50):
    e, p, r = ind
    (i, j), (k, l) = SPLITS[p]
    A, B, C, D = e[i], e[j], e[k], e[l]
    E1t = Ta[A] - Ta[B]; E2t = r * (Ta[C] - Ta[D])
    E1o = Oa[A] - Oa[B]; E2o = r * (Oa[C] - Oa[D])
    m = _metrics(E1t, E2t, E1o, E2o, pctl); s = _cnorm(r)   # M3 at the p-percentile; larger channel normalised to I_max
    return (m[0] * s, m[1], m[2])                            # (M1, M2, M3)


def _fmin(m):
    return np.array([-m[0], -m[1], m[2]])


# ---------- main ----------
def optimize_nsga(lf, target, allowed=None, precision="medium", seeds=6,
                  tgt_scan=2000, off_scan=6000, off_refine=20000,
                  min_dist_mm=0.0, weights=(0.5, 0.5, 0.5), pctl=50, hv_tol=0.002,
                  seed=0, verbose=True, progress=None):
    names = [e for e in (allowed if allowed is not None else list(lf.names)) if lf.has(e)]
    K = len(names)
    if K < 4:
        raise ValueError(f"fewer than 4 allowed electrodes: {names}")
    pop, max_gen, patience = _PRESET[precision]
    tgt = np.asarray(target.target_idx); off = np.asarray(target.off_idx)

    master = np.random.default_rng(seed)
    tsub = tgt if len(tgt) <= tgt_scan else np.sort(master.choice(tgt, tgt_scan, replace=False))
    osub = off if len(off) <= off_scan else np.sort(master.choice(off, off_scan, replace=False))
    Ta = np.stack([lf.elec_field(e, tsub) for e in names])   # (K, Nt, 3)
    Oa = np.stack([lf.elec_field(e, osub) for e in names])   # (K, No, 3)

    pos = getattr(lf, "pos", {}) or {}
    def feasible(e):
        if min_dist_mm <= 0 or not pos:
            return True
        ns = [names[i] for i in e]
        for x in range(4):
            for y in range(x + 1, 4):
                if ns[x] in pos and ns[y] in pos and \
                   np.linalg.norm(np.subtract(pos[ns[x]], pos[ns[y]])) < min_dist_mm:
                    return False
        return True

    n_eval = 0
    seed_fronts = []
    for s in range(seeds):
        rng = np.random.default_rng(master.integers(1 << 30))
        # initial population (feasible)
        P = []
        while len(P) < pop:
            ind = list(_rand(K, rng))
            if feasible(ind[0]):
                P.append(ind)
        M = np.array([_eval(i, Ta, Oa, pctl) for i in P]); n_eval += pop
        F = np.array([_fmin(m) for m in M])
        ref = F.max(0) + 0.1 * (np.abs(F.max(0)) + 1e-9)
        ideal = F.min(0)
        hv_hist = []
        for g in range(max_gen):
            if progress:
                progress(0.05 + 0.80 * (s + g / max_gen) / seeds, "NSGA 진화")
            # produce offspring
            fronts = _nondominated_sort(F); rank = np.zeros(pop, int)
            for ri, fr in enumerate(fronts):
                for idx in fr: rank[idx] = ri
            crowd = np.zeros(pop)
            for fr in fronts:
                cd = _crowding(F, fr)
                for kk, idx in enumerate(fr): crowd[idx] = cd[kk]
            def tourn():
                a, b = rng.integers(pop), rng.integers(pop)
                if rank[a] != rank[b]: return a if rank[a] < rank[b] else b
                return a if crowd[a] >= crowd[b] else b
            Q = []
            while len(Q) < pop:
                c = _mutate(_cross(P[tourn()], P[tourn()], K, rng), K, rng)
                if feasible(c[0]): Q.append(c)
            MQ = np.array([_eval(i, Ta, Oa, pctl) for i in Q]); n_eval += pop
            FQ = np.array([_fmin(m) for m in MQ])
            # select the next generation from parents plus offspring
            allP = P + Q; allF = np.vstack([F, FQ]); allM = np.vstack([M, MQ])
            fronts = _nondominated_sort(allF)
            newidx = []
            for fr in fronts:
                if len(newidx) + len(fr) <= pop:
                    newidx += fr
                else:
                    cd = _crowding(allF, fr)
                    order = np.array(fr)[np.argsort(-cd)]
                    newidx += list(order[:pop - len(newidx)]); break
            P = [allP[i] for i in newidx]; F = allF[newidx]; M = allM[newidx]
            # convergence measured by the hypervolume of the first front
            f0 = _nondominated_sort(F)[0]
            hv = _hv_mc(np.clip(F[f0], ideal, ref), ideal, ref, rng)
            hv_hist.append(hv)
            if len(hv_hist) > patience:
                base = hv_hist[-patience - 1]
                if (hv - base) <= hv_tol * max(base, 1e-9):
                    break
        f0 = _nondominated_sort(F)[0]
        for i in f0:
            seed_fronts.append((P[i], tuple(M[i])))
        if verbose:
            print(f"  seed {s}: gen {g+1}, front {len(f0)}, HV {hv_hist[-1]:.4g}")

    # ---- merge the six seeds, take the non-dominated set, refine against the full off pool ----
    uniq = {}
    for ind, m in seed_fronts:
        key = (tuple(sorted([names[x] for x in [ind[0][SPLITS[ind[1]][0][0]], ind[0][SPLITS[ind[1]][0][1]]]])),
               tuple(sorted([names[x] for x in [ind[0][SPLITS[ind[1]][1][0]], ind[0][SPLITS[ind[1]][1][1]]]])),
               round(ind[2], 2))
        uniq[key] = ind
    inds = list(uniq.values())
    off_r = off if len(off) <= off_refine else np.sort(master.choice(off, off_refine, replace=False))
    Tr = np.stack([lf.elec_field(e, tgt) for e in names])
    Or = np.stack([lf.elec_field(e, off_r) for e in names])
    recs = []
    for ind in inds:
        m = _eval(ind, Tr, Or, pctl); e, p, r = ind
        (i, j), (k, l) = SPLITS[p]
        recs.append(((names[e[i]], names[e[j]]), (names[e[k]], names[e[l]]), float(r), *m))
    M1 = np.array([x[3] for x in recs]); M2 = np.array([x[4] for x in recs]); M3 = np.array([x[5] for x in recs])
    WP = weighted_performance(M1, M2, M3, weights); par = pareto_front(M1, M2, M3)
    montages = [dict(ch1=x[0], ch2=x[1], ratio=x[2], M1=x[3], M2=x[4], M3=x[5],
                     WP=float(WP[i]), pareto=bool(par[i])) for i, x in enumerate(recs)]
    if verbose:
        print(f"[NSGA-II] allowed {K} | evaluated {n_eval} | merged candidates {len(recs)} "
              f"| pareto {int(par.sum())}")
    return dict(montages=montages, best=montages[int(np.argmax(WP))],
                n_eval=n_eval, n_pareto=int(par.sum()), allowed=names)
