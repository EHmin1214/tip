# -*- coding: utf-8 -*-
"""
report.py — plots
============================
plot_montage     : head plot of the chosen montage (two-channel wiring, or distributed currents)
plot_pareto      : Pareto front (M1 vs M2, colour = M3, the best one starred)
plot_field_slice : brain slice of the TI envelope

Projection: top-down axial — X (left-right) against -Z (anterior points up). Y is superior.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
try:
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
from . import ti


def _xy(lf, e):
    p = lf.pos[e]; return (p[0], -p[2])   # X (left-right), -Z (anterior up)


def _draw_head(ax, lf, target=None):
    xs, ys = zip(*[_xy(lf, e) for e in lf.names if e in lf.pos])
    xs, ys = np.array(xs), np.array(ys)
    cx, cy = xs.mean(), ys.mean()
    w = (xs.max() - xs.min()) * 1.3; h = (ys.max() - ys.min()) * 1.3
    ax.add_patch(Ellipse((cx, cy), w, h, fill=False, lw=2, ec="#aaa"))
    ax.plot([cx], [ys.max() + 0.07 * h], marker="^", ms=13, color="#aaa")
    ax.text(cx, ys.max() + 0.14 * h, "front", ha="center", fontsize=9, color="#888")
    for e in lf.names:
        if e in lf.pos:
            x, y = _xy(lf, e); ax.scatter([x], [y], s=64, c="#ececec", ec="#ccc", lw=0.5, zorder=2)
    if target is not None:
        c = target.summary()["center"]; ax.plot(c[0], -c[2], marker="*", ms=22,
            color="#e8b23a", mec="#a07000", mew=1.2, zorder=6)
        ax.text(c[0] + 6, -c[2], target.name, fontsize=8, color="#806000", va="center", zorder=6)
    ax.set_aspect("equal"); ax.axis("off")


def _montage_fields(lf, m, idx):
    if "currents" in m:
        c = m["currents"]; z = np.zeros((len(idx), 3))
        E1 = sum((lf.elec_field(e, idx) * I for e, I in c["ch0"].items()), z.copy())
        E2 = sum((lf.elec_field(e, idx) * I for e, I in c["ch1"].items()), z.copy())
        return E1, E2
    a, b = m["ch1"]; cc, d = m["ch2"]; r = m.get("ratio", 1.0)
    from .optimize.classic import channel_currents
    i1, i2 = channel_currents(r)   # total injected current = ITOTAL (one classic montage)
    return (i1 * (lf.elec_field(a, idx) - lf.elec_field(b, idx)),
            i2 * (lf.elec_field(cc, idx) - lf.elec_field(d, idx)))


def plot_montage(lf, m, target=None, out="montage.png", title="Chosen montage"):
    fig, ax = plt.subplots(figsize=(7, 7)); _draw_head(ax, lf, target)
    C0, C1 = "#d94a3d", "#3671c4"
    if "currents" in m:
        vals = [abs(v) for d in m["currents"].values() for v in d.values()] or [1e-9]
        mx = max(vals)
        for e, I in m["currents"]["ch0"].items():
            x, y = _xy(lf, e); ax.scatter([x], [y], s=80 + abs(I) / mx * 560, c=C0, ec="#333", zorder=4, alpha=.85)
            ax.text(x, y - 10, e, ha="center", fontsize=7, fontweight="bold")
        for e, I in m["currents"]["ch1"].items():
            x, y = _xy(lf, e); ax.scatter([x], [y], s=80 + abs(I) / mx * 560, fc="white", ec=C1, lw=2, zorder=4)
            ax.text(x, y - 10, e, ha="center", fontsize=7)
        ax.scatter([], [], s=180, c=C0, ec="#333", label="channel 1 f1")
        ax.scatter([], [], s=180, fc="white", ec=C1, lw=2, label="channel 2 f2")
        sub = (f"directional M1 {m.get('M1_dir', float('nan')):.3f}  "
               f"M2 {m.get('M2_dir', float('nan')):.2f}  M3 {m.get('M3_dir', float('nan')):.1f}%")
    else:
        for pair, col, lab in [(m["ch1"], C0, "channel 1 f1"), (m["ch2"], C1, "channel 2 f2")]:
            a, b = pair; xa, ya = _xy(lf, a); xb, yb = _xy(lf, b)
            ax.plot([xa, xb], [ya, yb], color=col, lw=2.6, zorder=3, label=lab)
            for e in pair:
                x, y = _xy(lf, e); ax.scatter([x], [y], s=260, c="white", ec=col, lw=2.5, zorder=5)
                ax.text(x, y, e, ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)
        sub = f"M1 {m['M1']:.3f}  M2 {m['M2']:.2f}  M3 {m['M3']:.1f}%  (ratio {m.get('ratio',1):.2f})"
    ax.legend(loc="lower center", fontsize=9, ncol=2, framealpha=.9)
    ax.set_title(f"{title}\n{sub}", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


def plot_pareto(result, out="pareto.png", title="Pareto front"):
    ms = result["montages"]
    M1 = np.array([m["M1"] for m in ms]); M2 = np.array([m["M2"] for m in ms])
    M3 = np.array([m["M3"] for m in ms]); par = np.array([m["pareto"] for m in ms])
    fig, ax = plt.subplots(figsize=(7.6, 6))
    if (~par).any():
        ax.scatter(M1[~par], M2[~par], c=M3[~par], cmap="viridis_r", s=22, alpha=.35)
    sc = ax.scatter(M1[par], M2[par], c=M3[par], cmap="viridis_r", s=95, ec="#111", lw=1, zorder=4)
    b = result["best"]; ax.scatter([b["M1"]], [b["M2"]], marker="*", s=420, c="#e8b23a",
                                   ec="#a07000", lw=1.2, zorder=6, label="best (WP)")
    plt.colorbar(sc, label="M3 collateral % (lower is better)")
    ax.set_xlabel("M1 strength (V/m)"); ax.set_ylabel("M2 selectivity")
    ax.set_title(title, fontsize=12, fontweight="bold"); ax.legend(); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out


def plot_field_slice(lf, m, target, out="field.png", axis="z", half=3.0, title="TI envelope"):
    ai = {"x": 0, "y": 1, "z": 2}[axis]
    c = target.summary()["center"]
    coords = lf.coords()
    sl = np.where(np.abs(coords[:, ai] - c[ai]) < half)[0]
    E1, E2 = _montage_fields(lf, m, sl)
    T = ti.tmax(E1, E2)
    ox, oy = [i for i in range(3) if i != ai]
    px, py = coords[sl, ox], coords[sl, oy]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    scv = ax.scatter(px, py, c=T, cmap="inferno", s=7, vmin=0, vmax=np.percentile(T, 99))
    tin = np.isin(sl, target.target_idx)
    ax.scatter(px[tin], py[tin], s=10, facecolors="none", edgecolors="cyan", lw=0.35, zorder=4)
    plt.colorbar(scv, label="Tmax envelope (V/m)")
    ax.set_xlabel("XYZ"[ox] + " (mm)"); ax.set_ylabel("XYZ"[oy] + " (mm)")
    ax.set_aspect("equal"); ax.set_title(f"{title} — {axis}={c[ai]:.0f}mm slice\n(cyan = target)",
                                         fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out
