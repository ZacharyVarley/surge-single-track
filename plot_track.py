#!/usr/bin/env python3
"""Plot a SURGE track: melt pool snapshot during the scan and grain orientations after it.

usage: python plot_track.py <track.npz> <out.png>

Orientations use IPF-Z colours: the colour of the crystal direction along the build direction z
(red <001>, green <101>, blue <111>). Unmelted substrate is drawn darker; the black line is the
fusion boundary (the edge of the region that melted).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, SURFACE, BLANK = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe8"
HEAT = LinearSegmentedColormap.from_list("heat", ["#fbe9df", "#f4a77f", "#eb6834", "#b8461b", "#6e2a0f"])
HEAT.set_bad(BLANK)
plt.rcParams.update({"font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def ipf_colour(d):
    """IPF colour of crystal directions d (..., 3): red <001>, green <101>, blue <111>."""
    a, b, c = np.moveaxis(np.sort(np.abs(d), -1), -1, 0)
    rgb = np.stack([c - b, b - a, a * np.sqrt(3)], -1)
    return rgb / rgb.max(-1, keepdims=True)


def ipf_z(q):
    """IPF-Z colour of crystal -> sample quaternions q (n, 4) (w, x, y, z)."""
    w, x, y, z = q.T
    zc = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)  # sample z in crystal frame
    return ipf_colour(zc)


def ipf_key(ax):
    """The standard triangle [001]-[101]-[111] in stereographic projection, coloured."""
    n = 300
    X, Y = np.meshgrid(np.linspace(0, 0.42, n), np.linspace(0, 0.38, n))
    r2 = X ** 2 + Y ** 2
    d = np.stack([2 * X, 2 * Y, 1 - r2], -1) / (1 + r2)[..., None]
    inside = (d[..., 1] >= 0) & (d[..., 0] >= d[..., 1]) & (d[..., 2] >= d[..., 0])
    img = np.where(inside[..., None], ipf_colour(d), 1.0)
    ax.imshow(np.dstack([img, inside.astype(float)]), origin="lower", extent=[0, 0.42, 0, 0.38])
    for (u, v), s, ha in (((0, 0), "001", "right"), ((0.414, 0), "101", "left"), ((0.366, 0.366), "111", "left")):
        ax.annotate(s, (u, v), xytext=(-3 if ha == "right" else 3, 0), textcoords="offset points",
                    ha=ha, va="center", color=INK2)
    ax.set_xlim(-0.08, 0.5); ax.set_ylim(-0.03, 0.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("IPF-Z key", loc="left")


def show(ax, img, extent, title, xlabel=True):
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
    ax.set_title(title, loc="left")
    ax.set_ylabel("µm")
    if xlabel:
        ax.set_xlabel("µm")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def outline(ax, mask, extent, **kw):
    if mask.any() and not mask.all():
        ax.contour(mask.T.astype(float), levels=[0.5], extent=extent, origin="lower", **kw)


def main():
    d = np.load(sys.argv[1])
    c = json.loads(str(d["config"]))
    gid, q, melted = d["grain_id"], d["orientations"], d["melted"]
    nx, ny, nz = gid.shape
    h = float(d["dx_um"])
    ext_top, ext_long, ext_cross = [0, nx * h, 0, ny * h], [0, nx * h, 0, nz * h], [0, ny * h, 0, nz * h]
    col = ipf_z(q)
    dim = lambda rgb, m: np.where(m[..., None], rgb, rgb * 0.5)
    jc, kt, ic = ny // 2, nz - 1, nx // 2
    has_pool = "pool_T_K" in d.files

    fig = plt.figure(figsize=(10, 9.6 if has_pool else 6.6))
    rows = ([ny, nz] if has_pool else []) + [ny, nz, nz * 1.6]
    gs = fig.add_gridspec(len(rows), 1, height_ratios=rows, hspace=0.62)
    r = 0
    if has_pool:
        T, liq = d["pool_T_K"], d["pool_liquid"]
        vmin, vmax = float(d["pool_T_min_K"]), float(np.nanpercentile(T, 99.9))
        t_ms = float(d["pool_time_s"]) * 1e3
        for img, lmask, ext, name in ((T[:, :, kt], liq[:, :, kt], ext_top, "top surface"),
                                      (T[:, jc, :], liq[:, jc, :], ext_long, "longitudinal section y = centre")):
            ax = fig.add_subplot(gs[r])
            im = ax.imshow(img.T, origin="lower", extent=ext, cmap=HEAT, vmin=vmin, vmax=vmax,
                           interpolation="nearest", aspect="equal")
            outline(ax, lmask, ext, colors=INK, linewidths=1.0)
            ax.set_title(f"Melt pool at t = {t_ms:.2f} ms, {name} (temperature; black line: liquid)",
                         loc="left")
            ax.set_ylabel("µm")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            cb = fig.colorbar(im, cax=ax.inset_axes([1.015, 0, 0.012, 1]))
            cb.set_label("K", rotation=0, labelpad=8)
            cb.outline.set_visible(False)
            r += 1

    show(fig.add_subplot(gs[r]), dim(col[gid[:, :, kt]], melted[:, :, kt]).transpose(1, 0, 2), ext_top,
         "Grains after solidification, top surface (IPF-Z colours; substrate darker)", xlabel=False)
    outline(plt.gca(), melted[:, :, kt], ext_top, colors=INK, linewidths=0.8)
    r += 1
    show(fig.add_subplot(gs[r]), dim(col[gid[:, jc, :]], melted[:, jc, :]).transpose(1, 0, 2), ext_long,
         "Grains, longitudinal section y = centre")
    outline(plt.gca(), melted[:, jc, :], ext_long, colors=INK, linewidths=0.8)
    r += 1

    sub = gs[r].subgridspec(1, 3, width_ratios=[1.0, 0.55, 1.25], wspace=0.25)
    ax = fig.add_subplot(sub[0])
    show(ax, dim(col[gid[ic]], melted[ic]).transpose(1, 0, 2), ext_cross, "Cross-section x = centre")
    outline(ax, melted[ic], ext_cross, colors=INK, linewidths=0.8)
    ipf_key(fig.add_subplot(sub[1]))
    ax = fig.add_subplot(sub[2]); ax.axis("off")
    L, n = c["laser"], c["noise"]
    s = json.load(open(sys.argv[1].replace("track.npz", "summary.json"))) if sys.argv[1].endswith("track.npz") else {}
    lines = [f"P = {L['power_W']:g} W (absorbed {L['power_W'] * L['absorptivity']:.0f} W),  v = {L['speed_m_s']:g} m/s",
             f"preheat {L['preheat_K']:g} K,  spot {L['spot_um']:g} µm,  noise σP = {n['sigma_P']:g}",
             f"grid {nx} × {ny} × {nz} at {h:g} µm"]
    if s:
        lines += [f"fusion zone: depth {s['depth_um']:.0f} µm, width {s['width_um']:.0f} µm",
                  f"grains in it: {s['grains_in_melt']} ({s['new_grains']} new)",
                  f"SURGE run time {s['surge_seconds']:.0f} s"]
    ax.text(0, 0.95, "\n".join(lines), va="top", ha="left", linespacing=1.6, color=INK)
    fig.savefig(sys.argv[2], dpi=150, bbox_inches="tight")
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
