#!/usr/bin/env python3
"""Run a SURGE single-track simulation from a TOML config and save the result to track.npz.

usage: python run_track.py <config.toml> <out dir> [--keep]

1. Builds the starting plate: Voronoi grains with random orientations (substrate.sites).
2. Writes the SPPARKS input deck (in.track) for the GPU app additive/texture/kk.
3. Runs SURGE ($SURGE_BIN, set by env.sh) in <out dir>; its screen output goes to surge.log.
4. Reads the dumps and writes <out dir>/track.npz and summary.json (see README.md, "Outputs").
   The large text files (substrate.sites and the dumps) are deleted unless --keep is given.
"""
import argparse, json, os, subprocess, sys, time, tomllib
import numpy as np
from scipy.spatial import cKDTree


def make_substrate(nx, ny, nz, dx_um, esd_um, seed):
    """Voronoi grains of about esd_um equivalent sphere diameter. Returns the grain map
    (nx, ny, nz) and unit quaternions (n, 4) (w, x, y, z), uniform on SO(3)."""
    rng = np.random.default_rng(seed)
    size = np.array([nx, ny, nz]) * dx_um
    n = max(1, int(np.prod(size) / (np.pi / 6 * esd_um ** 3)))
    centres = rng.random((n, 3)) * size
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1)
    _, g = cKDTree(centres).query(ijk.reshape(-1, 3) * dx_um)
    return g.reshape(nx, ny, nz).astype(np.int32), q.astype(np.float32)


def site_ids(nx, ny, nz):
    """SPPARKS site id of voxel (i, j, k) after create_sites on region block 0 nx 0 ny 0 nz."""
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return 1 + i + (nx + 1) * j + (nx + 1) * (ny + 1) * k


def write_sites(path, gmap, quats):
    """read_sites file: spin i1 = grain + 1, active flag i2 = 3 (solid), quaternion d1..d4."""
    nx, ny, nz = gmap.shape
    ids = site_ids(nx, ny, nz).ravel()
    per_grain = [f" {g + 1} 3 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n" for g, (a, b, c, d) in enumerate(quats)]
    with open(path, "w") as f:
        f.write(f"SURGE substrate\n\n3 dimension\n{ids.size} sites\n0 {nx} xlo xhi\n0 {ny} ylo yhi\n"
                f"0 {nz} zlo zhi\nid i1 i2 d1 d2 d3 d4 values\n\nValues\n\n")
        f.writelines(str(s) + per_grain[g] for s, g in zip(ids.tolist(), gmap.ravel().tolist()))


def times(c):
    t_scan = c["laser"]["track_mm"] * 1e-3 / c["laser"]["speed_m_s"]
    return t_scan, t_scan + c["time"]["extra_time_s"]


def pool_threshold(c):
    """The temperature snapshot keeps sites hotter than this (K)."""
    return max(c["laser"]["preheat_K"] + 1.0, c["material"]["solidus_K"] - 400.0)


def deck(c):
    d, L, th, m, tx, tm, r, nz_ = (c[k] for k in ("domain", "laser", "thermal", "material", "texture",
                                                  "time", "run", "noise"))
    nx, ny, nz = d["nx"], d["ny"], d["nz"]
    dx = d["dx_um"] * 1e-6
    track = L["track_mm"] * 1e-3
    x0 = (nx * dx - track) / 2
    if x0 <= 0:
        sys.exit(f"track_mm = {L['track_mm']} does not fit in nx * dx = {nx * dx * 1e3:.3f} mm")
    yc, zt = ny / 2 * dx, (nz - 0.5) * dx          # laser plane half a voxel above the top sites
    t_scan, t_end = times(c)
    T0 = L["preheat_K"]
    onoff = lambda b: "on" if b else "off"
    front = m["solid_front_vel"]
    lines = [
        "# app: nspins, voxel size dx (m), time step dt (s), smoothing passes nrefine",
        f"app_style        additive/texture/kk {r['nspins']} {dx:.6e} {tm['dt_s']:.6e} {tm['nrefine']}",
        f"seed             {r['kmc_seed']}",
        "dimension        3",
        "lattice          sc/26n 1.0",
        f"region           box block 0 {nx} 0 {ny} 0 {nz}",
        "boundary         n n n",
        "create_box       box",
        "create_sites     box",
        "read_sites       substrate.sites",
        f"set              d6 value {T0}",
        "temperature      0.0",
        "",
        "# solidification, nucleation and grain competition",
        f"liquidus         {m['liquidus_K']}",
        f"solidus          {m['solidus_K']}",
        f"solid_front_vel  {len(front)} " + " ".join(f"{v:.6g}" for v in front),
        f"nucleation_density      {m['nucleation_density_m3']:.6g}",
        f"critical_undercooling   {m['critical_undercooling_K']}",
        f"undercooling_deviation  {m['undercooling_deviation_K']}",
        f"mean_nuclei_volume      {m['mean_nuclei_volume_m3']:.6g}",
        f"nuclei_volume_deviation {m['nuclei_volume_deviation_m3']:.6g}",
        f"texture_parameters {tx['c2']} {tx['c3']}",
        f"smooth_greedy_multiproposal {onoff(tx['smooth_greedy_multiproposal'])}",
        f"smooth_orientation {tx['smooth_orientation']}",
        f"nuclei_smooth {onoff(tx['nuclei_smooth'])}",
        *c.get("extra", {}).get("commands", []),
        "",
        "# heat source: moser P absorptivity k alpha T0 cp sx sy sz; then noise and the scan",
        f"setup_temperature_source moser {L['power_W']} {L['absorptivity']} {th['conductivity_W_mK']} "
        f"{th['diffusivity_m2_s']:.6g} {T0} {th['heat_capacity_J_kgK']} "
        f"{L['spot_um'] * 1e-6:.4e} {L['spot_um'] * 1e-6:.4e} {L['depth_um'] * 1e-6:.4e}",
    ]
    if max(nz_["sigma_P"], nz_["sigma_W"], nz_["sigma_D"]) > 0:
        shape = {"lorentzian": f" tau {nz_.get('tau_s', 0):.6g}",
                 "narrow_band": f" f0 {nz_.get('f0_Hz', 0):.6g} df {nz_.get('df_Hz', 0):.6g}"}.get(nz_["psd"], "")
        lines.append(f"laser_fluctuations psd {nz_['psd']} sigma_W {nz_['sigma_W']} sigma_D {nz_['sigma_D']} "
                     f"sigma_P {nz_['sigma_P']}{shape} rho {nz_['rho']} seed {nz_['seed']} dt {nz_['dt_s']:.6g}")
    lines += [
        f"laser_path start {x0:.6e} {yc:.6e} {zt:.6e} end {x0 + track:.6e} {yc:.6e} "
        f"speed {L['speed_m_s']} repeats 1 pause_below {tm['pause_below_K']}",
        f"fast_forward_search_window {r['fast_forward_search_window_s']:.6g}",
        "",
        "# output: melted sites (grain i1, quaternion d1-d4, G d8, V d9), leftover liquid, T snapshot",
        f"stats            {t_end / 4:.6e}",
        f"dump             1 text {t_end:.6e} melt.txt id i1 d1 d2 d3 d4 d8 d9",
        f"dump_modify      1 thresh d8 != 0 delay {t_end * 0.999:.6e}",
        f"dump             2 text {t_end:.6e} liquid.txt id i2",
        f"dump_modify      2 thresh i2 == 2 delay {t_end * 0.999:.6e}",
    ]
    if r["snapshot_at"] > 0:
        ts = r["snapshot_at"] * t_scan          # dumps repeat every ts; run_track reads the first
        lines += [f"dump             3 text {ts:.6e} pool.txt id i2 d6",
                  f"dump_modify      3 thresh d6 > {pool_threshold(c):.1f} delay {ts * 0.999:.6e}"]
    lines += [f"run              {t_end:.6e}", ""]
    return f"# SURGE track, written by run_track.py from {c['_path']}\n" + "\n".join(lines)


def read_frame(path, which=-1):
    """One frame (first: 0, last: -1) of a SPPARKS text dump -> (time, column names, array)."""
    with open(path) as f:
        frames = f.read().split("ITEM: TIMESTEP\n")[1:]
    if not frames:
        return None, [], np.zeros((0, 0))
    lines = frames[which].splitlines()
    t = float(lines[0].split()[1])
    a = next(i for i, s in enumerate(lines) if s.startswith("ITEM: ATOMS"))
    names = lines[a].split()[2:]
    body = lines[a + 1:]
    arr = np.loadtxt(body, ndmin=2) if body else np.zeros((0, len(names)))
    return t, names, arr


def ijk(ids, nx, ny):
    k, rest = np.divmod(ids.astype(np.int64) - 1, (nx + 1) * (ny + 1))
    j, i = np.divmod(rest, nx + 1)
    return i, j, k


def relabel(gsub, qsub, ids, spins, quats):
    """Final grain map and orientation table. A melted site keeps its substrate grain's label when
    its spin is that grain and its orientation matches (epitaxial growth); every other
    (spin, orientation) pair gets a new label after the substrate grains."""
    nx, ny, nz = gsub.shape
    i, j, k = ijk(ids, nx, ny)
    final = gsub.copy()
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    quats[quats[:, 0] < 0] *= -1
    ngr = len(qsub)
    sp = spins.astype(np.int64) - 1
    same = np.zeros(len(sp), bool)
    sub = (sp >= 0) & (sp < ngr)
    same[sub] = np.abs(np.sum(qsub[sp[sub]] * quats[sub], 1)) > 1 - 1e-4
    lab = np.where(same, sp, -1)
    key = np.column_stack([sp[~same], np.round(quats[~same] * 1e4).astype(np.int64)])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    lab[~same] = ngr + inv.ravel()
    new_q = np.zeros((len(uniq), 4), np.float32)
    new_q[inv.ravel()] = quats[~same]
    final[i, j, k] = lab
    melted = np.zeros(gsub.shape, bool)
    melted[i, j, k] = True
    return final, np.concatenate([qsub, new_q]).astype(np.float32), melted, (i, j, k)


def summarize(melted, gid, nsub, dx_um, nz):
    """Fusion-zone depth and width (median over the middle half of the domain in x), grain counts."""
    nx = melted.shape[0]
    rows = []
    for x in range(nx // 4, 3 * nx // 4):
        js, ks = np.nonzero(melted[x])
        if js.size:
            rows.append(((nz - ks.min()) * dx_um, (js.max() - js.min() + 1) * dx_um))
    g = np.unique(gid[melted])
    return dict(depth_um=float(np.median([r[0] for r in rows])) if rows else 0.0,
                width_um=float(np.median([r[1] for r in rows])) if rows else 0.0,
                melted_voxels=int(melted.sum()), grains_in_melt=int(g.size),
                new_grains=int((g >= nsub).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("out")
    ap.add_argument("--keep", action="store_true", help="keep substrate.sites and the raw dumps")
    a = ap.parse_args()
    with open(a.config, "rb") as f:
        c = tomllib.load(f)
    c["_path"] = os.path.basename(a.config)
    binary = os.environ.get("SURGE_BIN", "")
    if not os.access(binary, os.X_OK):
        sys.exit(f"SURGE_BIN = '{binary}' is not an executable; run build.sh, then source env.sh")
    os.makedirs(a.out, exist_ok=True)
    d = c["domain"]
    nx, ny, nz, dx_um = d["nx"], d["ny"], d["nz"], d["dx_um"]
    t0 = time.time()
    gsub, qsub = make_substrate(nx, ny, nz, dx_um, c["substrate"]["grain_size_um"], c["substrate"]["seed"])
    write_sites(os.path.join(a.out, "substrate.sites"), gsub, qsub)
    with open(os.path.join(a.out, "in.track"), "w") as f:
        f.write(deck(c))
    print(f"{nx} x {ny} x {nz} voxels, {len(qsub)} substrate grains; deck {a.out}/in.track "
          f"({time.time() - t0:.1f} s)", flush=True)

    t1 = time.time()
    with open(os.path.join(a.out, "surge.log"), "w") as log:
        rc = subprocess.run([binary, "-in", "in.track", "-log", "none"], cwd=a.out,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    t_run = time.time() - t1
    if rc != 0:
        tail = open(os.path.join(a.out, "surge.log")).read()[-3000:]
        sys.exit(f"SURGE exited with code {rc} after {t_run:.0f} s; end of surge.log:\n{tail}")
    print(f"SURGE finished in {t_run:.1f} s", flush=True)

    _, names, melt = read_frame(os.path.join(a.out, "melt.txt"))
    col = {n: i for i, n in enumerate(names)}
    G = np.zeros(gsub.shape, np.float32)
    V = np.zeros(gsub.shape, np.float32)
    if melt.shape[0]:
        gid, qtab, melted, (i, j, k) = relabel(gsub, qsub, melt[:, col["id"]], melt[:, col["i1"]],
                                               melt[:, [col["d1"], col["d2"], col["d3"], col["d4"]]])
        G[i, j, k], V[i, j, k] = melt[:, col["d8"]], melt[:, col["d9"]]
    else:
        gid, qtab, melted = gsub.copy(), qsub, np.zeros(gsub.shape, bool)
    _, _, liq = read_frame(os.path.join(a.out, "liquid.txt"))

    out = dict(grain_id=gid.astype(np.uint32), orientations=qtab,
               substrate_grain_id=gsub.astype(np.uint32), substrate_orientations=qsub,
               melted=melted, G_K_per_m=G, V_m_per_s=V, liquid_left=np.int64(liq.shape[0]),
               dx_um=np.float64(dx_um), config=np.array(json.dumps(c)))
    pool_file = os.path.join(a.out, "pool.txt")
    if c["run"]["snapshot_at"] > 0 and os.path.exists(pool_file):
        tp, pn, pool = read_frame(pool_file, 0)
        pc = {n: i for i, n in enumerate(pn)}
        T = np.full(gsub.shape, np.nan, np.float32)
        liquid = np.zeros(gsub.shape, bool)
        if pool.shape[0]:
            i, j, k = ijk(pool[:, pc["id"]], nx, ny)
            T[i, j, k] = pool[:, pc["d6"]]
            liquid[i, j, k] = pool[:, pc["i2"]] == 2
        out.update(pool_T_K=T, pool_liquid=liquid, pool_time_s=np.float64(tp),
                   pool_T_min_K=np.float64(pool_threshold(c)))
    np.savez_compressed(os.path.join(a.out, "track.npz"), **out)

    s = summarize(melted, gid, len(qsub), dx_um, nz)
    s.update(liquid_left=int(liq.shape[0]), substrate_grains=len(qsub), surge_seconds=round(t_run, 1))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(json.dumps(s))
    if s["liquid_left"]:
        print(f"warning: {s['liquid_left']} sites still liquid at the end; raise extra_time_s or lower "
              "pause_below_K", file=sys.stderr)
    if not a.keep:
        for name in ("substrate.sites", "melt.txt", "liquid.txt", "pool.txt"):
            p = os.path.join(a.out, name)
            if os.path.exists(p):
                os.remove(p)
    print(f"wrote {a.out}/track.npz ({time.time() - t0:.0f} s in total)")


if __name__ == "__main__":
    main()
