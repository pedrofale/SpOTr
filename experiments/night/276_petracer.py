"""PEtracer with ground truth: SpOTr v1 on DestVI and Tangram, lambda by the anchored lineage cross-validation.  2026-10-28.
FIXED BEFORE RUNNING.
Pedro, 2026-10-28: "Ok so now we need to run on real data, right? ... we also need to include, like TemSOMap, data without ground
truth ... we need to make sure our paper is better"; on the ground-truth design: "Both (Recommended)" (decision-real-data-plan).
Data: Koblan et al. 2025 (PEtracer), mouse 4T1 lung metastasis, MERFISH (175 genes) with lineages read in situ:
/home/pedro/.treedata/datasets/koblan25_tumor.h5td -- 145,954 cells in sections M3-1..M3-4, 18 annotated subtypes (Malignant and 17
others), one time-calibrated tree over 10,061 traced malignant cells (ultrametric, leaves at time 35) with 18 clades; M3-3 has no
traced cell and is not used; doublets are removed.  Raw integer counts (median 169 per cell).
prepare() caches each section to SpOTr/data/petracer/petracer_<section>.npz (counts, positions, subtype, malignant / traced, clade,
and the tree distance 2 (35 - t_lca) between its traced cells).
Designs (build()):
  half-visium:<section>  as half, but the section's half measured as by Visium: spots of 55 um on a hexagonal lattice 100 um apart, the
                      cells between spots not measured (decision-visium-like-simulation, extended to PEtracer on 2026-11-02).
  half:<section>      the section's cells split at random in two halves (stream: seed); one half pooled into square spots of
                      PITCH = 55 um (the Visium spot diameter; about 10-15 cells per spot at this density), the dissociated sample
                      drawn uniformly from the other half (N_DISS = 8,000 cells, all types, so tracing and type proportions are kept);
                      xy_true = the dissociated cells' own positions (the floor).  Sections M3-1, M3-2, M3-4; seed 0.
  serial:M3-1>M3-2    the dissociated sample drawn uniformly from all of M3-1 (8,000 cells), the spots pooled from all of M3-2
                      (serial sections of one block; truly different cells); clades are shared (one tree); no floor (positions
                      are not registered across sections).
Kinds: traced malignant (DestVI label: its clade, 'c18' for the 567 traced cells without a clade; K = 19), untraced malignant
('untraced'), and each of the 17 other subtypes ('me<e>'; nT = 17).  Truth: each spot's clade composition among its own traced cells
(about one in six malignant cells is traced), scored on spots with at least one; the tumour share against all malignant cells.
Method and readouts: 275_real_runner (SpOTr v1, lineage tree, lambda {0, 0.1, 0.3, 0.5, 1, 2.5} with continuation, 4-fold anchored
cross-validation, argmax); comparators DestVI alone, Tangram alone, exact OT, floor, global mix.
Expectations stated before running:
  E128  half design (3 sections x 2 tools): SpOTr at the cross-validated lambda has a lower per-spot subclone error than the tool
        alone in at least 5 of 6
  E129  half design: ... and lower than SpOTr without the tree (lambda 0) in at least 5 of 6 (the tree helps on real data)
  E130  serial design (2 tools): SpOTr at the cross-validated lambda beats the tool alone in 2 of 2
  read: the 3 x 3 window error; the cross-validated lambda against the truth-best; the floor; the microenvironment-type and
        tumour-share errors (the tool's, by construction).
One job per problem (about 30 minutes), saved to spotr_out/petracer/<problem>.json, fold maps to spotr_out/petracer/maps/; resumable.
Usage: python 276_petracer.py PREPARE | RUN | SUMMARY
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "petracer"; DATA = HERE.parents[1] / "data" / "petracer"
H5TD = Path("/home/pedro/.treedata/datasets/koblan25_tumor.h5td"); SECTIONS = ["M3-1", "M3-2", "M3-4"]; T_LEAF = 35.0
PITCH, N_DISS = 55.0, 8000
VISIUM_D, VISIUM_A = 55.0, 100.0           # half-visium: Visium's spot diameter and centre-to-centre spacing (Pedro, 2026-11-02)
PROBLEMS = [("half:M3-4", 0), ("half:M3-1", 0), ("half:M3-2", 0), ("serial:M3-1>M3-2", 0)]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def prepare():
    import treedata as td, scipy.sparse as sp
    t = td.read_h5td(H5TD); o = t.obs; X = t.X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    sub = o["cell_subtype"].astype(str).values; others = sorted(set(sub) - {"Malignant"}); clades = sorted({c for c in o["clade"].astype(str) if c != "nan"}, key=int)
    tree = t.obst["tree"]; nodes = list(tree.nodes); nid = {v: i for i, v in enumerate(nodes)}; ntime = np.array([tree.nodes[v]["time"] for v in nodes])
    parent = {v: u for u, v in tree.edges}
    def root_path(v):
        p = [nid[v]]
        while v in parent:
            v = parent[v]; p.append(nid[v])
        return p[::-1]
    DATA.mkdir(parents=True, exist_ok=True)
    for s in SECTIONS:
        m = ((o["sample"].astype(str) == s) & (o["type"].astype(str) != "doublet")).values; idx = np.where(m)[0]
        traced = (o["type"].astype(str).values[idx] == "tracing"); cl = o["clade"].astype(str).values[idx]
        clade = np.array([clades.index(c) if c != "nan" else len(clades) for c in cl]); clade[~traced] = -1
        names = np.asarray(t.obs_names[idx]); tn = names[traced]; paths = [root_path(v) for v in tn]; L = max(map(len, paths))
        Pm = np.full((len(paths), L), -1, np.int64)
        for i, p in enumerate(paths):
            Pm[i, :len(p)] = p; Pm[i, len(p):] = -(i + 2)                 # unique padding: never matches another row
        d = np.zeros((len(tn), len(tn)), np.float32)
        for i in range(len(tn)):
            eq = np.cumprod(Pm == Pm[i], axis=1); k = eq.sum(1); d[i] = 2.0 * (T_LEAF - ntime[Pm[i, np.maximum(k - 1, 0)]])
        np.fill_diagonal(d, 0.0)
        np.savez_compressed(DATA / f"petracer_{s}.npz", counts=X[idx].astype(np.int32), xy=np.asarray(t.obsm["spatial"])[idx].astype(np.float32),
                            subtype=np.array([others.index(v) if v != "Malignant" else -1 for v in sub[idx]]), malignant=(sub[idx] == "Malignant"), traced=traced,
                            clade=clade, names=names, d=d, others=np.array(others), clades=np.array(clades), genes=np.asarray(t.var_names))
        print(f"{s}: {len(idx)} cells, {traced.sum()} traced, tree distances {d[~np.eye(len(d), dtype=bool)].min():.1f}-{d.max():.1f}", flush=True)


def spots_from(z, cells, visium=False):
    """square bins of PITCH tiling the section (every cell measured); or, visium=True, spots of VISIUM_D on a hexagonal lattice VISIUM_A apart
    holding the cells whose centres lie in them (the cells between spots not measured; spots without a cell dropped)"""
    if visium:
        from scipy.spatial import cKDTree
        xy = z["xy"][cells].astype(float); lo, hi = xy.min(0), xy.max(0); a = VISIUM_A; rows = np.arange(lo[1] + a / 4, hi[1], a * np.sqrt(3) / 2)
        grid = np.array([(x, y) for i, y in enumerate(rows) for x in np.arange(lo[0] + a / 4 + (a / 2) * (i % 2), hi[0], a)])
        dist, near = cKDTree(grid).query(xy); hit = dist <= VISIUM_D / 2; cells = np.asarray(cells)[hit]
        used, inv = np.unique(near[hit], return_inverse=True); inv = inv.ravel(); centres = grid[used]
    else:
        x0 = z["xy"][cells].min(0); g = np.floor((z["xy"][cells] - x0) / PITCH).astype(int); keys, inv = np.unique(g, axis=0, return_inverse=True); inv = inv.ravel()
        centres = x0 + (keys + 0.5) * PITCH
    M = len(centres); Y = np.zeros((M, z["counts"].shape[1])); np.add.at(Y, inv, z["counts"][cells])
    K, nT = len(z["clades"]) + 1, len(z["others"]); tr = z["traced"][cells]; mal = z["malignant"][cells]
    Ncl = np.zeros((M, K)); np.add.at(Ncl, (inv[tr], z["clade"][cells][tr]), 1.0)
    Nme = np.zeros((M, nT)); np.add.at(Nme, (inv[~mal], z["subtype"][cells][~mal]), 1.0)
    return dict(Y=Y, spots=centres, ncell=np.bincount(inv, minlength=M).astype(float), Ncl=Ncl, Nme=Nme,
                Ntum=np.bincount(inv[mal], minlength=M).astype(float))


def build(problem, seed):
    from scipy.spatial.distance import pdist, squareform
    import ot
    m239 = mod("m239", "239_deconv_fgw.py"); core = mod("core", "250_spotr_core.py")
    design, spec = problem.split(":"); rng = np.random.default_rng(seed)
    if design.startswith("half"):
        zs = zd = dict(np.load(DATA / f"petracer_{spec}.npz", allow_pickle=True)); n_all = len(zs["xy"]); perm = rng.permutation(n_all)
        side, pool = perm[: n_all // 2], perm[n_all // 2:]
    else:
        a, b = spec.split(">"); zd = dict(np.load(DATA / f"petracer_{a}.npz", allow_pickle=True)); zs = dict(np.load(DATA / f"petracer_{b}.npz", allow_pickle=True))
        side, pool = np.arange(len(zs["xy"])), np.arange(len(zd["xy"]))
    diss = np.sort(rng.choice(pool, min(N_DISS, len(pool)), replace=False)); S = spots_from(zs, side, visium=design == "half-visium")
    mal, traced = zd["malignant"][diss], zd["traced"][diss]; K, nT = len(zd["clades"]) + 1, len(zd["others"])
    cl = np.where(traced, zd["clade"][diss], 0); me_type = np.where(mal, 0, zd["subtype"][diss])
    labels = np.array([f"c{c}" if t else ("untraced" if mm else f"me{e}") for c, t, mm, e in zip(cl, traced, mal, me_type)])
    Xc = zd["counts"][diss].astype(float); kind = np.where(mal, 0, 1)
    ti_all = np.where(zd["traced"])[0]; pos = {c: i for i, c in enumerate(ti_all)}; tsel = np.array([pos[c] for c in diss[mal & traced]])
    d_raw = zd["d"][np.ix_(tsel, tsel)].astype(float)
    xy_true = zd["xy"][diss].astype(float) if design.startswith("half") else np.full((len(diss), 2), np.nan)
    Xsc, Xst = core.lognorm(Xc), core.lognorm(S["Y"]); Mc = ot.dist(Xsc, Xst); Mc /= Mc.max()
    X = dict(kind=kind, traced=traced, cl=cl, me_type=me_type, K=K, nT=nT, spots=S["spots"], pitch=VISIUM_A if design == "half-visium" else PITCH, D=squareform(pdist(S["spots"])),
             q=S["ncell"] / S["ncell"].sum(), M=Mc, Ncl=S["Ncl"], Nme=S["Nme"], Ntum=S["Ntum"], ncell=S["ncell"], xy_true=xy_true, Xsc=Xsc, Xst=Xst)
    meta = dict(problem=problem, seed=seed, cells=int(len(diss)), traced=int((mal & traced).sum()), untraced=int((mal & ~traced).sum()), me=int((~mal).sum()),
                spots=int(len(S["Y"])), cells_per_spot=float(S["ncell"].mean()), spots_with_traced=int((S["Ncl"].sum(1) > 0).sum()),
                clades=[str(c) for c in zd["clades"]], others=[str(e) for e in zd["others"]])
    return dict(X=X, Xc=Xc, Y=S["Y"], labels=labels, d_raw=d_raw, seed=seed, meta=meta)


def summary():
    rs = {f.stem: json.load(open(f)) for f in sorted(list(OUT.glob("half_*.json")) + list(OUT.glob("serial_*.json")))}
    for k, r in rs.items():
        m = r["meta"]; print(f"######## {k}: {m['cells']} cells ({m['traced']} traced), {m['spots']} spots ({m['cells_per_spot']:.1f} cells each, {m['spots_with_traced']} with traced)")
        for a in ["DestVI alone", "Tangram alone", "exact OT", "global mix", "floor"]:
            if a in r["arms"]:
                v = r["arms"][a]; print(f"   {a:14s}: subclone {v['clone_tv']:.3f}  window {v['clone_tv_win']:.3f}")
        for t, v in r["cv"].items():
            fu = v["full"]; L = str(v["argmax"])
            print(f"   SpOTr-{t:8s}: CV lambda {v['argmax']:g} -> subclone {fu[L]['clone_tv']:.3f} window {fu[L]['clone_tv_win']:.3f} | lambda 0 {fu['0.0']['clone_tv']:.3f} | "
                  f"0.5 {fu['0.5']['clone_tv']:.3f} | truth-best {v['truth_best']:g} {fu[str(v['truth_best'])]['clone_tv']:.3f} | type error {fu[L].get('me_type_tv', np.nan):.3f} tumour share {fu[L].get('tumour_share_err', np.nan):.3f}")
    alone = lambda r, t: r["arms"]["DestVI alone" if t == "destvi" else "Tangram alone"]["clone_tv"]
    cvv = lambda r, t: r["cv"][t]["full"][str(r["cv"][t]["argmax"])]["clone_tv"]
    half = [r for k, r in rs.items() if k.startswith("half")]; ser = [r for k, r in rs.items() if k.startswith("serial")]
    e128 = [cvv(r, t) < alone(r, t) for r in half for t in r["cv"]]; e129 = [cvv(r, t) < r["cv"][t]["full"]["0.0"]["clone_tv"] for r in half for t in r["cv"]]
    e130 = [cvv(r, t) < alone(r, t) for r in ser for t in r["cv"]]
    print(f"E128: {sum(e128)}/{len(e128)}   E129: {sum(e129)}/{len(e129)}   E130: {sum(e130)}/{len(e130)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "RUN"
    if cmd == "PREPARE":
        prepare(); sys.exit()
    if cmd == "SUMMARY":
        summary(); sys.exit()
    OUT.mkdir(parents=True, exist_ok=True); runner = mod("runner", "275_real_runner.py"); t0 = time.time()
    for prob, seed in PROBLEMS:
        f = OUT / (prob.replace(":", "_").replace(">", "_to_") + ".json")
        if f.exists() and json.load(open(f)).get("done"):
            continue
        P = build(prob, seed); print(f"{prob}: {P['meta']['cells']} cells ({P['meta']['traced']} traced), {P['meta']['spots']} spots, {P['meta']['cells_per_spot']:.1f} cells/spot", flush=True)
        runner.run(P, f, OUT / "maps"); print(f"{prob}: done ({time.time() - t0:.0f}s total)", flush=True)
    summary()
