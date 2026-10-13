"""SpOTr on KP-Tracer spatial data (KP-Spatial Slide-seq, Jones, Sun et al. 2026), built the way the PEtracer
sections were (2026-10-13), for the KP equivalent of figures/petracer_in_action.pdf.

Puck: the highest-capture puck whose tree resolves into usable clades -- at least three of the 20 greedy clades of
44_kpspatial_prep.py holding 5% of the cells or more.  B94_04 (capture 0.722) fails it (one clade holds 89%), so the
puck is M18_13 (capture 0.698, ten such clades); rule written before any mapping was run.
Same 4,000 tumour beads, tree, time matrix and 20 clades as 44_kpspatial_prep.py.

Construction, mirroring PEtracer's: a random half of the beads is "dissociated" (expression + tree, positions hidden);
the other half is pooled onto a square spot lattice whose pitch gives about five beads per occupied spot (PEtracer
M3-4: 5.1), giving spot expression and each spot's true clade counts.  Expression: 2,000 highly variable genes
(normalised dispersion on all 4,000 beads, no labels), counts per 10^4 + log1p, 30 principal components fitted on the
beads; the cost is the squared distance between a bead and a spot in that space.  Tree: n_ik is the number of
dissociated cells under the pair's common ancestor (time matrix Lt, ancestor at least as recent); the FGW arm uses
the divergence (t_i + t_j)/2 - Lt_ij.  The trees are not time-calibrated, which the law does not need: it uses
clone size, not age.
Law: r^2 = L^2 + c^2 n / (2 pi rho), fitted on the OTHER five pucks' true positions (leave one puck out), as
PEtracer's law came from its other sections; the fit's error is reported.
Arms, scored on the spots outside the 20% anchor: expression-only exact OT, Tangram (73_tangram.py settings), FGW
and SpOTr with a shared L on the exact weight path picked by the anchor, SpOTr with the weight and one L per clade
learned (70_spotr.learn).  Writes experiments/kpspatial/kpspotr_<puck>_s<seed>.npz and spotr_out/kp_<puck>_s<seed>.json.
Usage: python 78_kp_spotr.py [puck] [seed]
"""
import sys, json, heapq, pickle, importlib.util, time
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
KP = HERE.parent / "kpspatial"
RAW = HERE.parents[1] / "data" / "kpspatial" / "KPSpatial_Data" / "slideseq"
PUCK = sys.argv[1] if len(sys.argv) > 1 else "M18_13"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
PUCKS = ["B94_04", "M18_13", "M18_17", "M09_07", "M18_06", "M09_17"]
BEADS_PER_SPOT, N_HVG, N_PC, K_LINEAGE = 5.0, 2000, 30, 4


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def greedy_clades(G, root, lc, K):
    """44_kpspatial_prep.greedy_clades, verbatim: split the largest remaining clade until K clades exist"""
    h = [(-lc[root], 0, root)]
    tie = 1
    while len(h) < K:
        s_, _, n = heapq.heappop(h)
        while True:
            ch = [c for c in G.successors(n) if lc[c] > 0]
            if len(ch) != 1:
                break
            n = ch[0]
        if not ch:
            heapq.heappush(h, (0, tie, n)); tie += 1
            if all(x[0] == 0 for x in h):
                break
            continue
        for c in ch:
            heapq.heappush(h, (-lc[c], tie, c)); tie += 1
    return [n for _, _, n in h]


def leaves_of(G, roots, lc, names):
    idx = {nm: i for i, nm in enumerate(names)}
    y = np.full(len(names), -1, int)
    for cid, r in enumerate(roots):
        stack = [r]
        while stack:
            n = stack.pop()
            if lc[n] == 0:
                continue
            ch = [c for c in G.successors(n) if lc[c] > 0]
            if not ch:
                if n in idx:
                    y[idx[n]] = cid
            else:
                stack.extend(ch)
    return y


def law_points(puck, nbins=24):
    """binned (n / 2 pi rho, d^2 / 2) over all pairs of a puck's beads, from their TRUE positions"""
    from scipy.spatial import ConvexHull
    m55 = mod("m55", "55_decomposition_check.py")
    z = np.load(KP / f"kp_{puck}.npz", allow_pickle=True)
    xy = z["xy"].astype(float); n = len(xy); rho = n / ConvexHull(xy).volume
    NC = m55.clone_size(-z["Lt"].astype(float))
    iu = np.triu_indices(n, 1)
    x = NC[iu] / (2 * np.pi * rho)
    d2 = ((xy[iu[0]] - xy[iu[1]]) ** 2).sum(1) / 2.0
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    cnt = np.bincount(b, minlength=len(edges) - 1)
    ok = cnt > 200
    return np.c_[np.bincount(b, x, minlength=len(cnt))[ok] / cnt[ok], np.bincount(b, d2, minlength=len(cnt))[ok] / cnt[ok],
                 cnt[ok]]


def fit_law(points):
    from scipy.optimize import nnls
    P = np.vstack(points); w = np.sqrt(P[:, 2])
    coef, _ = nnls(np.c_[np.ones(len(P)), P[:, 0]] * w[:, None], P[:, 1] * w)
    L, c = float(np.sqrt(coef[0])), float(np.sqrt(coef[1]))
    pred = np.sqrt(coef[0] + coef[1] * P[:, 0]); obs = np.sqrt(P[:, 1])
    return L, c, float(np.median(np.abs(pred - obs) / obs))


def build():
    import anndata as ad, scanpy as sc, networkx as nx
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist, squareform, pdist
    m55 = mod("m55", "55_decomposition_check.py")
    z = np.load(KP / f"kp_{PUCK}.npz", allow_pickle=True)
    names = np.asarray(z["names"]).astype(str); xy = z["xy"].astype(float); n = len(names)
    Lt = z["Lt"].astype(float); tm = z["tm_leaf"].astype(float); y20 = np.asarray(z["y20"])
    # display lineages: the same greedy splitting, stopped at K_LINEAGE (so every one of the 20 clades nests in one)
    t = pickle.load(open(RAW / "trees" / f"{PUCK}_hybrid.pkl", "rb"))
    G = nx.DiGraph(); G.add_edges_from([(u, v) for u, v in t.edges])
    keep = set(names); lc = {v: 0 for v in G.nodes}
    for o in keep:
        for anc in nx.ancestors(G, o) | {o}:
            lc[anc] += 1
    lin_raw = leaves_of(G, greedy_clades(G, t.root, lc, K_LINEAGE), lc, list(names))
    assert all(len(np.unique(lin_raw[y20 == k])) == 1 for k in np.unique(y20)), "a clade straddles two lineages"
    # expression of the same beads
    a = ad.read_h5ad(RAW / "expression" / f"{PUCK}_adata.h5ad")
    a = a[names].copy(); a.var.index = a.var.index.astype(str); a.var_names_make_unique()
    b = a.copy(); sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, flavor="seurat", n_top_genes=N_HVG)     # normalised dispersion, no labels
    counts = a[:, b.var.highly_variable.values].X
    counts = np.asarray(counts.toarray() if hasattr(counts, "toarray") else counts, float)
    lognorm = lambda C: np.log1p(C / np.maximum(C.sum(1, keepdims=True), 1.0) * 1e4)
    # the split
    perm = np.random.default_rng(SEED).permutation(n); use, spot_side = perm[n // 2:], perm[: n // 2]
    # the spot lattice from the spot-side beads, about five beads per occupied spot
    area = ConvexHull(xy).volume
    def lattice(p):
        x0 = xy[spot_side].min(0)
        g = np.floor((xy[spot_side] - x0) / p).astype(int)
        keys, inv = np.unique(g, axis=0, return_inverse=True)
        return keys, inv.ravel(), x0
    pitches = np.linspace(40, 400, 721)
    occ = [np.bincount(lattice(p)[1]).mean() for p in pitches]
    pitch = float(pitches[int(np.argmin(np.abs(np.array(occ) - BEADS_PER_SPOT)))])
    keys, inv, x0 = lattice(pitch)
    m = len(keys); spots = x0 + (keys + 0.5) * pitch
    nb = np.bincount(inv, minlength=m).astype(float)
    S = np.zeros((m, counts.shape[1])); np.add.at(S, inv, counts[spot_side])
    K = int(y20.max()) + 1
    Nspot = np.zeros((m, K)); np.add.at(Nspot, (inv, y20[spot_side]), 1.0)
    # expression space: PCs of the beads (both halves, no labels); cells and spots projected
    Xb = lognorm(counts); mu = Xb.mean(0); sd = np.maximum(Xb.std(0), 1e-6)
    Zb = (Xb - mu) / sd
    V = np.linalg.svd(Zb, full_matrices=False)[2][:N_PC].T
    Pc = Zb[use] @ V; Ps = ((lognorm(S) - mu) / sd) @ V
    M = cdist(Pc, Ps, "sqeuclidean"); M /= M.max()
    # tree quantities among the dissociated cells
    LtA = Lt[np.ix_(use, use)]
    AGE = (tm[use][:, None] + tm[use][None, :]) / 2.0 - LtA; np.fill_diagonal(AGE, 0.0); AGE = np.maximum(AGE, 0.0)
    NC = m55.clone_size(-LtA).astype(float)
    J = np.random.default_rng(1000 * SEED + 20).choice(m, max(6, int(round(0.2 * m))), replace=False)
    # the law from the other five pucks
    pts = [law_points(p) for p in PUCKS if p != PUCK]
    L0, c0, law_err = fit_law(pts)
    L_own, c_own, own_err = fit_law([law_points(PUCK)])          # reported only: the puck's own fit
    X = dict(M=M, D=squareform(pdist(spots)), q=nb / nb.sum(), NC=NC, rho=len(use) / area, cl=y20[use], K=K,
             Nspot=Nspot, spots=spots, xy_true=xy[use], AGE=AGE, J=J, L0=L0, c0=c0, inB=None, tag=f"KP-{PUCK}_s{SEED}")
    extra = dict(pitch=pitch, n_spots=m, beads_per_spot=float(nb.mean()), law=dict(L0=L0, c0=c0, median_rel_err=law_err,
                 own_L=L_own, own_c=c_own, own_median_rel_err=own_err), lineage_raw=lin_raw, use=use, spot_side=spot_side,
                 Xsc_tg=lognorm(counts[use]), Xst_tg=lognorm(S), y20=y20, Lt=Lt, tm=tm, names=names, xy_all=xy)
    return X, extra


def path_job(args):
    X, form, w = args
    m70 = mod("m70", "70_spotr.py")
    return m70.path_point((X, form, w))


if __name__ == "__main__":
    import os, ot
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    m70 = mod("m70", "70_spotr.py"); m73 = mod("m73", "73_tangram.py")
    X, E = build()
    n, m = X["M"].shape
    print(f"{PUCK} seed {SEED}: {n} dissociated cells, {m} spots, pitch {E['pitch']:.0f} um, {E['beads_per_spot']:.1f} beads/spot; "
          f"law from the other pucks L {E['law']['L0']:.0f} um, c {E['law']['c0']:.2f} (median error {100 * E['law']['median_rel_err']:.0f}%); "
          f"own fit L {E['law']['own_L']:.0f}, c {E['law']['own_c']:.2f} ({100 * E['law']['own_median_rel_err']:.0f}%)  [{time.time() - t0:.0f}s]", flush=True)
    p = np.ones(n) / n
    T = {"expression": ot.emd(p, X["q"], X["M"], numItermax=10 ** 7)}
    f_tg = HERE / "spotr_out" / f"T_KP-{PUCK}_s{SEED}_tangram.npy"
    if f_tg.exists():
        T["tangram"] = np.load(f_tg).astype(float)
    else:
        T["tangram"] = m73.tangram(E["Xsc_tg"], E["Xst_tg"], X["q"], SEED); np.save(f_tg, T["tangram"].astype(np.float32))
    print(f"expression-only and Tangram done [{time.time() - t0:.0f}s]", flush=True)
    jobs = [(X, form, w) for form in ["fgw", "law"] for w in m70.WGRID[1:]]
    with get_context("spawn").Pool(min(24, len(jobs))) as pool:
        rows = pool.map(path_job, jobs)
    paths = {form: [r for r in rows if r["form"] == form] for form in ["fgw", "law"]}
    picks = {form: min(paths[form], key=lambda r: r["nll_val"]) for form in paths}
    print(f"paths done: FGW picks w {picks['fgw']['w']} ({picks['fgw']['err']:.0f} um), shared L picks w {picks['law']['w']} "
          f"({picks['law']['err']:.0f} um) [{time.time() - t0:.0f}s]", flush=True)
    log, best = m70.learn(X, X["cl"].copy(), learn_a=True, a0=0.197 / 1.197)
    T["spotr"] = best["T"]
    print(f"per-clade learner: {best['err']:.0f} um at step {best['step']}, alpha {best['a']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    np.save(HERE / "spotr_out" / f"T_KP-{PUCK}_s{SEED}_per-clade.npy", T["spotr"].astype(np.float32))
    # scores on the spots outside the anchor, at the clade and the lineage level
    lin_ids = {v: i for i, v in enumerate(pd.Series(E["lineage_raw"]).value_counts().index)}
    lin = np.array([lin_ids[v] for v in E["lineage_raw"]]); lin3 = np.where(lin < 3, lin, 3)
    clade_lin = np.array([lin3[E["y20"] == k][0] for k in range(X["K"])])
    has = X["Nspot"].sum(1) > 0; held = np.setdiff1d(np.where(has)[0], X["J"])
    true_cl = X["Nspot"].argmax(1); true_lin = (X["Nspot"] @ np.eye(4)[clade_lin]).argmax(1)
    Yc = np.eye(X["K"])[X["cl"]]; Yl = np.eye(4)[lin3[E["use"]]]
    res = {}
    for arm, TT in [("expression", T["expression"]), ("tangram", T["tangram"]), ("spotr", T["spotr"])]:
        res[arm] = dict(err=m70.err(TT, X), clade=float(np.mean((TT.T @ Yc).argmax(1)[held] == true_cl[held])),
                        lineage=float(np.mean((TT.T @ Yl).argmax(1)[held] == true_lin[held])))
    for form in ["fgw", "law"]:
        res["fgw" if form == "fgw" else "shared_L"] = dict(err=picks[form]["err"], w=picks[form]["w"])
    res["largest"] = dict(clade=float(np.mean(true_cl[held] == np.bincount(X["cl"]).argmax())),
                          lineage=float(np.mean(true_lin[held] == np.bincount(true_lin[has], minlength=4).argmax())))
    out = dict(puck=PUCK, seed=SEED, cells=int(n), spots=int(m), held_out=int(len(held)), pitch=E["pitch"], law=E["law"],
               lineage_cells=np.bincount(lin3[E["use"]], minlength=4).tolist(), K=int(X["K"]), alpha=best["a"], L=best["L"],
               results=res)
    json.dump(out, open(HERE / "spotr_out" / f"kp_{PUCK}_s{SEED}.json", "w"), indent=1, default=float)
    np.savez_compressed(KP / f"kpspotr_{PUCK}_s{SEED}.npz", spots=X["spots"], Nspot=X["Nspot"], J=X["J"], q=X["q"], pitch=E["pitch"],
                        cl=X["cl"], lin3=lin3, use=E["use"], Lt=E["Lt"], tm=E["tm"], xy_all=E["xy_all"], clade_lin=clade_lin,
                        Xst_tg=E["Xst_tg"], T_expression=T["expression"].astype(np.float32))
    print(json.dumps(res, indent=1), f"\n[{time.time() - t0:.0f}s]")
