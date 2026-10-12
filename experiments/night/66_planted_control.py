"""PLANTED POSITIVE CONTROL (2026-10-13): can the anchor loss recover a per-group mixing length when one exists?

PEtracer M3-4, the clone-map split (seed 0).  Half of the clades (group B, chosen at random, ~half the cells)
get extra dispersal: every group-B cell -- dissociated and spot side -- is displaced by N(0, sigma^2 I).
Spots are rebuilt on the original lattice from the moved spot-side cells (nearest centre; spot expression =
mean of members' log-normalised expression), so expression and tree stay real and only the mixing is planted.
Planting adds sigma^2 to r(a)^2 for group-B pairs, so the target is L_B ~ sqrt(L^2 + sigma^2).

The tree term is the law per pair with TWO free mixing lengths: within-clade pairs of group A use L_A, of
group B use L_B; cross-clade pairs keep the shared 305 um; c = 1.36; tree weight fixed at w = 0.197 (the
leave-one-section-out choice for the law arms in 55_).  An 8 x 8 grid over (L_A, L_B), exact solver, no
gradients.  Scored by the anchor loss (20% and 5% of spots, 5 draws each) and, for evaluation only, by
placement error against the moved true positions.  The "measured" (L_A, L_B) is each group's mixing length
computed from the moved true positions.
"""
import os, sys, json, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "planted"
SEC, SEED, W_TREE, L0, C0 = "M3-4", 0, 0.197, 305.0, 1.36
GRID = [150., 200., 250., 305., 375., 450., 550., 700.]
SIGMAS = [0.0, 300.0, 500.0]
GROUP_DRAWS = [0, 1]
_C = {}


def base():
    if "b" not in _C:
        import treedata as td
        t = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td")
        o = t.obs.copy(); X = t.X.tocsr().astype(float); lib = np.asarray(X.sum(1)).ravel()
        sp = np.asarray(t.obsm["spatial"], float); o["sx"], o["sy"] = sp[:, 0], sp[:, 1]   # frame of the prepared files
        _C["b"] = (o, X, lib, {c: i for i, c in enumerate(o.index)})
    return _C["b"]


def logexpr(cells):
    o, X, lib, ix = base(); r = np.array([ix[c] for c in cells])
    return np.log1p(X[r].multiply(1e4 / np.maximum(lib[r], 1)[:, None]).toarray())


def build(sigma, gdraw):
    key = (sigma, gdraw)
    if key in _C:
        return _C[key]
    spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
    m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
    o, X, lib, ix = base()
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    allc = np.asarray(z["cell_names"]); n_all = len(allc)
    use = np.random.default_rng(SEED).permutation(n_all)[n_all // 2:]
    cells = allc[use]; scells = np.asarray(z["spot_cell_names"]); centres = np.asarray(z["xy"], float)
    AGE = np.asarray(z["C_tree"], float)[np.ix_(use, use)] / 2.0
    clade = o["clade"].astype(str)
    names = sorted(set(clade.loc[cells]) | set(clade.loc[scells])); ci = {c: i for i, c in enumerate(names)}; K = len(names)
    cl = np.array([ci[clade.loc[c]] for c in cells]); cls = np.array([ci[clade.loc[c]] for c in scells])
    # group B: random clades until half the (dissociated) cells
    rng = np.random.default_rng(700 + gdraw); order = rng.permutation(K); sizes = np.bincount(cl, minlength=K)
    inB = np.zeros(K, bool); tot = 0
    for k in order:
        if tot < sizes.sum() / 2:
            inB[k] = True; tot += sizes[k]
    # true positions (centroids), then plant dispersal on group B
    off = (o.loc[allc, ["sx", "sy"]].values - np.asarray(z["xy_sc"], float)).mean(0)   # exact translation
    xy_d = o.loc[cells, ["sx", "sy"]].values.astype(float) - off
    xy_s = o.loc[scells, ["sx", "sy"]].values.astype(float) - off
    prng = np.random.default_rng(900 + gdraw)
    xy_d = xy_d + (inB[cl][:, None] * prng.normal(0, sigma, xy_d.shape) if sigma > 0 else 0)
    xy_s = xy_s + (inB[cls][:, None] * prng.normal(0, sigma, xy_s.shape) if sigma > 0 else 0)
    # rebuild spots on the original lattice
    from scipy.spatial import cKDTree, ConvexHull
    from scipy.spatial.distance import pdist, squareform
    import ot
    sof = cKDTree(centres).query(xy_s)[1]
    keep = np.unique(sof); remap = {j: i for i, j in enumerate(keep)}; sof = np.array([remap[j] for j in sof])
    centres = centres[keep]; m = len(keep)
    Es = logexpr(scells); Xst = np.vstack([Es[sof == j].mean(0) for j in range(m)])
    Xsc = logexpr(cells)
    M = ot.dist(Xsc, Xst); M /= M.max()
    Nspot = np.zeros((m, K))
    np.add.at(Nspot, (sof, cls), 1.0)
    D = squareform(pdist(centres)); q = np.bincount(sof, minlength=m).astype(float); q /= q.sum()
    NC = m55.clone_size(AGE).astype(float); rho = len(cells) / ConvexHull(centres).volume
    same = cl[:, None] == cl[None, :]
    grp = np.where(same, np.where(inB[cl][:, None], 2, 1), 0)          # 0 shared, 1 within-A, 2 within-B
    # measured group mixing lengths from the (moved) true positions
    meas = {}
    for g, lab in [(1, "A"), (2, "B")]:
        v = []
        for k in np.where(inB == (g == 2))[0]:
            idx = np.where(cl == k)[0]
            if len(idx) < 30:
                continue
            iu = np.triu_indices(len(idx), 1)
            d2 = ((xy_d[idx][:, None] - xy_d[idx][None]) ** 2).sum(-1)[iu]
            pk = NC[np.ix_(idx, idx)][iu] / (2 * np.pi * rho)
            v.append((np.sum(d2 / 2 - C0 * C0 * pk), len(iu[0])))
        meas[lab] = float(np.sqrt(max(sum(a for a, _ in v) / sum(b for _, b in v), 1.0)))
    anchors = {cov: [np.random.default_rng(1000 * r + int(cov * 100)).choice(m, max(4, int(round(cov * m))), replace=False)
                     for r in range(5)] for cov in (0.2, 0.05)}
    _C[key] = dict(m55=m55, M=M, D=D, q=q, NC=NC, rho=rho, grp=grp, cl=cl, K=K, Nspot=Nspot, xy_true=xy_d,
                   centres=centres, anchors=anchors, meas=meas, inB=inB, nB=int(tot), n=len(cells), m=m)
    return _C[key]


def anchor_nll(T, Y, Nspot, J):
    pred = (T.T @ Y)[J]; nj = Nspot[J].sum(1, keepdims=True); prior = Y.mean(0)
    pi = (pred * (nj / np.maximum(pred.sum(1, keepdims=True), 1e-300)) + prior) / (nj + 1.0)
    return float(-(Nspot[J] * np.log(np.maximum(pi, 1e-300))).sum() / Nspot[J].sum())


def job(args):
    sigma, gdraw, LA, LB = args
    B = build(sigma, gdraw); m55 = B["m55"]
    Lpair = np.choose(B["grp"], [L0, LA, LB]).astype(float)
    R2 = Lpair ** 2 + C0 * C0 * B["NC"] / (2 * np.pi * B["rho"])
    W = 2.0 / ((4 - np.pi) * R2); np.fill_diagonal(W, 0.0); WPhi = W * np.sqrt(np.pi / 2 * R2)
    p = np.ones(B["n"]) / B["n"]; T0 = np.outer(p, B["q"])
    g0 = 2.0 * (W @ T0 @ B["D"] ** 2 - 2.0 * WPhi @ T0 @ B["D"]); sc = m55.dc_std(B["M"]) / m55.dc_std(g0)
    T = m55.solve(B["M"], W * sc, WPhi * sc, B["D"], B["D"] ** 2, p, B["q"], W_TREE / (1 + W_TREE))
    err = float(np.median(np.linalg.norm(B["centres"][T.argmax(1)] - B["xy_true"], axis=1)))
    Y = np.eye(B["K"])[B["cl"]]
    row = dict(sigma=sigma, gdraw=gdraw, LA=LA, LB=LB, err=err, measA=B["meas"]["A"], measB=B["meas"]["B"],
               nB=B["nB"], n=B["n"], m=B["m"])
    for cov, Js in B["anchors"].items():
        for r, J in enumerate(Js):
            row[f"nll_{int(cov*100)}_{r}"] = anchor_nll(T, Y, B["Nspot"], J)
    return row


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    OUT.mkdir(parents=True, exist_ok=True)
    if "--check" in sys.argv:
        print(job((0.0, 0, L0, L0)))
        sys.exit()
    jobs = [(s, g, a, b) for s in SIGMAS for g in GROUP_DRAWS for a in GRID for b in GRID]
    with get_context("fork").Pool(int(os.environ.get("PROCS", 28))) as pool:
        rows = pool.map(job, jobs, chunksize=4)
    df = pd.DataFrame(rows); df.to_csv(OUT / "planted_grid.csv", index=False)
    for cov in (20, 5):
        cols = [c for c in df.columns if c.startswith(f"nll_{cov}_")]
        df[f"nll_{cov}"] = df[cols].mean(1)
    print("argmin of the anchor loss vs measured group mixing lengths (um)\n")
    for (s, g), h in df.groupby(["sigma", "gdraw"]):
        b = h.loc[h.err.idxmin()]
        line = f"sigma {s:>5.0f}  draw {g}: measured A {h.measA.iloc[0]:5.0f}  B {h.measB.iloc[0]:5.0f}  | best placement A {b.LA:4.0f} B {b.LB:4.0f} ({b.err:.0f} um)"
        for cov in (20, 5):
            a = h.loc[h[f"nll_{cov}"].idxmin()]
            # per anchor draw
            picks = [h.loc[h[f"nll_{cov}_{r}"].idxmin(), ["LA", "LB"]].values for r in range(5)]
            ratio = np.mean([np.log(p[1] / p[0]) for p in picks])
            line += f" | anchor {cov}%: A {a.LA:4.0f} B {a.LB:4.0f} (err {a.err:.0f}); mean log(LB/LA) over draws {ratio:+.2f}"
        print(line)
