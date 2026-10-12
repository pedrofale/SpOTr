"""ANCHOR-ONLY, STEP 1 (2026-10-12): does an anchor loss pick the accuracy optimum along the FGW weight path?

The anchor emulates a partial in-situ lineage readout: a random fraction of SPOTS reveal their clade
composition (the clades of the spot-side cells they hold).  The mapping never sees the anchor; the anchor
only scores couplings.  The last time this was asked (PEtracer, exact clade composition as anchor) it
picked alpha 0.1 against an accuracy optimum of 0.6 -- on the entropic solver whose clipped kernel (D1)
manufactured that 0.6, so the question is open.

Anchor scores for a coupling T, over the labelled spots J only:
  xent     multinomial log-likelihood of the observed clade counts n_jk under the composition the mapping
           predicts, pi_jk = (sum_{i in k} T_ij + a * prior_k) / (sum_i T_ij + a), prior from the
           mapped cells' clade frequencies, pseudo-count a = 1 cell
  density  clade-density agreement (eTRACER's metric): per clade, Pearson over J between predicted and
           observed density, averaged weighted by clade abundance
Exact FGW (55_'s solver, W = 1, Phi = patristic), all genes in the cost, observed spot marginal,
seeds 0 and 1 per section = 6 splits; coverages 50/20/10/5/2% of spots, 5 anchor draws each.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, treedata as td

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m55", HERE / "55_decomposition_check.py")
m55 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m55)
COVS = [0.5, 0.2, 0.1, 0.05, 0.02]
NDRAW = 5
_O = {}


def clade_of():
    if "o" not in _O:
        t = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td")
        _O["o"] = t.obs["clade"].astype(str)
    return _O["o"]


def scores(T, cl_map, Nspot, J):
    """cl_map: clade index of each mapped cell; Nspot: (m, K) observed clade counts of spot cells."""
    K = Nspot.shape[1]
    pred = np.vstack([T[cl_map == k].sum(0) for k in range(K)]).T          # (m, K) predicted mass
    prior = np.bincount(cl_map, minlength=K) / len(cl_map)
    pj = pred[J]; mass = pj.sum(1, keepdims=True)
    scale = Nspot[J].sum(1, keepdims=True) / np.maximum(mass, 1e-300)      # mass -> expected cells
    pi = (pj * scale + prior[None, :]) / (Nspot[J].sum(1, keepdims=True) + 1.0)
    xent = float((Nspot[J] * np.log(pi + 1e-300)).sum() / Nspot[J].sum())
    rs, ws = [], []
    for k in range(K):
        tru = Nspot[J, k]; prd = pj[:, k]
        if tru.sum() < 5 or tru.std() == 0 or prd.std() == 0:
            continue
        rs.append(np.corrcoef(prd, tru)[0, 1]); ws.append(tru.sum())
    dens = float(np.average(rs, weights=ws)) if rs else np.nan
    return xent, dens


def job(args):
    sec, seed, w = args
    o = clade_of()
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    S = m55.load(sec)
    n = len(S["AGE"]); use = np.random.default_rng(seed).permutation(n)[n // 2:]
    cl_names = sorted(set(o.loc[np.asarray(z["cell_names"])]) | set(o.loc[np.asarray(z["spot_cell_names"])]))
    ci = {c: i for i, c in enumerate(cl_names)}; K = len(cl_names)
    cl_map = np.array([ci[o.loc[c]] for c in np.asarray(z["cell_names"])[use]])
    scells = np.asarray(z["spot_cell_names"]); sof = np.asarray(z["spot_of_cell"]); m = S["M"].shape[1]
    Nspot = np.zeros((m, K))
    for c, j in zip(scells, sof):
        Nspot[j, ci[o.loc[c]]] += 1
    AGE = S["AGE"][np.ix_(use, use)]; M = S["M"][use]; D = S["D"]; q = S["q"]
    p = np.ones(len(use)) / len(use)
    Wm = np.ones_like(AGE); np.fill_diagonal(Wm, 0.0); WPhi = Wm * 2.0 * AGE
    T0 = np.outer(p, q); g0 = 2.0 * (Wm @ T0 @ D ** 2 - 2.0 * WPhi @ T0 @ D)
    sc = m55.dc_std(M) / max(m55.dc_std(g0), 1e-300)
    T = m55.solve(M, Wm * sc, WPhi * sc, D, D ** 2, p, q, w / (1 + w))
    am = T.argmax(1)
    err = float(np.median(np.linalg.norm(S["xy_st"][am] - S["xy_sc"][use], axis=1)))
    rows = []
    for cov in COVS:
        for r in range(NDRAW):
            J = np.random.default_rng(1000 * seed + 10 * r + int(cov * 100)).choice(m, max(3, int(round(cov * m))), replace=False)
            xent, dens = scores(T, cl_map, Nspot, J)
            rows.append(dict(sec=sec, seed=seed, w=w, err=err, cov=cov, draw=r, xent=xent, dens=dens))
    return rows


if __name__ == "__main__":
    jobs = [(s, sd, float(w)) for s in m55.SECS for sd in (0, 1) for w in m55.WGRID]
    out = []
    with Pool(26) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            out += r
    df = pd.DataFrame(out)
    df.to_csv(HERE / "anchor_alignment.csv", index=False)
    print("share of the tree's gain over expression-only recovered at the anchor's pick; error (um) at the pick\n")
    res = []
    for (cov, crit), _ in [((c, k), None) for c in COVS for k in ("xent", "dens")]:
        e0s, ebs, eps, w0 = [], [], [], 0
        for (sec, seed, r), g in df[df["cov"] == cov].groupby(["sec", "seed", "draw"]):
            g = g.sort_values("w"); pick = g.loc[g[crit].idxmax()]
            e0s.append(g[g.w == 0].err.iloc[0]); ebs.append(g.err.min()); eps.append(pick.err); w0 += pick.w == 0
        e0s, ebs, eps = map(np.array, (e0s, ebs, eps))
        res.append(dict(coverage=f"{int(cov*100)}%", criterion=crit, share_of_gain=(e0s - eps).sum() / (e0s - ebs).sum(),
                        err_pick=eps.mean(), err_opt=ebs.mean(), err_expr=e0s.mean(), picks_no_tree=f"{w0}/{len(eps)}"))
    print(pd.DataFrame(res).round(3).to_string(index=False))
