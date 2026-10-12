"""SpOTr on eTRACER (2026-10-13): the genuinely unmatched pair -- dissociated scRNA-seq with per-clone trees, and
Stereo-seq bins aggregated to 512 k-d spots (21_etracer_prep.py), from the same tumours.

eTRACER's trees are per clone and not time-calibrated (ML branch lengths; clones joined at a constant), so the
growth law's clone-size term does not apply.  The tree term keeps the law's form with c = 0: within clone k a pair
has spread L_k, across clones the shared L.  The anchor is the clone composition of the barcoded bins in 20% of
the spots (half fit, half validation); accuracy is clone-density agreement (22_'s metric) on the OTHER spots.
L is initialised from the anchor itself (clone spread over the fit spots), so no held-out information enters.

Arms: expression only; FGW on the exact weight path, weight picked by the anchor (validation spots);
SpOTr with the tree weight learned; SpOTr with the weight and one L per clone learned.
Caveat carried from 21_/prep_stage.py: the dissociated cells were sampled to match the tissue's clone mix using
the barcoded bins (an answer-key leak in the cell selection, not in the mapping).
Usage: python 71_etracer.py
"""
import os, sys, json, time, importlib.util
from pathlib import Path
from multiprocessing import get_context
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m70", HERE / "70_spotr.py")
m70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m70)


def data_etracer(tag, seed, cov=0.2):
    import ot
    from scipy.spatial.distance import pdist, squareform
    d = np.load(HERE / f"etracer_spots_{tag}.npz", allow_pickle=True)
    comp = d["spot_comp"].astype(float); xy = d["xy_st"].astype(float); m = len(xy)
    clones = d["clones"].astype(str); cellc = d["cell_clone"].astype(str); ci = {c: i for i, c in enumerate(clones)}
    cl = np.array([ci[c] for c in cellc]); K = len(clones)
    M = ot.dist(d["X_sc"].astype(float), d["X_st"].astype(float)); M /= M.max()
    nb = d["n_bins"].astype(float); q = nb / nb.sum()
    J = np.random.default_rng(1000 * seed + int(cov * 100)).choice(m, int(round(cov * m)), replace=False)
    Jf = J[: len(J) // 2]
    # initial L from the anchor: RMS spread of each clone's barcoded bins over the fit spots, pooled
    num = den = 0.0
    for k in range(K):
        w = comp[Jf, k]
        if w.sum() < 5:
            continue
        mu = (w[:, None] * xy[Jf]).sum(0) / w.sum()
        num += (w * ((xy[Jf] - mu) ** 2).sum(1)).sum() / 2.0; den += w.sum()
    L0 = float(np.sqrt(num / den))
    # pairs in different clones: the tissue-wide random-pair spread, from spot positions and bin counts only
    from scipy.spatial.distance import cdist
    wq = nb / nb.sum(); Lx = float(np.sqrt((wq[:, None] * wq[None, :] * cdist(xy, xy) ** 2).sum() / 2.0))
    n = len(cl)
    return dict(M=M, D=squareform(pdist(xy)), q=q, NC=np.zeros((n, n)), rho=1.0, cl=cl, K=K, Nspot=comp, spots=xy,
                xy_true=np.zeros((n, 2)), AGE=d["C_tree"].astype(float) / 2.0, J=J, L0=L0, Lx=Lx, c0=0.0, inB=None,
                tag=f"eTRACER-{tag}_s{seed}", comp=comp)


def density_agreement(T, X):
    """22_etracer_map.score on the spots that are NOT anchors"""
    from scipy.stats import pearsonr
    held = np.setdiff1d(np.arange(len(X["spots"])), X["J"]); comp = X["comp"][held]
    rs, ws = [], []
    for k in range(X["K"]):
        tru = comp[:, k]
        if tru.sum() < 20 or tru.std() == 0:
            continue
        pred = T[X["cl"] == k][:, held].sum(0)
        if pred.sum() <= 0 or pred.std() == 0:
            continue
        rs.append(pearsonr(pred, tru).statistic); ws.append(tru.sum())
    return float(np.average(rs, weights=ws))


def job(args):
    tag, seed, arm = args
    X = data_etracer(tag, seed)
    n = X["M"].shape[0]; p = np.ones(n) / n; D = X["D"]
    if arm == "expression":
        import ot
        T = ot.emd(p, X["q"], X["M"], numItermax=10 ** 7)
        return dict(tag=tag, seed=seed, arm=arm, dens=density_agreement(T, X))
    if arm == "fgw":
        W = np.ones((n, n)); np.fill_diagonal(W, 0.0); WPhi = W * 2.0 * X["AGE"]
        sc = m70.scale(X, W, WPhi); Jv = X["J"][len(X["J"]) // 2:]; rows = []
        for w in m70.WGRID[1:]:
            T = m70.solve_exact(X["M"], W * sc, WPhi * sc, D, D ** 2, p, X["q"], w / (1 + w))
            rows.append(dict(w=w, nll_val=m70.anchor_nll(T, X, Jv), dens=density_agreement(T, X)))
        pick = min(rows, key=lambda r: r["nll_val"])
        return dict(tag=tag, seed=seed, arm=arm, dens=pick["dens"], w=pick["w"], path=rows)
    if arm == "law-path":                                   # shared within-clone L, weight picked by the anchor
        same = X["cl"][:, None] == X["cl"][None, :]
        W, WPhi = m70.moments_np(X, np.where(same, X["L0"], X["Lx"]), 0.0)
        sc = m70.scale(X, W, WPhi); Jv = X["J"][len(X["J"]) // 2:]; rows = []
        for w in m70.WGRID[1:]:
            T = m70.solve_exact(X["M"], W * sc, WPhi * sc, D, D ** 2, p, X["q"], w / (1 + w))
            rows.append(dict(w=w, nll_val=m70.anchor_nll(T, X, Jv), dens=density_agreement(T, X)))
        pick = min(rows, key=lambda r: r["nll_val"])
        return dict(tag=tag, seed=seed, arm=arm, dens=pick["dens"], w=pick["w"], path=rows, L0=X["L0"], Lx=X["Lx"])
    groups = X["cl"].copy() if arm == "per-clone" else np.zeros(X["K"], int)[X["cl"]]
    log, best = m70.learn(X, groups, learn_a=True, a0=0.197 / 1.197)
    return dict(tag=tag, seed=seed, arm=arm, dens=density_agreement(best["T"], X), a=best["a"], L=best["L"],
                best_step=best["step"], L0=X["L0"], sizes=np.bincount(X["cl"], minlength=X["K"]).tolist(),
                log=[{k: v for k, v in r.items() if k != "T"} for r in log])


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ARMS = os.environ.get("ARMS", "expression,fgw,law-path,weight,per-clone").split(",")
    SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    jobs = [(t, s, a) for t in ["LE", "EE"] for s in SEEDS for a in ARMS]
    t0 = time.time()
    with get_context("spawn").Pool(int(os.environ.get("PROCS", 12))) as pool:
        res = pool.map(job, jobs)
    json.dump(res, open(HERE / "spotr_out" / os.environ.get("OUTJSON", "etracer.json"), "w"), default=float)
    for t in ["LE", "EE"]:
        for a in ARMS:
            v = [r["dens"] for r in res if r["tag"] == t and r["arm"] == a]
            print(f"{t} {a:<11} clone-density agreement on non-anchor spots: {np.mean(v):.3f}  ({', '.join(f'{x:.3f}' for x in v)})")
    print(f"{len(res)} jobs in {time.time() - t0:.0f}s")
