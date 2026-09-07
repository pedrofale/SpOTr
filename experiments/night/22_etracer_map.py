"""eTRACER: can the relationship be learned from CLONE-LABELLED SPATIAL DATA alone?

No cell here has a position.  What the spatial assay provides is a clone label per bin.  For a
pair of spots with clone compositions (comp_j, comp_l), the tree distance between the cells they
hold is unknown but its distribution is available from the dissociated data: draw a clone from
each composition, then draw a patristic distance from the empirical distribution for that clone
pair.  Sampling this gives (tree distance, spatial distance) pairs, and the correspondence
analysis proceeds unchanged.

  reference : a random half of the spots -- only their clone compositions and coordinates
  query     : all dissociated cells, with expression and the tree, no positions
  scoring   : on the OTHER half of the spots, per clone, predicted density against the clone's
              true share of each spot

Nothing about the held-out spots' composition is used to fit anything.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "3"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
import numpy as np, pandas as pd, ot
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import GenFGW, fit_kernel, standardise, apply_lookup

WGRID = np.geomspace(0.004, 1.2, 11)
_S = {}


def load(tag):
    d = np.load(HERE / f"etracer_spots_{tag}.npz", allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def clone_pair_distances(CT, cellc, clones, rng, cap=4000):
    """empirical patristic distances for every ordered clone pair"""
    idx = {c: np.where(cellc == c)[0] for c in clones}
    out = {}
    for a in range(len(clones)):
        for b in range(a, len(clones)):
            ia, ib = idx[clones[a]], idx[clones[b]]
            if len(ia) == 0 or len(ib) == 0:
                continue
            D = CT[np.ix_(ia, ib)]
            v = D[np.triu_indices(len(ia), 1)] if a == b else D.ravel()
            if len(v) == 0:
                continue
            if len(v) > cap:
                v = rng.choice(v, cap, replace=False)
            out[(a, b)] = out[(b, a)] = v
    return out


def sample_joint(comp, xy, spots, cpd, rng, n_draw=1_500_000):
    """draw (tree distance, spatial distance) pairs from the reference spots"""
    P = comp[spots] / np.maximum(comp[spots].sum(1, keepdims=True), 1e-12)
    cum = P.cumsum(1)
    j = rng.integers(0, len(spots), n_draw); l = rng.integers(0, len(spots), n_draw)
    keep = j != l
    j, l = j[keep], l[keep]
    k1 = (rng.random(len(j))[:, None] < cum[j]).argmax(1)
    k2 = (rng.random(len(l))[:, None] < cum[l]).argmax(1)
    s = np.linalg.norm(xy[spots[j]] - xy[spots[l]], axis=1)
    t = np.empty(len(j))
    ok = np.zeros(len(j), bool)
    for key in np.unique(np.stack([k1, k2]), axis=1).T:
        m = (k1 == key[0]) & (k2 == key[1])
        v = cpd.get((int(key[0]), int(key[1])))
        if v is None or not m.any():
            continue
        t[m] = rng.choice(v, m.sum()); ok |= m
    return t[ok], s[ok]


def _init(tag, R, seed):
    S = load(tag)
    rng = np.random.default_rng(seed)
    CT = S["C_tree"].astype(float); comp = S["spot_comp"].astype(float)
    xy = S["xy_st"].astype(float); clones = S["clones"].astype(str)
    m = len(xy)
    perm = rng.permutation(m)
    ref, held = perm[:m // 2], perm[m // 2:]
    CS = squareform(pdist(xy))
    M = ot.dist(S["X_sc"].astype(float), S["X_st"].astype(float)); M /= M.max()
    if R == 0:
        Phi = [standardise(CT)]; Psi = [standardise(CS)]; rho = np.array([1.0])
    else:
        cpd = clone_pair_distances(CT, S["cell_clone"].astype(str), clones, rng)
        t, s = sample_joint(comp, xy, ref, cpd, rng)
        K = fit_kernel(t, s, ncomp=R, nbin=17)
        Phi = [standardise(apply_lookup(CT, K["tc"], K["phi"][r])) for r in range(R)]
        Psi = [standardise(apply_lookup(CS, K["sc"], K["psi"][r])) for r in range(R)]
        rho = K["rho"][:R]
    _S["d"] = dict(M=M, Phi=Phi, Psi=Psi, rho=rho, comp=comp, held=held,
                   cellc=S["cell_clone"].astype(str), clones=clones, n=M.shape[0], m=m)


def score(T, S, permute=None, rng=None):
    lab = S["cellc"] if permute is None else rng.permutation(S["cellc"])
    held = S["held"]; comp = S["comp"][held]
    rs, ws = [], []
    for k, c in enumerate(S["clones"]):
        tru = comp[:, k]
        if tru.sum() < 20 or tru.std() == 0:
            continue
        pred = T[lab == c][:, held].sum(0)
        if pred.sum() <= 0:
            continue
        rs.append(pearsonr(pred / pred.sum(), tru / tru.sum()).statistic); ws.append(tru.sum())
    return float(np.average(rs, weights=ws))


def _solve(w):
    S = _S["d"]
    p = np.ones(S["n"]) / S["n"]; q = np.ones(S["m"]) / S["m"]
    if w == 0:
        T = ot.emd(p, q, S["M"], numItermax=1_000_000)
    else:
        T, _ = GenFGW(S["M"], S["Phi"], S["Psi"], S["rho"], w / (1 + w)).solve(p, q, 200, 1e-9)
    T = np.asarray(T, float)
    rng = np.random.default_rng(0)
    return w, score(T, S), float(np.mean([score(T, S, True, rng) for _ in range(20)]))


if __name__ == "__main__":
    rows = []
    for tag in ["LE", "EE"]:
        for seed in (0, 1, 2):
            for R, name in [(0, "FGW"), (2, "learned")]:
                with Pool(6, initializer=_init, initargs=(tag, R, seed)) as pool:
                    for w, r, rn in pool.imap_unordered(_solve, [0.0] + list(WGRID)):
                        rows.append(dict(tag=tag, seed=seed, arm=name, w=w, r=r, null=rn))
        g = pd.DataFrame(rows).query("tag == @tag")
        e = g[g.w == 0].r.mean()
        print(f"\n{tag}:  expression only r = {e:+.3f}   (clone labels permuted "
              f"{g.null.mean():+.3f})")
        for name in ["FGW", "learned"]:
            h = g[(g.arm == name) & (g.w > 0)]
            b = h.groupby("seed").r.max()
            print(f"   {name:<9} best r = {b.mean():+.3f}   per seed "
                  + " ".join(f"{x:+.3f}" for x in b.values), flush=True)
        pd.DataFrame(rows).to_csv(HERE / "etracer_map.csv", index=False)
    print("\ndone")
