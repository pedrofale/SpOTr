"""At what level should the tree be cut to define the per-clade alpha groups?

The 18 "clades" are a precomputed annotation from the source dataset, not a choice SpOTr made.
Cutting finer gives groups that are more phylogenetically coherent and more spatially localised,
but fewer cells each, so alpha_k is noisier. Cutting coarser does the reverse. There should be
an optimum and nobody has looked for it.

Groups are defined by average linkage on the cell x cell tree metric itself (its within-clade
cophenetic r is 1.000, so linkage reproduces the tree faithfully), cut with fcluster at each K.
For every K: best global alpha, and the HELD-OUT per-clade gain (alpha fitted by coordinate
ascent on half the spots, scored on the other half), which is the honest measure of headroom.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
GRID_G = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
GRID_K = [0.0, 0.10, 0.30, 0.60]
KS = [4, 9, 18, 36, 72]
NSPLIT = 2
SEC = "M3-4"
rng = np.random.default_rng(0)

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
CT = np.asarray(d["C_tree"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
COMP = np.load(HERE / f"comp_{SEC}.npy").astype(float)
clades0 = list(np.asarray(d["clades"]).astype(str))
cidx0 = np.array([clades0.index(c) for c in cc])
n, m = len(cc), X_st.shape[0]

Z = linkage(squareform(CT, checks=False), "average")
print(f"[tree] whole-matrix cophenetic r = "
      f"{pearsonr(cophenet(Z), squareform(CT, checks=False))[0]:.4f}", flush=True)

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         CT.astype(np.float32),
                                         np.asarray(d["C_space"], np.float32))
g0 = a[:, None] * b[None, :]
iu = np.triu_indices(n, 1); v = CT[iu]

rows = []
for K in KS:
    lab = fcluster(Z, K, criterion="maxclust") - 1
    Kr = lab.max() + 1
    A = np.zeros((n, Kr), np.float32); A[np.arange(n), lab] = 1.0
    omega = jnp.asarray(A)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    sizes = np.bincount(lab)
    same = lab[iu[0]] == lab[iu[1]]
    wcv = v[same].std() / v[same].mean() if same.sum() > 10 else np.nan
    agree = (pd.crosstab(lab, cidx0).max(1).sum() / n)

    def solve(ak):
        def build(g):
            return build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf,
                                                C_T, C_S, a, b, g, omega, sF, sL,
                                                equalize="measured")
        g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
        return np.asarray(g)

    def score(g, cols):
        if not np.isfinite(g).all(): return np.nan
        return float(COMP[cols[np.asarray(g)[:, cols].argmax(1)], cidx0].mean())

    allc = np.arange(m)
    gs = {vv: solve(np.full(Kr, vv)) for vv in GRID_G}
    bg = max(gs, key=lambda vv: score(gs[vv], allc))
    print(f"[K={K:>3} -> {Kr:>3} groups] sizes {sizes.min()}-{sizes.max()} "
          f"(median {int(np.median(sizes))}) | within-group CV {wcv:.3f} | "
          f"purity vs the 18 annotated clades {agree:.3f} | best global a={bg} "
          f"{score(gs[bg], allc):.4f}", flush=True)

    for rep in range(NSPLIT):
        h = rng.random(m) < 0.5
        fit, hel = np.where(h)[0], np.where(~h)[0]
        bgf = max(gs, key=lambda vv: score(gs[vv], fit))
        ak = np.full(Kr, float(bgf), np.float32)
        cur = score(solve(ak), fit)
        for k in range(Kr):
            best, bv = cur, ak[k]
            for vv in GRID_K:
                if vv == ak[k]: continue
                tk = ak.copy(); tk[k] = vv
                s_ = score(solve(tk), fit)
                if s_ > best: best, bv = s_, vv
            ak[k], cur = bv, best
        pc = score(solve(ak), hel)
        gl = score(gs[bgf], hel)
        un = score(solve(np.full(Kr, float(ak.mean()))), hel)
        rows.append(dict(K=K, Kr=Kr, rep=rep, n_min=int(sizes.min()),
                         n_med=int(np.median(sizes)), within_cv=wcv, purity=agree,
                         best_global=bgf, held_global=gl, held_perclade=pc, held_uniform=un,
                         alpha_mean=float(ak.mean()), alpha_sd=float(ak.std())))
        print(f"    rep{rep}: held-out global {gl:.4f} | per-clade {pc:.4f} "
              f"| uniform@mean {un:.4f} || pc-global {pc-gl:+.4f}", flush=True)

r = pd.DataFrame(rows); r.to_csv(HERE / "cut_level.csv", index=False)
print("\n" + "=" * 88)
print(f"{'cut K':>7}{'groups':>8}{'median n':>10}{'within CV':>11}{'purity':>9}"
      f"{'held global':>13}{'held per-clade':>16}{'gain':>9}")
print("-" * 83)
for K, g_ in r.groupby("K"):
    print(f"{K:>7}{g_.Kr.iloc[0]:>8}{g_.n_med.iloc[0]:>10}{g_.within_cv.iloc[0]:>11.3f}"
          f"{g_.purity.iloc[0]:>9.3f}{g_.held_global.mean():>13.4f}"
          f"{g_.held_perclade.mean():>16.4f}{(g_.held_perclade-g_.held_global).mean():>+9.4f}")
