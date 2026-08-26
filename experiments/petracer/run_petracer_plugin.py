"""Set alpha_k from an INDEPENDENT measurement instead of from the OT loss.

The anchor loss is biased, not noisy: it selects alpha = 0.1 where accuracy peaks at 0.6, a
systematic offset across the whole grid. A prior would shrink toward wherever it is centred and
would not recover the missing information, so it cannot fix this.

But the information the loss cannot see IS measurable, from the spatial side alone. If the
spatial assay carries a lineage readout, then for each clade you can ask directly: are this
clade's cells spatially concentrated, or scattered? That needs no mapping and no OT. It is the
practical form of claim (b): measure which clades are spatially coherent, weight the tree
accordingly per clade.

Three ways to score a clade's spatial concentration, all computed from spot composition and
spot coordinates only:
  moran     Moran's I of the clade's per-spot share over the spot graph
  disp      1 - (spread of the clade's mass) / (spread of all spots), i.e. concentration
  knn       excess same-clade share among a spot's k nearest neighbours, weighted by share

Compared against: uniform alpha, the per-clade ORACLE (chosen on test accuracy, an upper
bound not attainable in practice), and shuffled plug-in as the null.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.spatial.distance import cdist
from scipy import stats

HERE = Path(__file__).resolve().parent
SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)
d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clade = np.asarray(d["cell_clade"]).astype(str)
clades = np.asarray(d["clades"]).astype(str)
A = np.asarray(d["A"], np.float32); xy = np.asarray(d["xy"], float)
K, n, m = len(clades), len(cell_clade), X_st.shape[0]
cidx = np.array([list(clades).index(c) for c in cell_clade])

import pycea as py
t = py.datasets.koblan25("tumor")
cl_of = t.obs["clade"].astype(str).to_dict()
COMP = np.zeros((m, K))
for nm, sp in zip(np.asarray(d["spot_cell_names"]).astype(str), np.asarray(d["spot_of_cell"])):
    c = cl_of.get(nm)
    if c in list(clades):
        COMP[sp, list(clades).index(c)] += 1
COMP = COMP / np.maximum(COMP.sum(1, keepdims=True), 1e-12)

perm = rng.permutation(m)
test_m = np.zeros(m, bool); test_m[perm[int(0.7 * m):]] = True
cols = np.where(test_m)[0]

# ---- per-clade spatial concentration, from spots only. No mapping, no OT. ----
D = cdist(xy, xy)
kn = np.argsort(D, axis=1)[:, 1:13]
W = np.zeros((m, m)); W[np.arange(m)[:, None], kn] = 1.0; W = (W + W.T) / 2
Wn = W / np.maximum(W.sum(), 1e-12)
tot = xy.std(0).mean()
feat = {}
for k in range(K):
    z = COMP[:, k]
    zc = z - z.mean()
    denom = (zc ** 2).sum()
    moran = float((m / W.sum()) * (zc @ W @ zc) / denom) if denom > 1e-12 else 0.0
    w = z / max(z.sum(), 1e-12)
    cen = w @ xy
    spread = float(np.sqrt((w * ((xy - cen) ** 2).sum(1)).sum()))
    same = float((z[:, None] * z[kn]).sum() / max((z[:, None] * np.ones_like(z[kn])).sum(), 1e-12))
    feat[clades[k]] = dict(moran=moran, disp=1 - spread / (tot * np.sqrt(2)), knn=same - z.mean())
F = pd.DataFrame(feat).T
print(f"[plug] per-clade spatial concentration from spots alone (no mapping):")
print(F.round(4).to_string())

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)


def solve(ak):
    def build(g):
        return build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf, C_T,
                                            C_S, a, b, g, omega, sF, sL, equalize="measured")
    if TAU <= 0:
        g = g0
        for _ in range(J):
            g, _u = S.sinkhorn_unrolled_safe(build(g), a, b, EPS, T)
        return np.asarray(g)
    g, _uv = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
    return np.asarray(g)


def score(g):
    if not np.isfinite(g).all():
        return np.nan
    j = cols[np.asarray(g)[:, cols].argmax(1)]
    return float(COMP[j, cidx].mean())


BEST = 0.60          # accuracy-optimal uniform alpha from the sweep
res = {"alpha = 0": score(solve(np.zeros(K))),
       f"uniform alpha = {BEST}": score(solve(np.full(K, BEST)))}
print(f"\n{'setting':<40}{'soft acc':>10}")
print("-" * 50)
for k, v in res.items():
    print(f"{k:<40}{v:>10.4f}")

best_plug, best_nm = -np.inf, None
for nm in ["moran", "disp", "knn"]:
    v = F[nm].to_numpy()
    r = stats.rankdata(v) / K                       # rank-normalise; scale-free
    for lo, hi in [(0.0, 2 * BEST), (0.2, 2 * BEST), (0.3, 0.9)]:
        ak = lo + (hi - lo) * r
        s = score(solve(ak))
        print(f"{'plug-in ' + nm + f' [{lo:.1f},{hi:.1f}]':<40}{s:>10.4f}")
        if s > best_plug:
            best_plug, best_nm, best_ak = s, f"{nm} [{lo:.1f},{hi:.1f}]", ak
sh = [score(solve(rng.permutation(best_ak))) for _ in range(8)]
print("-" * 50)
print(f"{'best plug-in (' + best_nm + ')':<40}{best_plug:>10.4f}")
print(f"{'control: shuffled plug-in (8)':<40}{np.nanmean(sh):>10.4f}")
u = res[f"uniform alpha = {BEST}"]
print(f"\nplug-in vs uniform-at-best-alpha : {best_plug - u:+.4f}   <- THE BAR")
print(f"plug-in vs shuffled              : {best_plug - np.nanmean(sh):+.4f} "
      f"(shuffled {np.nanmean(sh):.4f} +- {np.nanstd(sh):.4f}, "
      f"z = {(best_plug - np.nanmean(sh)) / max(np.nanstd(sh), 1e-9):+.1f})")
F.to_csv(HERE / f"petracer_{SEC}_clade_features.csv")
