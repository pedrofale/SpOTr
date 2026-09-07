"""Is 18 clades too fine a resolution for alpha_k?

Coarsen by cutting the CLADE-level tree (the cell-level tree's top is a near-polytomy, so
fcluster jumps 18 -> 2). For each granularity: eta2 on per-cell best alpha, group-level
test-retest reliability across disjoint spot halves, and a SIZE-MATCHED RANDOM control -- which
is mandatory here, because larger groups average more cells and so reproduce better on their own.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90])
LEVELS = [2, 3, 4, 6, 9, 18]
NREP = 3
SEC = "M3-4"
rng = np.random.default_rng(0)

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
CT = np.asarray(d["C_tree"], np.float32); CSp = np.asarray(d["C_space"], np.float32)
n, m, K = len(cc), X_st.shape[0], len(clades)
A18 = np.zeros((n, K), np.float32); A18[np.arange(n), cidx] = 1.0

dd = np.load("petracer_xy.npz", allow_pickle=True)
Dcl = np.asarray(dd["D"], float); allcl = list(dd["clades"].astype(str))
Zc = linkage(squareform(Dcl, checks=False), "average")
MAP = {}
for L in LEVELS:
    lab_cl = fcluster(Zc, L, "maxclust") - 1
    MAP[L] = np.array([lab_cl[allcl.index(c)] for c in cc])
    print(f"  L={L:>2}: {len(set(MAP[L]))} groups, sizes "
          f"{sorted(np.bincount(MAP[L]), reverse=True)}")

def errors(cols):
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(
        ad.AnnData(X_sc), ad.AnnData(X_st[cols]), CT, CSp[np.ix_(cols, cols)])
    om = jnp.asarray(A18); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    E = np.zeros((n, len(GRID)))
    for c_, av in enumerate(GRID):
        def build(g):
            return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T,
                                                C_S, a, b, g, om, sF, sL, equalize="measured")
        g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
        E[:, c_] = np.linalg.norm(xy_st[cols][np.asarray(g).argmax(1)] - xy_sc, axis=1)
    return E

HALVES = []
for rep in range(NREP):
    h = rng.random(m) < 0.5
    HALVES.append((errors(np.where(h)[0]), errors(np.where(~h)[0])))
    print(f"  solved split {rep}", flush=True)
np.save(HERE / f"granularity_halves_{SEC}.npy", np.array(HALVES))
E_full = np.load(HERE / f"percell_err_{SEC}.npy")
best_full = GRID[E_full.argmin(1)]

def gopt(E, lab):
    return np.array([GRID[E[lab == k].mean(0).argmin()] for k in sorted(set(lab))])
def eta2(v, lab):
    gm = v.mean()
    return sum((v[lab==k].mean()-gm)**2*(lab==k).sum() for k in set(lab))/max(((v-gm)**2).sum(),1e-12)
def rand_like(lab):
    sizes = np.bincount(lab); perm = rng.permutation(n); out = np.full(n,-1); p0=0
    for gi, sz in enumerate(sizes): out[perm[p0:p0+sz]] = gi; p0 += sz
    return out

rows = []
for L in LEVELS:
    lab = MAP[L]
    rel, rnd, e_r = [], [], []
    for EA, EB in HALVES:
        oA, oB = gopt(EA, lab), gopt(EB, lab)
        rel.append(pearsonr(oA, oB)[0] if len(oA) > 2 else np.nan)
        rl = rand_like(lab)
        rnd.append(pearsonr(gopt(EA, rl), gopt(EB, rl))[0] if len(oA) > 2 else np.nan)
        e_r.append(eta2(best_full, rl))
    rows.append(dict(L=L, groups=len(set(lab)), min_size=int(np.bincount(lab).min()),
                     eta2=eta2(best_full, lab), eta2_rand=float(np.nanmean(e_r)),
                     rel=float(np.nanmean(rel)), rel_rand=float(np.nanmean(rnd)),
                     exact=float(np.mean([np.mean(gopt(a_,lab)==gopt(b_,lab)) for a_,b_ in HALVES]))))
r = pd.DataFrame(rows); r.to_csv(HERE / f"granularity_{SEC}.csv", index=False)
print("\n" + "="*94)
print(f"{'groups':>7}{'min size':>10}{'eta2':>9}{'eta2 rand':>11}{'reliability':>13}"
      f"{'rand ctrl':>11}{'advantage':>11}{'exact':>8}")
print("-"*94)
for _, x in r.iterrows():
    print(f"{int(x.groups):>7}{int(x.min_size):>10}{x.eta2:>9.4f}{x.eta2_rand:>11.4f}"
          f"{x.rel:>13.3f}{x.rel_rand:>11.3f}{x.rel-x.rel_rand:>+11.3f}{x.exact:>8.2f}")
