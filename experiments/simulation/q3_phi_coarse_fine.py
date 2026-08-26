"""Q3: per-clade PHI on the coarse-to-fine testbed.

Q2 showed per-clade ALPHA cannot express the difference between the two clade types, and
the reason is structural. For cell i, the GW cost sums over all other cells k; only ~1/K
of those pairs are inside i's own clade, so alpha_k -- a single weight on the whole GW
term -- is dominated by the CROSS-clade contribution, which is identical for every clade.
That is why the tree helped expr-fine clades too (their between-clade structure is just as
real) and why the oracle alpha split lost to a uniform alpha.

phi_k is the parameter that can express it:

    L_hat = (1 - phi_k) * L_cross + phi_k * L_within

    tree-fine clades  within-clade tree predicts within-region position -> phi_k HIGH
    expr-fine clades  within-clade tree is iid noise                    -> phi_k LOW

Both keep the cross-clade term that helps everyone. Only the within-clade contribution is
switched per clade, which is the only thing that actually differs by construction.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)
f = HERE / "sim_cf.npz"
if time.time() - f.stat().st_mtime > 86400:
    raise SystemExit(f"REFUSING: sim_cf.npz is {(time.time()-f.stat().st_mtime)/60:.0f} min old.")
d = np.load(f, allow_pickle=True)

xy_sc, xy_st = d["xy_sc"], d["xy_st"]
diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))
cell_clade, cell_sub, spot_sub = d["cell_clade"], d["cell_sub"], d["spot_sub"]
spot_clade = d["spot_clade"]
clades, is_tf = d["clades"], d["is_tree_fine"]
K = len(clades)
tf_cell = np.isin(cell_clade, clades[is_tf])

A = np.zeros((len(cell_clade), K), dtype=np.float32)
for i, c in enumerate(cell_clade):
    A[i, list(clades).index(c)] = 1.0

asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
omega = jnp.asarray(A)
Omega = jnp.asarray(A @ A.T)                    # [n,n] same-clade mask
g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)


def solve(ak, pk):
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                         Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                         use_decomp=True, Omega=Omega,
                                         phi_k=jnp.asarray(np.asarray(pk, np.float32)),
                                         equalize="measured")
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    return np.asarray(g)



# Barycentre error ||E_j[xy_j] - xy_i|| is minimised by HEDGING: spreading mass over the
# region returns its centroid, which has low error, while committing to one spot is punished
# whenever it is wrong. By Jensen it systematically understates a sharp coupling. A refiner
# commits, so barycentre error is the wrong instrument here. err_exp = E_j||xy_j - xy_i|| is
# the proper transport-distance analogue and does not reward hedging; err_top1 scores the
# argmax spot only.
from scipy.spatial.distance import cdist as _cdist
_D = _cdist(xy_sc, xy_st) / diam


def sc(g):
    if not np.isfinite(g).all():
        return (np.nan,) * 7
    w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
    e = (w * _D).sum(1)                              # expected distance, hedging-neutral
    j = g.argmax(1)
    return (e.mean(), e[tf_cell].mean(), e[~tf_cell].mean(),
            adjusted_rand_score(cell_sub, spot_sub[j]),
            (spot_sub[j] == cell_sub).mean(),                       # subclade accuracy
            (spot_sub[j][tf_cell] == cell_sub[tf_cell]).mean(),     # ... on tree-fine
            (spot_clade[j] == cell_clade).mean())                   # clade accuracy


print("global (alpha, phi) grid -- spatial error")
print(f"{'alpha':>6} " + "".join(f"{p:>9.2f}" for p in [0.0, 0.25, 0.5, 0.75, 1.0]))
print("-" * 56)
best_g, gcfg = np.inf, None
for av in [0.20, 0.35, 0.50, 0.70]:
    row = []
    for pv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        v = sc(solve(np.full(K, av), np.full(K, pv)))[0]
        row.append(v)
        if v < best_g:
            best_g, gcfg = v, (av, pv)
    print(f"{av:>6.2f} " + "".join(f"{v:>9.4f}" for v in row))
ga, gp = gcfg
print(f"\nbest GLOBAL config: alpha={ga}  phi={gp}  spatial error {best_g:.4f}\n")

need = is_tf.astype(float)
H = f"{'setting':<46} {'E-dist':>8} {'tree-fine':>10} {'expr-fine':>10} {'ARIsub':>7} {'acc sub':>8} {'acc sub tf':>11} {'acc clade':>10}"
print(H); print("-" * len(H))
best, cfg = np.inf, None
for lo, hi in [(0.0, 1.0), (0.0, 0.75), (0.25, 1.0), (0.25, 0.75), (0.0, 0.5), (0.5, 1.0)]:
    v = sc(solve(np.full(K, ga), lo + (hi - lo) * need))
    print(f"{'ORACLE phi  expr-fine=' + f'{lo:.2f}' + '  tree-fine=' + f'{hi:.2f}':<46} "
          f"{v[0]:>8.4f} {v[1]:>10.4f} {v[2]:>10.4f} {v[3]:>7.4f} {v[4]:>8.4f} {v[5]:>11.4f} {v[6]:>10.4f}")
    if v[0] < best:
        best, cfg = v[0], (lo, hi)
lo, hi = cfg
sh = [sc(solve(np.full(K, ga), lo + (hi - lo) * rng.permutation(need)))[0] for _ in range(5)]
inv = sc(solve(np.full(K, ga), lo + (hi - lo) * (1 - need)))[0]
gv = sc(solve(np.full(K, ga), np.full(K, gp)))
print("-" * len(H))
print(f"{'best GLOBAL (alpha,phi)':<46} {gv[0]:>8.4f} {gv[1]:>10.4f} {gv[2]:>10.4f} {gv[3]:>7.4f} {gv[4]:>8.4f} {gv[5]:>11.4f} {gv[6]:>10.4f}")
print(f"{'ORACLE per-clade phi':<46} {best:>8.4f}   gain {gv[0]-best:+.4f} ({100*(gv[0]-best)/gv[0]:+.1f}%)")
print(f"{'control: shuffled phi (5 seeds)':<46} {np.nanmean(sh):>8.4f} +- {np.nanstd(sh):.4f}")
print(f"{'control: INVERTED phi':<46} {inv:>8.4f}")
z = (np.nanmean(sh) - best) / max(np.nanstd(sh), 1e-9)
print(f"\noracle {z:+.1f} sd better than shuffled; inverted {inv-best:+.4f} worse than oracle")
pd.DataFrame([dict(alpha=ga, phi_global=gp, err_global=gv[0], err_oracle=best,
                   err_shuffled=float(np.nanmean(sh)), sd_shuffled=float(np.nanstd(sh)),
                   err_inverted=inv, gain=gv[0]-best, z=z)]).to_csv(HERE / "q3_phi_coarse_fine.csv", index=False)
