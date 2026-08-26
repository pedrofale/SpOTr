"""Q2 on the coarse-to-fine testbed: does per-clade alpha beat the best global alpha?

Q1 showed the tree helps here (spatial error 0.1253 -> 0.1069 at alpha 0.10), so unlike
every earlier testbed the precondition for this question is actually met.

  tree-fine clades  within-lambda 1, no subclade markers -> tree resolves the fine level
                                                            -> alpha_k should be HIGH
  expr-fine clades  within-lambda 0, subclade markers    -> tree within them is noise
                                                            -> alpha_k should be LOW

Oracle uses the ground-truth split; controls shuffle and invert it. Metric is spatial
error, as in Q1 -- clade-ARI is saturated by expression and cannot see a refiner.
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
if time.time() - f.stat().st_mtime > 3600:
    raise SystemExit(f"REFUSING: sim_cf.npz is {(time.time()-f.stat().st_mtime)/60:.0f} min old -- likely stale.")
d = np.load(f, allow_pickle=True)

xy_sc, xy_st = d["xy_sc"], d["xy_st"]
diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))
cell_clade, cell_sub, spot_sub = d["cell_clade"], d["cell_sub"], d["spot_sub"]
clades, is_tf = d["clades"], d["is_tree_fine"]
K = len(clades)
tf_cell = np.isin(cell_clade, clades[is_tf])

A = np.zeros((len(cell_clade), K), dtype=np.float32)
for i, c in enumerate(cell_clade):
    A[i, list(clades).index(c)] = 1.0

asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
omega = jnp.asarray(A)
g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)


def solve(ak):
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                         Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                         equalize="measured")
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    return np.asarray(g)


def err(g):
    if not np.isfinite(g).all():
        return np.nan, np.nan, np.nan, np.nan
    w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
    e = np.linalg.norm(w @ xy_st - xy_sc, axis=1) / diam
    return (e.mean(), e[tf_cell].mean(), e[~tf_cell].mean(),
            adjusted_rand_score(cell_sub, spot_sub[g.argmax(1)]))


GRID = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
gl = {v: err(solve(np.full(K, v))) for v in GRID}
bv = min(gl, key=lambda k: gl[k][0])
print("global alpha (per-clade solver): " + "  ".join(f"{v}:{gl[v][0]:.4f}" for v in GRID))
print(f"  best global alpha = {bv}  spatial error {gl[bv][0]:.4f}\n")

need = is_tf.astype(float)                       # 1 = tree-fine, 0 = expr-fine
print(f"{'setting':<44} {'err ALL':>8} {'tree-fine':>10} {'expr-fine':>10} {'ARI sub':>8}")
print("-" * 84)
best, cfg = np.inf, None
for lo, hi in [(0.0, bv), (0.0, 0.20), (0.0, 0.35), (0.05, 0.35), (0.05, 0.50), (0.10, 0.50)]:
    v = err(solve(lo + (hi - lo) * need))
    print(f"{'ORACLE  expr-fine=' + f'{lo:.2f}' + '  tree-fine=' + f'{hi:.2f}':<44} "
          f"{v[0]:>8.4f} {v[1]:>10.4f} {v[2]:>10.4f} {v[3]:>8.4f}")
    if v[0] < best:
        best, cfg = v[0], (lo, hi)
lo, hi = cfg
sh = [err(solve(lo + (hi - lo) * rng.permutation(need)))[0] for _ in range(5)]
inv = err(solve(lo + (hi - lo) * (1 - need)))[0]
print("-" * 84)
print(f"{'best global alpha':<44} {gl[bv][0]:>8.4f} {gl[bv][1]:>10.4f} {gl[bv][2]:>10.4f} {gl[bv][3]:>8.4f}")
print(f"{'ORACLE per-clade (best cfg)':<44} {best:>8.4f}   gain {gl[bv][0]-best:+.4f} "
      f"({100*(gl[bv][0]-best)/gl[bv][0]:+.1f}%)")
print(f"{'control: shuffled per-clade (5 seeds)':<44} {np.nanmean(sh):>8.4f} +- {np.nanstd(sh):.4f}")
print(f"{'control: INVERTED':<44} {inv:>8.4f}")
z = (np.nanmean(sh) - best) / max(np.nanstd(sh), 1e-9)
print(f"\noracle is {z:+.1f} sd better than shuffled; inverted is {inv-best:+.4f} worse than oracle")
pd.DataFrame([dict(best_global=bv, err_global=gl[bv][0], err_oracle=best,
                   err_shuffled=float(np.nanmean(sh)), err_inverted=inv,
                   gain=gl[bv][0]-best, z_vs_shuffled=z)]).to_csv(HERE / "q2_coarse_fine.csv", index=False)
