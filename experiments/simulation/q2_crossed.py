"""Q2 on the CROSSED design: does per-clade alpha beat a global alpha when clades differ in
WHICH modality is informative?

expression-driven clades (expr_alpha 1, lambda 0) -> tree unnecessary, alpha_k should be LOW
tree-driven clades      (expr_alpha 0, lambda 1) -> tree required,    alpha_k should be HIGH

First validates that the two groups really do differ in expression separability, then runs
the oracle against the best global alpha with shuffled and inverted controls.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)

import time as _t
_f = HERE / "sim_crossed.npz"
if _t.time() - _f.stat().st_mtime > 1800:
    raise SystemExit(f"REFUSING: sim_crossed.npz is "
                     f"{(_t.time()-_f.stat().st_mtime)/60:.0f} min old -- likely stale.")
d = np.load(_f, allow_pickle=True)
A, clones, truth = d["A"], d["clones"], d["bin_clone"]
lam, ea, cell_clone = d["true_lambda"], d["expr_alpha_clade"], d["cell_clone"]
X = d["X_sc"]

# --- validate the design: is expression separability actually different? ---
print("design check -- can expression tell each clade from the rest?")
for grp, name in [(ea == 1.0, "expression-driven"), (ea == 0.0, "tree-driven")]:
    accs = []
    for k in np.where(grp)[0]:
        y = (cell_clone == clones[k]).astype(int)
        if y.sum() < 15:
            continue
        accs.append(cross_val_score(LogisticRegression(max_iter=400, class_weight="balanced"),
                                    X, y, cv=3, scoring="balanced_accuracy").mean())
    print(f"  {name:<20} n={len(accs):<3} mean balanced accuracy from expression = {np.mean(accs):.3f}")

asc, ast = ad.AnnData(X), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
omega = jnp.asarray(A); K = len(clones)
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


def ari(g):
    if not np.isfinite(g).all():
        return np.nan
    p = g.T @ A
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return adjusted_rand_score(truth, clones[p.argmax(1)])


grid = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
gl = {v: ari(solve(np.full(K, v))) for v in grid}
bv = max(gl, key=lambda k: gl[k])
print(f"\nglobal alpha sweep: " + "  ".join(f"{v}:{gl[v]:.4f}" for v in grid))
print(f"  best global alpha = {bv} (ARI {gl[bv]:.4f})")

need_tree = (lam >= 0.5).astype(float)          # 1 = tree-driven, 0 = expression-driven
print(f"\n{'setting':<46} {'ARI':>8}")
print("-" * 58)
best_o, best_cfg = -np.inf, None
for lo, hi in [(0.0, bv), (0.0, 0.35), (0.0, 0.50), (0.0, 0.70), (0.05, 0.50)]:
    ak = lo + (hi - lo) * need_tree
    v = ari(solve(ak))
    print(f"{'ORACLE: low=' + f'{lo:.2f}' + ' expr-driven, high=' + f'{hi:.2f}' + ' tree-driven':<46} {v:>8.4f}")
    if v > best_o:
        best_o, best_cfg = v, (lo, hi)
lo, hi = best_cfg
sh = [ari(solve(lo + (hi - lo) * rng.permutation(need_tree))) for _ in range(3)]
inv = ari(solve(lo + (hi - lo) * (1 - need_tree)))
print("-" * 58)
print(f"{'best global alpha':<46} {gl[bv]:>8.4f}")
print(f"{'ORACLE per-clade':<46} {best_o:>8.4f}   gain {best_o - gl[bv]:+.4f}")
print(f"{'control: shuffled per-clade (3 seeds)':<46} {np.nanmean(sh):>8.4f}")
print(f"{'control: INVERTED':<46} {inv:>8.4f}")
pd.DataFrame([dict(best_global_alpha=bv, ari_global=gl[bv], ari_oracle=best_o,
                   ari_shuffled=float(np.nanmean(sh)), ari_inverted=inv,
                   gain=best_o - gl[bv])]).to_csv(HERE / "q2_crossed.csv", index=False)
