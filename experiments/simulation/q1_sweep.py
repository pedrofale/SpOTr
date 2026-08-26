"""Q1 across expression regimes: at what expr_alpha does the GW tree term start to help?"""
import sys
from pathlib import Path
import numpy as np, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]

print(f"{'expr_alpha':>11} " + "".join(f"{g:>9.2f}" for g in GRID) + f"{'  best':>10}{'  gain':>9}")
print("-" * 84)
for A in ["1.0", "0.2", "0.0"]:
    d = np.load(HERE / f"sim_a{A}.npz", allow_pickle=True)
    Amat, clones, truth = d["A"], d["clones"], d["bin_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    aris = []
    for av in GRID:
        g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=T, J_alt=J, alpha=av)
        g = np.asarray(g)
        if not np.isfinite(g).all():
            aris.append(np.nan); continue
        p = g.T @ Amat
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        aris.append(adjusted_rand_score(truth, clones[p.argmax(1)]))
    aris = np.array(aris)
    best_i = int(np.nanargmax(aris))
    gain = aris[best_i] - aris[0]
    flag = "  <-- tree HELPS" if gain > 1e-3 else ""
    print(f"{A:>11} " + "".join(f"{v:>9.4f}" for v in aris)
          + f"{GRID[best_i]:>10.2f}{gain:>+9.4f}{flag}")
