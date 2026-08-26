"""Fine alpha sweep around the optimum: is the real tree's gain over alpha=0 robust?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from sklearn.metrics import adjusted_rand_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
A, clones, truth, C_space = d["A"], d["clones"], d["bin_clone"], d["C_space"]
asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
rng = np.random.default_rng(1)

ALPHAS = [0.0, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30]
variants = {"real": d["C_tree"]}
for s in range(3):
    p = rng.permutation(d["C_tree"].shape[0])
    variants[f"shuf{s+1}"] = d["C_tree"][np.ix_(p, p)]

rows = []
for name, T in variants.items():
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, T, C_space)
    for al in ALPHAS:
        g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, 0.02, T_sinkhorn=100, J_alt=5, alpha=al)
        g = np.asarray(g)
        if not np.isfinite(g).all():
            continue
        prop = g.T @ A
        prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        pred = np.asarray(clones)[prop.argmax(1)]
        rows.append(dict(variant=name, alpha=al, acc=accuracy_score(truth, pred),
                         ari=adjusted_rand_score(truth, pred)))
df = pd.DataFrame(rows)
df.to_csv(HERE / "etracer_fine_sweep.csv", index=False)

piv = df.pivot_table(index="alpha", columns="variant", values="ari")
piv["shuf_mean"] = piv[[c for c in piv.columns if c.startswith("shuf")]].mean(1)
piv["shuf_sd"] = piv[[c for c in piv.columns if c.startswith("shuf")]].std(1)
print("ARI by alpha:")
print(piv[["real", "shuf_mean", "shuf_sd"]].to_string(float_format=lambda v: f"{v:.4f}"))
base = float(piv.loc[0.0, "real"])
best_a = piv["real"].idxmax(); best = float(piv.loc[best_a, "real"])
print(f"\nalpha=0 (no tree)      ARI = {base:.4f}")
print(f"best real-tree alpha   = {best_a}  ARI = {best:.4f}   gain = {best-base:+.4f}")
sd = float(piv.loc[best_a, "shuf_sd"])
print(f"shuffled at that alpha = {float(piv.loc[best_a,'shuf_mean']):.4f} +/- {sd:.4f}"
      f"   -> real is {(best-float(piv.loc[best_a,'shuf_mean']))/max(sd,1e-6):.1f} sd above shuffled")
n_better = int((piv["real"] > piv["shuf_mean"]).sum())
print(f"real > shuffled at {n_better}/{len(piv)} alphas")
