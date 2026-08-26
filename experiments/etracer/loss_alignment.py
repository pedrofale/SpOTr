"""Is the loss SpOTr optimises aligned with the metric that actually improves?

The forward sweep showed mapping quality (ARI) peaks near alpha=0.075. If the training
loss does NOT have its minimum there, then learning alpha from that loss cannot recover
the useful value -- which would explain both the flat loss and the failed alpha-vs-
concordance correlation.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
X_sc, X_st = d["X_sc"], d["X_st"]
A, clones, truth, cell_clone = d["A"], d["clones"], d["bin_clone"], d["cell_clone"]
asc, ast = ad.AnnData(X_sc), ad.AnnData(X_st)
sig = jnp.asarray(np.stack([X_sc[cell_clone == c].mean(0) for c in clones]) * 2.0)
Y = jnp.asarray(X_st)
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])

print(f"{'alpha':>7} {'train loss':>12} {'ARI':>8}")
print("-" * 30)
rows = []
for al in [0.0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]:
    g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, 0.02, T_sinkhorn=100, J_alt=5, alpha=al)
    if not np.isfinite(np.asarray(g)).all():
        continue
    loss = float(S.deconvolution_loss_ls(Y, g, jnp.asarray(A), sig))
    prop = np.asarray(g).T @ A
    pred = clones[(prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)).argmax(1)]
    ari = adjusted_rand_score(truth, pred)
    rows.append(dict(alpha=al, loss=loss, ari=ari))
    print(f"{al:>7.3f} {loss:>12.6f} {ari:>8.4f}")

df = pd.DataFrame(rows)
df.to_csv(HERE / "loss_alignment.csv", index=False)
la, aa = df.loc[df.loss.idxmin(), "alpha"], df.loc[df.ari.idxmax(), "alpha"]
r = np.corrcoef(df.loss, df.ari)[0, 1]
print("-" * 30)
print(f"  loss is minimised at alpha = {la}")
print(f"  ARI  is maximised  at alpha = {aa}")
print(f"  corr(loss, ARI) = {r:+.3f}   (want strongly NEGATIVE: lower loss -> higher ARI)")
print(f"  loss range across alpha: {df.loss.max()-df.loss.min():.2e} "
      f"({100*(df.loss.max()-df.loss.min())/df.loss.mean():.2f}% of its mean)")
print(f"\n  -> {'ALIGNED' if la == aa else 'MISALIGNED: the objective does not point at the useful alpha'}")
