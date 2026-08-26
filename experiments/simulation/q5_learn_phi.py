"""Q5: does the LEARNED per-clade phi identify which clades are spatially structured?

Q4 settled the performance question: a uniform phi = 1 beats the oracle per-clade split,
because the clades where you would turn the tree down are exactly the clades where
expression already resolves position, so turning it down buys nothing. Per-clade weighting
is therefore not a route to a better map.

But Q4 also showed the per-clade SIGNAL is real -- oracle > shuffled > inverted, z = +2.1.
That supports the other half of the claim: phi_k as a READOUT of which parts of the tree
behave differently in space. This tests it end to end, with nothing supervised: phi_k is
fitted against the deconvolution loss (fit each spot's expression as a mixture of per-clade
mean profiles), and only afterwards compared to the ground-truth tree-fine / expr-fine split.

Run from several inits, because a readout that only appears from one starting point is not
a readout.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

# the stock equalizer rescales a clade's rows BECAUSE its phi deviates, which in entropic OT
# is a per-clade temperature -- that is what made per-clade parameters unreadable. v2
# equalises on the measured cost magnitude instead, so s_k does not depend on phi_k.
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")

d = np.load(HERE / "sim_cf.npz", allow_pickle=True)
X_sc, X_st = d["X_sc"], d["X_st"]
cell_clade, clades, is_tf = d["cell_clade"], d["clades"], d["is_tree_fine"]
K = len(clades)
A = np.zeros((len(cell_clade), K), dtype=np.float32)
for i, c in enumerate(cell_clade):
    A[i, list(clades).index(c)] = 1.0

sc_ad = ad.AnnData(np.asarray(X_sc, dtype=np.float32))
sc_ad.obs["clade"] = pd.Categorical([str(c) for c in cell_clade])
st_ad = ad.AnnData(np.asarray(X_st, dtype=np.float32))
st_ad.layers["expr"] = st_ad.X.copy()
sig = np.stack([X_sc[cell_clade == c].mean(0) for c in clades]).astype(np.float32) * 2.0

truth = pd.Series(is_tf, index=[str(c) for c in clades])
rows = []
for init in [0.2, 0.5, 0.8]:
    res = S.run_spotr_perclade_alpha(
        sc_ad, st_ad, np.asarray(d["C_tree"]), np.asarray(d["C_space"]), A, sig,
        spot_layer="expr", clade_column="clade", clade_to_ignore="__none__",
        eps=0.02, T_sinkhorn=50, J_alt=3, n_iters=400, lr=0.05,
        alpha=0.7, use_decomp=True, phi=init, use_loss_ls=True)
    phi = pd.Series({str(k): float(v) for k, v in res["phi_by_clade"].items()}).reindex(truth.index)
    al = pd.Series({str(k): float(v) for k, v in res["alpha_by_clade"].items()}).reindex(truth.index)
    loss = np.asarray(res["loss_hist"])
    auc = roc_auc_score(truth.values, phi.values)
    r, p = stats.pointbiserialr(truth.values.astype(int), phi.values)
    print(f"\n=== phi_init = {init} ===  loss {loss[0]:.5f} -> {loss[-1]:.5f}"
          f"{'  (NaN!)' if np.isnan(loss).any() else ''}")
    print(f"  mean phi   tree-fine {phi[truth].mean():.4f}   expr-fine {phi[~truth].mean():.4f}"
          f"   diff {phi[truth].mean()-phi[~truth].mean():+.4f}")
    print(f"  AUC(phi -> tree-fine) = {auc:.3f}    point-biserial r = {r:+.3f} (p = {p:.3f})")
    print(f"  mean alpha tree-fine {al[truth].mean():.4f}   expr-fine {al[~truth].mean():.4f}")
    rows.append(dict(init=init, auc=auc, r=r, p=p,
                     phi_tf=phi[truth].mean(), phi_ef=phi[~truth].mean(),
                     spread=float(phi.max() - phi.min()), loss_end=float(loss[-1])))
    for c in truth.index:
        rows[-1][f"phi_{c}"] = phi[c]

r = pd.DataFrame(rows)
print("\n" + "=" * 66)
print(f"{'init':>6} {'AUC':>7} {'r':>8} {'p':>8} {'phi tf':>8} {'phi ef':>8} {'spread':>8}")
for _, x in r.iterrows():
    print(f"{x.init:>6.1f} {x.auc:>7.3f} {x.r:>+8.3f} {x.p:>8.3f} "
          f"{x.phi_tf:>8.4f} {x.phi_ef:>8.4f} {x.spread:>8.4f}")
print(f"\nAUC across inits: {r.auc.min():.3f} - {r.auc.max():.3f}  (0.5 = no signal)")
print("-> " + ("phi IDENTIFIES the tree-structured clades, consistently across inits"
               if r.auc.min() > 0.65 else
               "phi does NOT reliably identify them" if r.auc.max() < 0.65 else
               "signal present but INIT-DEPENDENT -- does not meet the standard"))
r.to_csv(HERE / "q5_learn_phi.csv", index=False)
