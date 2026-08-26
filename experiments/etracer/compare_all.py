"""One scoring pass over identical bins for every method.

The per-method reports are not comparable as they stand: the competitors were scored on all
1,026 barcoded bins while SpOTr was scored on a held-out subset, and no method was compared
against the trivial baseline. That baseline matters enormously here -- C4 alone holds 40% of
barcoded bins, so ALWAYS PREDICTING C4 scores 0.4055, and Tangram's headline 0.4503 is a
+0.045 edge over guessing.

Everything here is scored on the SAME bins with the SAME metrics, and accuracy is reported
next to balanced accuracy (mean per-clone recall), which a prior-hugging predictor cannot win.
"""
import functools, sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src")); sys.path.insert(0, str(HERE))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J = 0.02, 50, 3
d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clone = np.asarray(d["cell_clone"]).astype(str)
bin_clone = np.asarray(d["bin_clone"]).astype(str)
clones = np.asarray(d["clones"]).astype(str)
A = np.asarray(d["A"], np.float32); is_lab = np.asarray(d["is_labelled"], bool)
K, m = len(clones), len(bin_clone)
lab = np.where(is_lab)[0]
truth = bin_clone[lab]
print(f"[cmp] scoring every method on the SAME {len(lab)} barcoded bins, {K} clones")
vc = pd.Series(truth).value_counts()
print(f"[cmp] largest clone {vc.index[0]} holds {vc.iloc[0]}/{len(truth)} = {vc.iloc[0]/len(truth):.4f}")


def score(P, name, secs=np.nan):
    """P: [m, K] predicted clone proportions."""
    pred = clones[np.asarray(P)[lab].argmax(1)]
    return dict(method=name, acc=float((pred == truth).mean()),
                bal_acc=float(balanced_accuracy_score(truth, pred)),
                ari=float(adjusted_rand_score(truth, pred)),
                n_clones_used=int(len(set(pred))), secs=secs)


rows = []
# --- trivial baselines ---
maj = np.zeros((m, K)); maj[:, list(clones).index(vc.index[0])] = 1.0
rows.append(score(maj, "majority clone (trivial)"))
prior = np.tile(A.mean(0), (m, 1))
rows.append(score(prior, "cell-side prior (trivial)"))

# --- competitors ---
for mod, fn, nm in [("tangram_run", "run", "Tangram"), ("destvi_run", "run", "DestVI"),
                    ("scdot_run", "run", "scDOT"),
                    ("baselines", "run_ot", "plain OT (expression)"),
                    ("baselines", "run_centroid", "nearest centroid")]:
    try:
        import importlib
        M = importlib.import_module(f"bench.{mod}")
        t0 = time.time()
        P = getattr(M, fn)(X_sc, X_st, cell_clone, clones, xy=d["xy"], seed=0)
        rows.append(score(P, nm, time.time() - t0))
        print(f"  {nm}: done")
    except Exception as e:
        print(f"  {nm}: FAILED -- {type(e).__name__}: {str(e)[:120]}")
        rows.append(dict(method=nm, acc=np.nan, bal_acc=np.nan, ari=np.nan,
                         n_clones_used=0, secs=np.nan))

# --- SpOTr ---
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)


def spotr(alpha, tau=None):
    def build(gam):
        return build_perclade_alpha_cost_v2(jnp.full((K,), alpha, dtype=jnp.float32), Cf, C_T,
                                            C_S, a, b, gam, omega, sF, sL, equalize="measured")
    t0 = time.time()
    if tau is None:
        g = g0
        for _ in range(J):
            g, _u = S.sinkhorn_unrolled_safe(build(g), a, b, EPS, T)
    else:
        g, _uv = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=tau)
    g = np.asarray(g)
    P = g.T @ A
    return P / np.maximum(P.sum(1, keepdims=True), 1e-12), time.time() - t0


for alpha, tau, nm in [(0.0, None, "SpOTr alpha=0 (no tree)"),
                       (0.30, None, "SpOTr tree, balanced"),
                       (0.0, 0.01, "SpOTr alpha=0, semi-relaxed"),
                       (0.45, 0.01, "SpOTr tree + semi-relaxed")]:
    P, s = spotr(alpha, tau)
    rows.append(score(P, nm, s)); print(f"  {nm}: done")

r = pd.DataFrame(rows).sort_values("bal_acc", ascending=False)
print(f"\n{'method':<32}{'accuracy':>10}{'bal.acc':>10}{'ARI':>9}{'clones':>8}{'sec':>8}")
print("-" * 77)
for _, x in r.iterrows():
    f = lambda v, w, dd=4: (f"{v:>{w}.{dd}f}" if np.isfinite(v) else f"{'--':>{w}}")
    print(f"{x.method:<32}{f(x.acc,10)}{f(x.bal_acc,10)}{f(x.ari,9)}"
          f"{int(x.n_clones_used):>8}{f(x.secs,8,1)}")
print("-" * 77)
mj = r[r.method.str.startswith('majority')].iloc[0]
print(f"\nmajority baseline: accuracy {mj.acc:.4f}, balanced accuracy {mj.bal_acc:.4f}")
print("balanced accuracy is the honest column -- a predictor that hugs the clone prior")
print("cannot win it, and with C4 at 40% of bins raw accuracy rewards exactly that.")
r.to_csv(HERE / "compare_all.csv", index=False)
