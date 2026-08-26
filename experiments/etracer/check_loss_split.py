"""Is the barcode loss monotone because it OVERFITS the labelled bins it is fitted on?

On real eTRACER the train barcode loss rises monotonically with alpha (0.01606 -> 0.01757) and
so selects alpha = 0, while test-bin accuracy peaks well above 0. Two explanations:

  overfitting   a flexible coupling can nail 619 specific train bins with expression alone;
                the tree term spreads mass by geometry, which fits those bins worse but
                generalises better. Then the TEST barcode loss should have an interior minimum.
  misalignment  the loss is simply not measuring what accuracy measures, in which case the
                test loss is monotone too.

Evaluating the same loss on held-out labelled bins separates the two.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.45]
rng = np.random.default_rng(0)
d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clone = np.asarray(d["cell_clone"]).astype(str)
bin_clone = np.asarray(d["bin_clone"]).astype(str)
clones = np.asarray(d["clones"]).astype(str)
A = np.asarray(d["A"], np.float32); is_lab = np.asarray(d["is_labelled"], bool)
K, m = len(clones), X_st.shape[0]

lab_idx = np.where(is_lab)[0]
train = np.zeros(m, bool)
for c in clones:
    ii = lab_idx[bin_clone[lab_idx] == c]
    if len(ii):
        train[rng.choice(ii, size=max(1, int(round(0.6 * len(ii)))), replace=False)] = True
test = is_lab & ~train
bidx = np.array([list(clones).index(c) if c in list(clones) else -1 for c in bin_clone])
Y = np.zeros((m, K), np.float32); Y[np.arange(m)[bidx >= 0], bidx[bidx >= 0]] = 1.0
Yj = jnp.asarray(Y)

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)


def loss_on(g, sel):
    p = np.asarray(g).T @ A
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return float((((Y - p) ** 2)[sel]).sum() / (sel.sum() * K))


rows = []
cols = np.where(test)[0]
for av in GRID:
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T, C_S,
                                         a, b, g, omega, sF, sL, equalize="measured")
        g, _u = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    gn = np.asarray(g)
    j = cols[gn[:, cols].argmax(1)]
    rows.append(dict(alpha=av, train_loss=loss_on(gn, train), test_loss=loss_on(gn, test),
                     test_acc=float((bin_clone[j] == cell_clone).mean())))
r = pd.DataFrame(rows)
print(f"{'alpha':>7}{'train loss':>13}{'TEST loss':>12}{'test acc':>11}")
print("-" * 43)
for _, x in r.iterrows():
    print(f"{x.alpha:>7.3f}{x.train_loss:>13.5f}{x.test_loss:>12.5f}{x.test_acc:>11.4f}")
print("-" * 43)
print(f"argmin train loss : alpha = {r.alpha[r.train_loss.idxmin()]}")
print(f"argmin TEST loss  : alpha = {r.alpha[r.test_loss.idxmin()]}")
print(f"argmax test acc   : alpha = {r.alpha[r.test_acc.idxmax()]}")
mono_tr = bool((r.sort_values('alpha').train_loss.diff().dropna() > 0).all())
mono_te = bool((r.sort_values('alpha').test_loss.diff().dropna() > 0).all())
print(f"\ntrain loss monotone increasing: {mono_tr}")
print(f"TEST  loss monotone increasing: {mono_te}")
print("\n-> " + ("OVERFITTING: held-out loss has an interior optimum, so a proper split makes "
                "the barcode anchor usable for selecting alpha"
                if not mono_te else
                "MISALIGNMENT: the held-out loss is monotone too, so this anchor cannot select "
                "alpha on this data however it is split"))
r.to_csv(HERE / "loss_train_test_split.csv", index=False)
