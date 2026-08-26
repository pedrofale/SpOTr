"""How reliably does a held-out anchor loss select alpha on real eTRACER?

Two runs disagreed qualitatively on the SAME couplings: with a 407-bin held-out set the loss
had an interior minimum at alpha = 0.075, with a 308-bin one it was monotone and picked 0.
Since solve() ignores labels, the couplings are identical -- so the disagreement comes purely
from which labelled bins land in the held-out set. That is worth quantifying: if the argmin
moves around under resampling, the selection is not usable however the split is drawn.

Couplings are computed once, then the loss curve is re-evaluated on many random held-out
subsets at several sizes.
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
GRID = [0.0, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.45]
d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
cell_clone = np.asarray(d["cell_clone"]).astype(str)
bin_clone = np.asarray(d["bin_clone"]).astype(str)
clones = np.asarray(d["clones"]).astype(str)
A = np.asarray(d["A"], np.float32); is_lab = np.asarray(d["is_labelled"], bool)
K, m = len(clones), len(bin_clone)
bidx = np.array([list(clones).index(c) if c in list(clones) else -1 for c in bin_clone])
Y = np.zeros((m, K), np.float32); Y[np.arange(m)[bidx >= 0], bidx[bidx >= 0]] = 1.0

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(np.asarray(d["X_sc"], np.float32)),
                                         ad.AnnData(np.asarray(d["X_st"], np.float32)),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
P, ACC = {}, {}
lab_idx = np.where(is_lab)[0]
for av in GRID:
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T, C_S,
                                         a, b, g, omega, sF, sL, equalize="measured")
        g, _u = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    gn = np.asarray(g)
    p = gn.T @ A
    P[av] = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    j = lab_idx[gn[:, lab_idx].argmax(1)]
    ACC[av] = float((bin_clone[j] == cell_clone).mean())
SE = {av: ((Y - P[av]) ** 2).sum(1) for av in GRID}      # per-bin squared error

rng = np.random.default_rng(0)
print(f"accuracy on all labelled bins: " + "  ".join(f"{v}:{ACC[v]:.4f}" for v in GRID))
print(f"  accuracy-optimal alpha = {max(ACC, key=lambda k: ACC[k])}\n")
print(f"{'held-out size':>14}{'argmin distribution over 400 resamples':>46}{'  frac > 0':>11}")
print("-" * 72)
rows = []
for frac in [0.6, 0.4, 0.3, 0.2, 0.1]:
    n = int(round(frac * len(lab_idx)))
    picks = []
    for _ in range(400):
        sel = rng.choice(lab_idx, size=n, replace=False)
        curve = np.array([SE[av][sel].mean() for av in GRID])
        picks.append(GRID[int(curve.argmin())])
    picks = np.array(picks)
    vc = pd.Series(picks).value_counts(normalize=True).sort_index()
    desc = " ".join(f"{k}:{100*v:.0f}%" for k, v in vc.items())
    print(f"{n:>10} bins  {desc:>44}{100*(picks > 0).mean():>10.0f}%")
    rows.append(dict(n=n, frac_gt0=float((picks > 0).mean()),
                     modal=float(pd.Series(picks).mode()[0])))
print("-" * 72)
print("accuracy-optimal alpha is 0.30; a usable selector should put its mass near there.")
pd.DataFrame(rows).to_csv(HERE / "split_stability.csv", index=False)
