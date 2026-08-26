"""Q4: does per-clade phi have real headroom, selected and scored on ASSIGNMENT accuracy?

Q3 left two metrics disagreeing. Expected transport distance narrowly preferred a uniform
config; every assignment metric preferred the oracle. The two are measuring different
things: raising phi sharpens the coupling for a clade, which improves the argmax but costs
expected distance whenever the sharpened guess is wrong. SpOTr's claim is about WHERE a
cell is mapped, so assignment accuracy is the metric that matches the claim -- and the
decisive quantity is subclade accuracy on the TREE-FINE clades, the ones whose fine-level
position only the tree can resolve.

Selection and scoring both on subclade accuracy. Controls: 10 shuffled per-clade vectors
(same values, permuted across clades) and the inverted assignment.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)
d = np.load(HERE / "sim_cf.npz", allow_pickle=True)
xy_sc, xy_st = d["xy_sc"], d["xy_st"]
diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))
_D = cdist(xy_sc, xy_st) / diam
cell_clade, cell_sub = d["cell_clade"], d["cell_sub"]
spot_sub, spot_clade = d["spot_sub"], d["spot_clade"]
clades, is_tf = d["clades"], d["is_tree_fine"]
K = len(clades)
tf = np.isin(cell_clade, clades[is_tf])

A = np.zeros((len(cell_clade), K), dtype=np.float32)
for i, c in enumerate(cell_clade):
    A[i, list(clades).index(c)] = 1.0
asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
omega, Omega = jnp.asarray(A), jnp.asarray(A @ A.T)
g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)


def solve(ak, pk):
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(
            jnp.asarray(np.asarray(ak, np.float32)), Cf, C_T, C_S, a, b, g, omega, sF, sL,
            use_decomp=True, Omega=Omega, phi_k=jnp.asarray(np.asarray(pk, np.float32)),
            equalize="measured")
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    return np.asarray(g)


def M(g):
    if not np.isfinite(g).all():
        return dict(acc_sub=np.nan, acc_sub_tf=np.nan, acc_sub_ef=np.nan,
                    ari_sub=np.nan, acc_clade=np.nan, edist=np.nan)
    j = g.argmax(1)
    ok = spot_sub[j] == cell_sub
    w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
    return dict(acc_sub=ok.mean(), acc_sub_tf=ok[tf].mean(), acc_sub_ef=ok[~tf].mean(),
                ari_sub=adjusted_rand_score(cell_sub, spot_sub[j]),
                acc_clade=(spot_clade[j] == cell_clade).mean(), edist=(w * _D).sum(1).mean())


AG = [0.0, 0.20, 0.35, 0.50, 0.70]
PG = [0.0, 0.25, 0.50, 0.75, 1.0]
best_g, gcfg = -np.inf, None
print("global (alpha,phi) grid -- SUBCLADE ACCURACY")
print(f"{'alpha':>6} " + "".join(f"{p:>9.2f}" for p in PG))
print("-" * 56)
for av in AG:
    row = []
    for pv in PG:
        v = M(solve(np.full(K, av), np.full(K, pv)))["acc_sub"]
        row.append(v)
        if v > best_g:
            best_g, gcfg = v, (av, pv)
    print(f"{av:>6.2f} " + "".join(f"{v:>9.4f}" for v in row))
ga, gp = gcfg
GV = M(solve(np.full(K, ga), np.full(K, gp)))
print(f"\nbest GLOBAL: alpha={ga} phi={gp}  subclade accuracy {best_g:.4f}\n")

need = is_tf.astype(float)
cols = ["acc_sub", "acc_sub_tf", "acc_sub_ef", "ari_sub", "acc_clade", "edist"]
H = f"{'setting':<44}" + "".join(f"{c:>12}" for c in cols)
print(H); print("-" * len(H))
best, cfg, BV = -np.inf, None, None
for lo, hi in [(0.0, 1.0), (0.0, 0.75), (0.25, 1.0), (0.25, 0.75), (0.5, 1.0), (0.0, 0.5)]:
    v = M(solve(np.full(K, ga), lo + (hi - lo) * need))
    print(f"{'ORACLE phi  ef=' + f'{lo:.2f}' + '  tf=' + f'{hi:.2f}':<44}"
          + "".join(f"{v[c]:>12.4f}" for c in cols))
    if v["acc_sub"] > best:
        best, cfg, BV = v["acc_sub"], (lo, hi), v
lo, hi = cfg
SH = [M(solve(np.full(K, ga), lo + (hi - lo) * rng.permutation(need))) for _ in range(10)]
IV = M(solve(np.full(K, ga), lo + (hi - lo) * (1 - need)))
print("-" * len(H))
for nm, v in [("best GLOBAL", GV), (f"ORACLE phi (ef={lo:.2f}, tf={hi:.2f})", BV),
              ("control: shuffled (mean of 10)", {c: np.nanmean([s[c] for s in SH]) for c in cols}),
              ("control: INVERTED", IV)]:
    print(f"{nm:<44}" + "".join(f"{v[c]:>12.4f}" for c in cols))
print("-" * len(H))
for c in ["acc_sub", "acc_sub_tf", "ari_sub"]:
    sm = np.nanmean([s[c] for s in SH]); ss = np.nanstd([s[c] for s in SH])
    print(f"{c:<14} oracle {BV[c]:.4f}  global {GV[c]:.4f} ({BV[c]-GV[c]:+.4f})  "
          f"shuffled {sm:.4f}+-{ss:.4f}  z={(BV[c]-sm)/max(ss,1e-9):+.1f}  "
          f"inverted {IV[c]:.4f}")
pd.DataFrame([dict(metric=c, oracle=BV[c], glob=GV[c],
                   shuf=np.nanmean([s[c] for s in SH]), shuf_sd=np.nanstd([s[c] for s in SH]),
                   inv=IV[c]) for c in cols]).to_csv(HERE / "q4_headroom.csv", index=False)
