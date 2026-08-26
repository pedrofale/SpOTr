"""Q7: does the tree correct confusable expression, and does per-clade alpha pay?

Design (verified in simulate_confusable.py): expression identifies a clade's confusable PAIR
at 0.877 but picks the right member of that pair at 0.566 (chance 0.500), and the pair's
territories sit 1.8 layout units apart. So expression systematically mis-maps about half of
each confusable clade into its partner's territory. That is the error the tree is supposed
to fix, by forcing the clade to move as one.

Two questions:
  Q7a  does a global alpha > 0 fix it?     (pooling)
  Q7b  does per-clade alpha beat it?       (pooling is right for compact clades,
                                            wrong for dispersed ones)

Headline metric is PAIR RESOLUTION -- among cells whose clade has a confusable partner, the
fraction mapped into the correct member's territory. That isolates exactly the ambiguity the
tree is meant to resolve. Reported alongside overall clade accuracy and expected distance.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)
f = HERE / "sim_conf.npz"
if time.time() - f.stat().st_mtime > 86400:
    raise SystemExit("REFUSING: sim_conf.npz looks stale.")
d = np.load(f, allow_pickle=True)

xy_sc, xy_st = d["xy_sc"], d["xy_st"]
diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))
_D = cdist(xy_sc, xy_st) / diam
cell_clade, spot_clade = d["cell_clade"], d["spot_clade"]
clades, is_disp, pair_id = d["clades"], d["is_dispersed"], d["pair_id"]
K = len(clades)
cidx = np.array([list(clades).index(c) for c in cell_clade])
disp_cell = is_disp[cidx]
has_pair = pair_id[cidx] >= 0

A = np.zeros((len(cell_clade), K), dtype=np.float32)
A[np.arange(len(cell_clade)), cidx] = 1.0
asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
omega = jnp.asarray(A)
g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
L = np.asarray(S.compute_Lgw(C_T, C_S, a, b, g0))
print(f"[gw] |L|/|F| = {np.abs(L).mean()/np.abs(np.asarray(Cf)).mean():.3e}")

spot_ci = np.array([list(clades).index(c) for c in spot_clade])


def solve(ak):
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                         Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                         equalize="measured")
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    return np.asarray(g)


def M(g):
    if not np.isfinite(g).all():
        return dict(pair=np.nan, acc=np.nan, acc_c=np.nan, acc_d=np.nan, ed=np.nan)
    j = g.argmax(1)
    ok = spot_ci[j] == cidx
    # pair resolution: among cells with a confusable partner, did we land on the right member?
    sel = has_pair & np.isin(spot_ci[j], np.where(pair_id >= 0)[0])
    same_pair = pair_id[spot_ci[j]] == pair_id[cidx]
    denom = has_pair & same_pair
    w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
    return dict(pair=ok[denom].mean() if denom.sum() else np.nan,
                acc=ok.mean(), acc_c=ok[~disp_cell].mean(), acc_d=ok[disp_cell].mean(),
                ed=(w * _D).sum(1).mean())


GRID = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
rows = [dict(alpha=v, **M(solve(np.full(K, v)))) for v in GRID]
r = pd.DataFrame(rows)
print(f"\n{'alpha':>6} {'PAIR resol':>11} {'acc clade':>10} {'compact':>9} {'dispersed':>10} {'E-dist':>8}")
print("-" * 60)
for _, x in r.iterrows():
    print(f"{x.alpha:>6.2f} {x.pair:>11.4f} {x.acc:>10.4f} {x.acc_c:>9.4f} {x.acc_d:>10.4f} {x.ed:>8.4f}")
ib = int(r.pair.idxmax())
bv = float(r.alpha[ib])
print(f"\nQ7a  pair resolution {r.pair[0]:.4f} (alpha=0) -> {r.pair[ib]:.4f} (alpha={bv})  "
      f"{r.pair[ib]-r.pair[0]:+.4f}")
print(f"     compact-clade accuracy {r.acc_c[0]:.4f} -> {r.acc_c[ib]:.4f}  {r.acc_c[ib]-r.acc_c[0]:+.4f}")
print(f"     dispersed-clade acc.   {r.acc_d[0]:.4f} -> {r.acc_d[ib]:.4f}  {r.acc_d[ib]-r.acc_d[0]:+.4f}"
      f"   <- tree should HURT these")

need = (~is_disp).astype(float)                  # 1 = compact -> wants HIGH alpha
GV = M(solve(np.full(K, bv)))
print(f"\n{'setting':<42}{'PAIR':>9}{'acc':>9}{'compact':>9}{'disp':>9}{'E-dist':>9}")
print("-" * 87)
best, cfg, BV = -np.inf, None, None
for lo, hi in [(0.0, bv), (0.0, 0.70), (0.0, 0.85), (0.05, 0.70), (0.10, 0.85), (0.20, 0.85)]:
    v = M(solve(lo + (hi - lo) * need))
    print(f"{'ORACLE  disp=' + f'{lo:.2f}' + '  compact=' + f'{hi:.2f}':<42}"
          f"{v['pair']:>9.4f}{v['acc']:>9.4f}{v['acc_c']:>9.4f}{v['acc_d']:>9.4f}{v['ed']:>9.4f}")
    if v["pair"] > best:
        best, cfg, BV = v["pair"], (lo, hi), v
lo, hi = cfg
SH = [M(solve(lo + (hi - lo) * rng.permutation(need))) for _ in range(8)]
IV = M(solve(lo + (hi - lo) * (1 - need)))
print("-" * 87)
for nm, v in [(f"best GLOBAL alpha = {bv}", GV), (f"ORACLE per-clade (d={lo:.2f}, c={hi:.2f})", BV),
              ("control: shuffled (mean of 8)", {k: np.nanmean([s[k] for s in SH]) for k in GV}),
              ("control: INVERTED", IV)]:
    print(f"{nm:<42}{v['pair']:>9.4f}{v['acc']:>9.4f}{v['acc_c']:>9.4f}{v['acc_d']:>9.4f}{v['ed']:>9.4f}")
print("-" * 87)
for c in ["pair", "acc"]:
    sm = np.nanmean([s[c] for s in SH]); ss = np.nanstd([s[c] for s in SH])
    print(f"Q7b  {c:<5} oracle {BV[c]:.4f}  vs global {GV[c]:.4f} ({BV[c]-GV[c]:+.4f})  "
          f"shuffled {sm:.4f}+-{ss:.4f} z={(BV[c]-sm)/max(ss,1e-9):+.1f}  inverted {IV[c]:.4f}")
r.to_csv(HERE / "q7_confusable_sweep.csv", index=False)
pd.DataFrame([dict(metric=c, oracle=BV[c], glob=GV[c],
                   shuf=np.nanmean([s[c] for s in SH]), shuf_sd=np.nanstd([s[c] for s in SH]),
                   inv=IV[c]) for c in GV]).to_csv(HERE / "q7_confusable.csv", index=False)
