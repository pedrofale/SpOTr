"""Q1 on the coarse-to-fine testbed: does the tree improve the mapping?

ARI on clade identity cannot answer this. Expression pins the clade by construction
(balanced accuracy 1.000), so clade-ARI is saturated at alpha = 0 and a refiner has
nowhere to show up. The primary metric is SPATIAL ERROR -- the distance between a cell's
true position and the barycentre of its coupling row, in units of tissue diameter --
plus ARI at the SUBCLADE level, which is the level the tree is supposed to resolve.

Reported separately for tree-fine clades (only the tree resolves them) and expr-fine
clades (expression already resolves them, and their within-clade tree signal is noise).
If the account is right, alpha > 0 should cut spatial error on the tree-fine half and
raise it on the expr-fine half.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

EPS, T, J = 0.02, 50, 3
f = HERE / "sim_cf.npz"
if time.time() - f.stat().st_mtime > 86400:
    raise SystemExit(f"REFUSING: sim_cf.npz is {(time.time()-f.stat().st_mtime)/60:.0f} min old -- likely stale.")
d = np.load(f, allow_pickle=True)

xy_sc, xy_st = d["xy_sc"], d["xy_st"]
diam = np.linalg.norm(xy_st.max(0) - xy_st.min(0))
cell_clade, cell_sub = d["cell_clade"], d["cell_sub"]
spot_sub, spot_clade = d["spot_sub"], d["spot_clade"]
clades, is_tf = d["clades"], d["is_tree_fine"]
tf_cell = np.isin(cell_clade, clades[is_tf])

asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])

# is the GW term numerically alive? on a balanced ultrametric tree it is annihilated by
# double-centering and alpha is a no-op regardless of the design
g0 = np.asarray(a)[:, None] * np.asarray(b)[None, :]
L = np.asarray(S.compute_Lgw(C_T, C_S, a, b, g0))
print(f"[gw] |L|/|F| = {np.abs(L).mean() / np.abs(np.asarray(Cf)).mean():.3e}  "
      f"(GW term is {'ALIVE' if np.abs(L).mean() / np.abs(np.asarray(Cf)).mean() > 1e-3 else 'DEAD'})")



# Barycentre error ||E_j[xy_j] - xy_i|| is minimised by HEDGING: spreading mass over the
# region returns its centroid, which has low error, while committing to one spot is punished
# whenever it is wrong. By Jensen it systematically understates a sharp coupling. A refiner
# commits, so barycentre error is the wrong instrument here. err_exp = E_j||xy_j - xy_i|| is
# the proper transport-distance analogue and does not reward hedging; err_top1 scores the
# argmax spot only.
from scipy.spatial.distance import cdist as _cdist
_D = _cdist(xy_sc, xy_st) / diam


def metrics(g):
    g = np.asarray(g)
    if not np.isfinite(g).all():
        return dict(err=np.nan, err_tf=np.nan, err_ef=np.nan, ari_sub=np.nan, ari_clade=np.nan)
    w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
    e = (w * _D).sum(1)                              # expected distance, hedging-neutral
    j = g.argmax(1)
    t1 = _D[np.arange(len(j)), j]
    return dict(err=e.mean(), err_tf=e[tf_cell].mean(), err_ef=e[~tf_cell].mean(),
                err_bary=(np.linalg.norm(w @ xy_st - xy_sc, axis=1) / diam).mean(),
                err_top1=t1.mean(), err_top1_tf=t1[tf_cell].mean(),
                ari_sub=adjusted_rand_score(cell_sub, spot_sub[j]),
                ari_clade=adjusted_rand_score(cell_clade, spot_clade[j]))


GRID = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
rows = []
for av in GRID:
    g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=T, J_alt=J, alpha=av)
    rows.append(dict(alpha=av, **metrics(g)))
r = pd.DataFrame(rows)

base = _D.mean()
print(f"\nchance E-dist (uniform coupling) = {base:.4f}\n")
print(f"{'alpha':>6} {'E-dist':>8} {'tree-fine':>10} {'expr-fine':>10} "
      f"{'top1':>8} {'top1 tf':>8} {'bary':>8} {'ARIsub':>8} {'ARIcl':>7}")
print("-" * 82)
for _, x in r.iterrows():
    print(f"{x.alpha:>6.2f} {x.err:>8.4f} {x.err_tf:>10.4f} {x.err_ef:>10.4f} "
          f"{x.err_top1:>8.4f} {x.err_top1_tf:>8.4f} {x.err_bary:>8.4f} "
          f"{x.ari_sub:>8.4f} {x.ari_clade:>7.4f}")

i0 = 0
for col, nm in [("err", "ALL"), ("err_tf", "tree-fine"), ("err_ef", "expr-fine")]:
    ib = int(r[col].idxmin())
    print(f"\nspatial error {nm:<10} best alpha = {r.alpha[ib]:.2f}  "
          f"{r[col][ib]:.4f} vs {r[col][i0]:.4f} at alpha=0  "
          f"({100*(r[col][i0]-r[col][ib])/r[col][i0]:+.1f}% change)")
ib = int(r.ari_sub.idxmax())
print(f"ARI subclade        best alpha = {r.alpha[ib]:.2f}  "
      f"{r.ari_sub[ib]:.4f} vs {r.ari_sub[i0]:.4f} at alpha=0  ({r.ari_sub[ib]-r.ari_sub[i0]:+.4f})")
r.to_csv(HERE / "q1_coarse_fine.csv", index=False)
