"""Does the optimal alpha vary by REGION, and is that variation predictable WITHOUT labels?

Today's result: alpha_k is a clade x region property, and the clade index is the wrong one.
The natural fix is to index alpha by tissue region instead. That is only useful if:

  (1) the optimal alpha genuinely varies across regions of one section, and
  (2) that variation correlates with something measurable on the SPOT side alone -- no
      labels, no ground truth, no mapping -- because that is all the real data offers.

Cheap to test: one OT solve per alpha per section, then score restricted to the cells whose
TRUE position (xy_sc) falls in each region.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from petracer_common import Sec, GRID_G, SECS

R = 4
rows = []
for name in SECS:
    S_ = Sec(name)
    d = np.load(HERE / f"petracer_{name}.npz", allow_pickle=True)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    X_st = np.asarray(d["X_st"], np.float32)

    km = KMeans(R, n_init=10, random_state=0).fit(xy_st)
    reg_st = km.labels_
    reg_sc = km.predict(xy_sc)                      # cells labelled by their TRUE position

    # one solve per alpha, reused by every region
    G = {v: S_.solve(np.full(S_.K, v)) for v in GRID_G}
    cols = np.arange(S_.m)

    for r in range(R):
        cm = reg_sc == r
        if cm.sum() < 40: continue
        acc = {}
        for v, g in G.items():
            j = cols[np.asarray(g)[:, cols].argmax(1)]
            acc[v] = float(S_.COMP[j[cm], S_.cidx[cm]].mean())
        best = max(acc, key=lambda k: acc[k])

        # ---- observables computed on SPOTS ONLY, no labels, no mapping ----
        sm = reg_st == r
        P, E = xy_st[sm], X_st[sm]
        En = (E - E.mean(0)) / (E.std(0) + 1e-8)
        k = min(8, len(P) - 1)
        nb = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
        Ez = En / (np.linalg.norm(En, axis=1, keepdims=True) + 1e-9)
        C = Ez @ Ez.T
        near = float(np.mean([C[i, nb[i]].mean() for i in range(len(P))]))
        iu = np.triu_indices(len(P), 1)
        far = float(C[iu].mean())
        expr_coh = near - far                       # local expression coherence
        # expression-vs-distance decay (Mantel-style, spots only)
        Dsp = np.linalg.norm(P[:, None] - P[None, :], axis=2)
        decay = float(-pearsonr(Dsp[iu], C[iu])[0])
        rows.append(dict(section=name, region=r, n_cells=int(cm.sum()), n_spots=int(sm.sum()),
                         best_alpha=best, acc_best=acc[best], acc_a0=acc[0.0],
                         gain=acc[best] - acc[0.0], expr_coh=expr_coh, decay=decay,
                         **{f"acc_{v}": acc[v] for v in GRID_G}))
        print(f"  {name} r{r}: {cm.sum():>4} cells / {sm.sum():>3} spots | best alpha {best:<5} "
              f"acc {acc[best]:.4f} (a=0 {acc[0.0]:.4f}) | expr_coh {expr_coh:+.4f} decay {decay:+.4f}",
              flush=True)

t = pd.DataFrame(rows)
t.to_csv(HERE / "petracer_region_alpha.csv", index=False)
print("\n" + "=" * 78)
print("(1) DOES THE OPTIMAL ALPHA VARY BY REGION?")
for s, g in t.groupby("section"):
    print(f"  {s}: best alpha per region {list(g.best_alpha)}  (section-wide best "
          f"{max(GRID_G, key=lambda v: g[f'acc_{v}'].mean())})")
print(f"  spread of best alpha within a section: "
      f"{t.groupby('section').best_alpha.apply(lambda x: x.max()-x.min()).to_dict()}")

print("\n(2) IS IT PREDICTABLE FROM SPOT-SIDE OBSERVABLES?")
for v in ["expr_coh", "decay"]:
    r_, p_ = pearsonr(t[v], t.best_alpha); rho, pr = spearmanr(t[v], t.best_alpha)
    print(f"  {v:<10} vs best alpha: pearson r={r_:+.3f} (p={p_:.3f})  "
          f"spearman rho={rho:+.3f} (p={pr:.3f})   n={len(t)}")
    # within-section (removes the section-level effect)
    ws = []
    for s, g in t.groupby("section"):
        if g[v].std() > 0 and g.best_alpha.std() > 0:
            ws.append(spearmanr(g[v], g.best_alpha)[0])
    print(f"             within-section spearman: {[f'{x:+.2f}' for x in ws]} "
          f"mean {np.mean(ws):+.3f}" if ws else "             within-section: undefined")

print("\n(3) HEADROOM: region-wise oracle alpha vs one global alpha")
for s, g in t.groupby("section"):
    bg = max(GRID_G, key=lambda v: (g[f"acc_{v}"] * g.n_cells).sum() / g.n_cells.sum())
    glob = (g[f"acc_{bg}"] * g.n_cells).sum() / g.n_cells.sum()
    reg = (g.acc_best * g.n_cells).sum() / g.n_cells.sum()
    print(f"  {s}: global alpha={bg} -> {glob:.4f} | region-wise oracle -> {reg:.4f} "
          f"| headroom {reg-glob:+.4f}")
