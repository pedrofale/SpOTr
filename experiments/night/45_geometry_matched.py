"""GEOMETRY-MATCHED DEPTH TEST.

The concentric-strata result is confounded: the outer third is an annulus spanning the whole
tumour, the core a compact disc, so their separation distributions differ for reasons that have
nothing to do with lineage.  Fix it by comparing LOCAL NEIGHBOURHOODS THAT ARE THE SAME SHAPE AND
THE SAME SIZE at different depths:

  - centre a disc of fixed radius R on a cell
  - require at least k cells inside it, then subsample to EXACTLY k
    (so extent and cell count are both matched by construction)
  - measure corr(divergence age, separation) among those k cells
  - record the depth of the centre (obs['tumor_boundary_dist']) and regress

Controls reported alongside: the fraction of candidate neighbourhoods dropped for having too few
cells (a selection effect that would bite at the edge), the mean divergence age, and the ceiling
fraction, all as a function of depth.  Null shuffles the tree assignment within each neighbourhood,
which preserves that neighbourhood's geometry exactly.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, pycea
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent

R_UM, K = 260.0, 25
t = pycea.datasets.koblan25("tumor"); obs = t.obs
rng = np.random.default_rng(0)

rows, drops = [], []
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    names = np.asarray(d["cell_names"]).astype(str)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    bd = obs.loc[names, "tumor_boundary_dist"].to_numpy(float)
    fit = obs.loc[names, "fitness"].to_numpy(float)
    cap = CT.max()
    area = np.ptp(xy, 0).prod()
    print(f"{SEC}: {len(xy)} cells, density {len(xy)/area*1e6:.0f} per mm^2, "
          f"expected cells in a {R_UM:.0f} um disc: {len(xy)/area*np.pi*R_UM**2:.0f}")
    tree = cKDTree(xy)
    nb = tree.query_ball_point(xy, R_UM)
    for c, members in enumerate(nb):
        members = np.array(members)
        if len(members) < K:
            drops.append(dict(section=SEC, depth=bd[c], dropped=1))
            continue
        drops.append(dict(section=SEC, depth=bd[c], dropped=0))
        sel = rng.choice(members, K, replace=False)
        iu = np.triu_indices(K, 1)
        a = CT[np.ix_(sel, sel)][iu] / 2.0
        dd = np.linalg.norm(xy[sel][iu[0]] - xy[sel][iu[1]], axis=1)
        if a.std() < 1e-9 or dd.std() < 1e-9:
            continue
        r_obs = np.corrcoef(a, dd)[0, 1]
        nl = []
        for _ in range(8):
            pm = rng.permutation(K)
            an = CT[np.ix_(sel[pm], sel[pm])][iu] / 2.0
            if an.std() > 1e-9:
                nl.append(np.corrcoef(an, dd)[0, 1])
        rows.append(dict(section=SEC, depth=bd[c], fitness=fit[c], corr=r_obs,
                         null=float(np.mean(nl)) if nl else np.nan,
                         mean_age=a.mean(), frac_ceiling=float((a >= cap / 2 - 1e-9).mean()),
                         extent=dd.max()))

Rw = pd.DataFrame(rows); Dr = pd.DataFrame(drops)
Rw.to_csv(HERE / "geometry_matched.csv", index=False)

print(f"\n{len(Rw)} matched neighbourhoods (radius {R_UM:.0f} um, exactly {K} cells each)")
print(f"dropped for too few cells: {Dr.dropped.mean():.1%} overall")
print("\ndrop rate by depth tertile (a selection effect would show here):")
Dr["bin"] = pd.qcut(Dr.depth, 3, labels=["edge", "middle", "core"])
print(Dr.groupby("bin", observed=True).dropped.mean().round(3).to_string())

Rw["bin"] = pd.qcut(Rw.depth, 3, labels=["edge", "middle", "core"])
print("\nGEOMETRY-MATCHED RESULT")
print(Rw.groupby("bin", observed=True)[["corr", "null", "mean_age", "frac_ceiling",
                                        "extent", "fitness"]].mean().round(3).to_string())
print("\nper section (mean corr by depth tertile):")
print(Rw.pivot_table(index="section", columns="bin", values="corr", observed=True)
      .round(3).to_string())
print("\ncontinuous: spearman(corr, depth)")
for SEC in ["M3-4", "M3-1", "M3-2"]:
    g = Rw[Rw.section == SEC]
    s = spearmanr(g.depth, g["corr"])
    print(f"   {SEC}: rho = {s.statistic:+.3f}   p = {s.pvalue:.1e}   n = {len(g)}")
g = Rw
print(f"   pooled: rho = {spearmanr(g.depth, g['corr']).statistic:+.3f}")
print("\nNOTE: neighbourhoods overlap, so the p-values are anticonservative; the effect size and "
      "its consistency across sections are the evidence.")
