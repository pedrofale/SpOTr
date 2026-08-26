"""Re-select cells so the clone composition MATCHES the spatial composition.

The first run sampled ~70 cells per clone while the bins are 84% C4/C2/C3. Balanced OT
forces each cell to carry equal mass, so an equal-per-clone reference imposes an equal-per-clone
spot composition -- and the GW term then enforces that wrong composition through the tree's
block geometry. This variant removes that confound.
"""
from pathlib import Path
import numpy as np, pandas as pd
from ete3 import Tree

HERE = Path(__file__).resolve().parent
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
TOTAL = 1500
rng = np.random.default_rng(0)

matched = pd.read_csv(HERE / "sc_cells_LE_matched.csv")
bins_ = pd.read_csv(HERE / "sel_bins.csv")
target = bins_.lineageGrp.value_counts(normalize=True)

sc_trees = {f.name.split("-")[0]: f for f in (B / "single_cell/sc_trees_LE").glob("*.phy.treefile")}
rows = []
for clone, frac in target.items():
    g = matched[matched.lineageGrp.astype(str) == clone]
    leaves = {l.name for l in Tree(str(sc_trees[clone]), format=1).iter_leaves()}
    g = g[g.cellBC.isin(leaves)]
    want = max(8, int(round(frac * TOTAL)))
    take = g if len(g) <= want else g.sample(want, random_state=0)
    rows.append(take.assign(clone=clone))
cells = pd.concat(rows, ignore_index=True)
print(f"[matched] {len(cells)} cells, {cells.clone.nunique()} clones")
comp = cells.clone.value_counts(normalize=True)
cmp = pd.DataFrame({"cells": comp, "bins": target}).fillna(0).sort_values("bins", ascending=False)
print(cmp.head(8).to_string(float_format=lambda v: f"{v:.3f}"))
print(f"[matched] composition TVD = {0.5*np.abs(cmp.cells-cmp.bins).sum():.3f}  (was ~0.63)")
cells.to_csv(HERE / "sel_cells.csv", index=False)
pd.Series(cells.matrix_id.values).to_csv(HERE / "sel_cell_ids.txt", index=False, header=False)
