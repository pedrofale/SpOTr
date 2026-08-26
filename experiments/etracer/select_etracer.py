"""Pick the cells and bins for the eTRACER SpOTr run, and write id lists for R subsetting."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from ete3 import Tree

HERE = Path(__file__).resolve().parent
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
CELLS_PER_CLONE = 70          # cap; small clones contribute what they have
N_BINS = 2000                 # random sample preserving the natural spatial composition
rng = np.random.default_rng(0)

sc_trees = {f.name.split("-")[0]: f for f in (B / "single_cell/sc_trees_LE").glob("*.phy.treefile")}
sp_trees = {f.name.split("-")[0] for f in (B / "spatial/spatial_trees_LE").glob("*.phy.treefile")}
shared = sorted(set(sc_trees) & sp_trees, key=lambda c: int(c[1:]))

matched = pd.read_csv(HERE / "sc_cells_LE_matched.csv")
by_clone = {c: g for c, g in matched.groupby(matched.lineageGrp.astype(str))}

rows = []
for c in shared:
    g = by_clone.get(c)
    if g is None:
        continue
    leaves = {l.name for l in Tree(str(sc_trees[c]), format=1).iter_leaves()}
    g = g[g.cellBC.isin(leaves)]                       # must be in the tree
    if len(g) < 8:
        continue
    take = g if len(g) <= CELLS_PER_CLONE else g.sample(CELLS_PER_CLONE, random_state=0)
    rows.append(take.assign(clone=c))
cells = pd.concat(rows, ignore_index=True)
print(f"[select] {len(cells)} cells across {cells.clone.nunique()} clones "
      f"(cap {CELLS_PER_CLONE}/clone)")
print("        ", cells.clone.value_counts().sort_index().to_dict())

sp = pd.read_csv(B / "spatial/spatial_lineage_LE.csv").dropna(subset=["lineageGrp", "x", "y"])
sp["lineageGrp"] = sp.lineageGrp.astype(str)
sp = sp[sp.lineageGrp.isin(set(cells.clone))]
if len(sp) > N_BINS:
    sp = sp.sample(N_BINS, random_state=0)
print(f"[select] {len(sp)} spatial bins across {sp.lineageGrp.nunique()} clones "
      f"(natural composition preserved)")
print("        top:", sp.lineageGrp.value_counts().head(6).to_dict())

cells.to_csv(HERE / "sel_cells.csv", index=False)
sp.to_csv(HERE / "sel_bins.csv", index=False)
pd.Series(cells.matrix_id.values).to_csv(HERE / "sel_cell_ids.txt", index=False, header=False)
pd.Series(sp.CellID.astype(str).values).to_csv(HERE / "sel_bin_ids.txt", index=False, header=False)
print("[select] wrote sel_cells.csv / sel_bins.csv + id lists")
