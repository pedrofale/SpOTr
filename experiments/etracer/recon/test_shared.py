import re, collections
from pathlib import Path
import pandas as pd
from ete3 import Tree
HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE/"Tumor_Immune_Evasion_eTRACER"/"LineageTracing"

sc_trees = {f.name.split("-")[0]: f for f in (B/"single_cell/sc_trees_LE").glob("*.phy.treefile")}
sp_trees = {f.name.split("-")[0] for f in (B/"spatial/spatial_trees_LE").glob("*.phy.treefile")}
sp_raw = pd.read_csv(B/"spatial/spatial_lineage_LE.csv")
print("spatial_lineage_LE.csv raw rows:", len(sp_raw), "cols", list(sp_raw.columns))
print("  nulls:", sp_raw.isna().sum().to_dict())
sp = sp_raw.dropna(subset=["lineageGrp"]).copy()
sp["lineageGrp"]=sp.lineageGrp.astype(str)
print("  rows with a lineageGrp:", len(sp), " distinct clones:", sp.lineageGrp.nunique())
sp2 = sp_raw.dropna(subset=["lineageGrp","x","y"]).copy(); sp2["lineageGrp"]=sp2.lineageGrp.astype(str)
print("  rows with lineageGrp AND x,y:", len(sp2), "distinct clones:", sp2.lineageGrp.nunique())

A = set(sc_trees) & set(sp.lineageGrp)
print(f"\nDEF-A  sc_trees_LE ({len(sc_trees)}) INTERSECT spatial_lineage_LE labels ({sp.lineageGrp.nunique()}) = {len(A)}")
B2 = set(sc_trees) & sp_trees
print(f"DEF-B  sc_trees_LE INTERSECT spatial_trees_LE ({len(sp_trees)}) = {len(B2)}")
lin = pd.read_csv(B/"single_cell/sc_cell_lineage.csv"); lin["lineageGrp"]=lin.lineageGrp.astype(str)
C = set(lin.lineageGrp) & set(sp.lineageGrp)
print(f"DEF-C  sc_cell_lineage clones ({lin.lineageGrp.nunique()}) INTERSECT spatial_lineage_LE labels = {len(C)}")
print("\nDEF-A minus DEF-B (have sc tree + spatial bins but NO spatial tree):", sorted(A-B2, key=lambda c:int(c[1:])))
print("bins those carry:", {c:int((sp.lineageGrp==c).sum()) for c in sorted(A-B2,key=lambda c:int(c[1:]))})
print("\nspatial_trees_LE clones with NO sc tree:", sorted(sp_trees-set(sc_trees)))

# leaf that is not in lineage csv
allbc=set(lin.cellBC)
for c in list(sorted(B2, key=lambda x:int(x[1:])))[:5]:
    lv={l.name for l in Tree(str(sc_trees[c]),format=1).iter_leaves()}
    print(c,"extra leaves:", sorted(lv-allbc))
