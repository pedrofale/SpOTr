import pandas as pd, numpy as np, os, re
ROOT="/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer"
LT=f"{ROOT}/Tumor_Immune_Evasion_eTRACER/LineageTracing"

lin=pd.read_csv(f"{LT}/spatial/spatial_lineage_LE.csv", dtype={"CellID":str},
                keep_default_na=False, na_values=[])
print("lineage CSV rows:", len(lin))
print("  lineageGrp == 'NA' :", (lin.lineageGrp=="NA").sum())
lin_ok = lin[lin.lineageGrp!="NA"].copy()
print("  rows with a REAL clone:", len(lin_ok), "clones:", lin_ok.lineageGrp.nunique())

md=pd.read_csv(f"{ROOT}/recon/LE_bin_index.csv", dtype={"bc":str})
mset=set(md.cellid)
print("\nall 7568 CellIDs in matrix:", lin.CellID.isin(mset).all())
print("all 7351 real-clone CellIDs in matrix:", lin_ok.CellID.isin(mset).all())

real=set(lin_ok.CellID)
md["labelled"]=md.cellid.isin(real)
print("\nmatrix bins WITH real clone   :", int(md.labelled.sum()), "(%.2f%%)"%(100*md.labelled.mean()))
print("matrix bins WITHOUT real clone:", int((~md.labelled).sum()), "(%.2f%%)"%(100*(~md.labelled).mean()))
print("  of which: NA-in-csv %d, absent-from-csv %d"%(
      (lin.lineageGrp=="NA").sum(), len(md)-len(lin)))

# trees
tf=[f for f in os.listdir(f"{LT}/single_cell/sc_trees_LE") if f.endswith(".treefile")]
tree_clone={re.sub(r'-\d+$','',f.replace(".phy.treefile","")):f for f in tf}
print("\ntreefiles:", len(tf), "unique clone ids:", len(tree_clone))
sp=set(lin_ok.lineageGrp)
both=sp & set(tree_clone)
print("spatial LE clones:", len(sp), "| sc-tree clones:", len(tree_clone), "| BOTH:", len(both))
print("spatial-only (no tree):", sorted(sp-set(tree_clone)))
nb=lin_ok.lineageGrp.isin(both).sum()
print("labelled bins whose clone HAS a tree: %d / %d (%.1f%%)"%(nb,len(lin_ok),100*nb/len(lin_ok)))

md["clone"]=md.cellid.map(dict(zip(lin_ok.CellID,lin_ok.lineageGrp)))
md["clone_has_tree"]=md.clone.isin(both)
md[["bc","cellid","x","y","clone","clone_has_tree"]].to_csv(f"{ROOT}/recon/LE_bin_index.csv",index=False)
print("\nrewrote recon/LE_bin_index.csv with clone + clone_has_tree")
print("\nmask sizes for the loss:")
print("  all bins (C_space rows)          :", len(md))
print("  loss mask, any clone             :", int(md.clone.notna().sum()))
print("  loss mask, clone with sc tree    :", int(md.clone_has_tree.sum()))
