import pandas as pd, numpy as np
ROOT="/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer"
LIN=f"{ROOT}/Tumor_Immune_Evasion_eTRACER/LineageTracing/spatial/spatial_lineage_LE.csv"
SC=f"{ROOT}/Tumor_Immune_Evasion_eTRACER/LineageTracing/single_cell/sc_cell_lineage.csv"

md=pd.read_csv(f"{ROOT}/recon/LE_bin_index.csv", dtype={"bc":str})
xs=np.sort(md.x.unique()); ys=np.sort(md.y.unique())
print("grid: %d x-vals * %d y-vals = %d lattice sites; occupied %d (%.1f%%) -> tissue mask, not full rect"
      %(len(xs),len(ys),len(xs)*len(ys),len(md),100*len(md)/(len(xs)*len(ys))))
print("x on regular 50-step lattice:", np.array_equal(xs, np.arange(xs.min(),xs.max()+1,50)))
print("y on regular 50-step lattice:", np.array_equal(ys, np.arange(ys.min(),ys.max()+1,50)))

lin=pd.read_csv(LIN,dtype=str)
sc=pd.read_csv(SC,dtype=str)
sp_cl=set(lin.lineageGrp); sc_cl=set(sc.lineageGrp)
print("\nspatial LE lineageGrp: %d unique"%len(sp_cl))
print("sc lineageGrp: %d unique"%len(sc_cl))
print("intersection: %d"%len(sp_cl&sc_cl))
print("spatial-only: %d %s"%(len(sp_cl-sc_cl), sorted(sp_cl-sc_cl)[:20]))
import os
tre=f"{ROOT}/Tumor_Immune_Evasion_eTRACER/LineageTracing/single_cell/sc_trees_LE"
trees={f.split('.')[0] for f in os.listdir(tre) if f.endswith('.treefile')}
print("clones with sc treefile: %d"%len(trees))
print("spatial LE clones WITH a tree: %d"%len(sp_cl&trees))
n_bins_with_tree=lin.lineageGrp.isin(trees).sum()
print("labelled bins whose clone has a tree: %d / %d"%(n_bins_with_tree,len(lin)))

# matrix orientation check: read header + max col index
with open(f"{ROOT}/export/spatial_LE/matrix.mtx") as f:
    f.readline(); dims=f.readline().split()
print("\nmtx dims: %s rows(genes) x %s cols(bins), %s nnz"%tuple(dims))
print("features.tsv lines:", sum(1 for _ in open(f"{ROOT}/export/spatial_LE/features.tsv")))
print("barcodes.tsv lines:", sum(1 for _ in open(f"{ROOT}/export/spatial_LE/barcodes.tsv")))
