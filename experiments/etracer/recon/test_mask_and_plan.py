import re
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B=HERE/"Tumor_Immune_Evasion_eTRACER"/"LineageTracing"
sb=[l.strip() for l in open(HERE/"export/spatial_LE/barcodes.tsv") if l.strip()]
sp=pd.read_csv(B/"spatial/spatial_lineage_LE.csv")
sp["CellID"]=sp.CellID.astype(str)
print("direct CellID -> spatial barcodes.tsv hits:", sp.CellID.isin(set(sb)).sum(), "/", len(sp))
md=pd.read_csv(HERE/"export/spatial_LE/metadata.csv", index_col=0)
md.index=[str(int(float(i))) if str(i).replace(".","",1).isdigit() else str(i) for i in md.index]
md["xy"]=md.x.astype(int).astype(str)+"_"+md.y.astype(int).astype(str)
sp["xy"]=sp.x.astype(int).astype(str)+"_"+sp.y.astype(int).astype(str)
sp["mid"]=sp.xy.map(dict(zip(md.xy,md.index)))
print("via xy -> matrix id hits:", sp.mid.notna().sum(), "/", len(sp), "; md xy unique:", md.xy.nunique(),"/",len(md))
lab=sp.dropna(subset=["lineageGrp"]).copy(); lab["lineageGrp"]=lab.lineageGrp.astype(str)
df=pd.read_csv(HERE/"recon/usable_cells_per_clone.csv")
shared=set(df.clone)
m=lab[lab.lineageGrp.isin(shared) & lab.mid.notna()]
print(f"MASK: bins labelled & in shared-45 & resolving to a matrix column = {len(m)}  ({100*len(m)/35835:.1f}% of 35,835)")

# --- sampling plan feasibility
tgt=m.lineageGrp.value_counts(normalize=True)
av=df.set_index("clone").usable
u=[c for c in tgt.index]
feas=min(av[c]/tgt[c] for c in u)
print(f"\nComposition-matched sampling: exact-proportional feasible total = {int(feas)} cells "
      f"(binding clone: {min(u,key=lambda c: av[c]/tgt[c])}, usable={av[min(u,key=lambda c: av[c]/tgt[c])]}, target frac={tgt[min(u,key=lambda c: av[c]/tgt[c])]:.4f})")
print(f"Use-everything: {int(av[u].sum())} usable cells; TVD vs bin composition = "
      f"{0.5*np.abs((av[u]/av[u].sum()).reindex(tgt.index).fillna(0)-tgt).sum():.3f}")
# per-clone ratio table
r=pd.DataFrame({"sp_bins":m.lineageGrp.value_counts(),"bin_frac":tgt,"usable":av.reindex(tgt.index),
                "cell_frac":(av.reindex(tgt.index)/av.reindex(tgt.index).sum())})
r["cells_per_bin"]=r.usable/r.sp_bins
print("\ntop over/under-represented (cell_frac / bin_frac):")
r["ratio"]=r.cell_frac/r.bin_frac
print(r.sort_values("ratio",ascending=False).head(8).to_string())
print(r.sort_values("ratio").head(8).to_string())
r.to_csv(HERE/"recon/clone_composition_LE.csv")
for cap in (2000,5000,10000,14831):
    t=int(min(cap,feas)); print(f"  cap={cap}: exact-proportional draw = {t}")
