from pathlib import Path
import pandas as pd
from ete3 import Tree
HERE=Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B=HERE/"Tumor_Immune_Evasion_eTRACER"/"LineageTracing"
sc={f.name.split("-")[0]:f for f in (B/"single_cell/sc_trees_LE").glob("*.phy.treefile")}
df=pd.read_csv(HERE/"recon/usable_cells_per_clone.csv")
nsyn=0; dup=0; fname_n=[]
for c in df.clone:
    t=Tree(str(sc[c]),format=1); lv=[l.name for l in t.iter_leaves()]
    nsyn += ("synthetic" in lv); dup += len(lv)-len(set(lv))
    fname_n.append((c, int(sc[c].name.split("-")[1].split(".")[0]), len(lv)-("synthetic" in lv)))
print(f"trees with a 'synthetic' outgroup leaf: {nsyn}/{len(df)}; duplicate leaf names: {dup}")
bad=[(c,n,m) for c,n,m in fname_n if n!=m]
print(f"filename suffix (-N) == real leaf count for {len(fname_n)-len(bad)}/{len(fname_n)} trees; mismatches: {bad[:5]}")
# also: sc_trees_LE per-clone .csv character matrices
import itertools
c0=df.clone.iloc[0]
p=list((B/"single_cell/sc_trees_LE").glob(f"{c0}-*.csv"))[0]
print("example clone csv:", p.name); print(open(p).readline().strip()[:150])
