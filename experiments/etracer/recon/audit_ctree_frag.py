import re, numpy as np, pandas as pd
from pathlib import Path
from ete3 import Tree
HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
def clone_of(p): return re.sub(r"-\d+$", "", p.name.replace(".phy.treefile", ""))
tf = {clone_of(f): f for f in (B/"single_cell/sc_trees_LE").glob("*.phy.treefile")}
cells = pd.read_csv(HERE/"recon/audit_cells_order.csv")
C = np.load(HERE/"etracer_LE_full.npz")["C_tree"].astype(np.float64)

# 1. zero-distance distinct-cell pairs: are they real zero-length branches?
print("=== zero-distance pairs among distinct selected cells ===")
tot=0
for cl in ["C4","C28","C36","C61"]:
    g = cells[cells.clone==cl]; ii=g.index.to_numpy(); nm=g.cellBC.to_numpy()
    sub = C[np.ix_(ii,ii)]; a,b = np.where(np.triu(sub==0,1))
    tot += len(a)
    t = Tree(str(tf[cl]), format=1)
    lm = {l.name:l for l in t.iter_leaves()}
    x,y = nm[a[0]], nm[b[0]]
    anc = t.get_common_ancestor([lm[x],lm[y]])
    print(f" {cl}: {len(a)} pairs. example {x[-8:]} / {y[-8:]}: leaf dists {lm[x].dist:g},{lm[y].dist:g}"
          f"  LCA children={len(anc.children)}  -> identical barcodes / zero-length branches")
print(" total zero-distance distinct-cell pairs:", tot, f"({100*tot/ (np.triu(np.ones((1,1)))).sum():.0f}" if False else "")

# 2. how seed/subsample-fragile is the cross-clone constant?
print("\n=== fragility of within_max (sets the cross-clone constant) ===")
bcs=[l.strip() for l in open(HERE/"export/sc/barcodes.tsv")]
def key_mat(x):
    x=re.sub(r"-\d+$","",x); p=x.split("_")
    if len(p)>1 and p[0]==p[1]: p=p[1:]
    a=[t for t in p if re.fullmatch(r"a\d+",t)]; return (p[0],a[0] if a else "",p[-1])
def key_lin(x):
    p=x.split("_"); a=[t for t in p if re.fullmatch(r"a\d+",t)]; return (p[0],a[0] if a else "",p[-1])
mk={key_mat(b):i+1 for i,b in enumerate(bcs)}
lin=pd.read_csv(B/"single_cell/sc_cell_lineage.csv"); lin["lineageGrp"]=lin.lineageGrp.astype(str)
lin["mid"]=lin.cellBC.map(key_lin).map(mk)
rows=[]
for cl in sorted(cells.clone.unique(), key=lambda c:int(c[1:])):
    t=Tree(str(tf[cl]),format=1); lv={l.name for l in t.iter_leaves()}-{"synthetic"}
    us=lin[(lin.lineageGrp==cl)&lin.cellBC.isin(lv)&lin.mid.notna()].drop_duplicates("cellBC").cellBC.tolist()
    lm={l.name:l for l in t.iter_leaves()}
    # full diameter over ALL usable cells (not just the 73 sampled)
    mx=0.0
    nodes=[lm[x] for x in us]
    # farthest-pair via two-sweep on the tree restricted to usable leaves (exact on trees)
    src=nodes[0]
    d1=[(t.get_distance(src,x),x) for x in nodes]; f1=max(d1)[1]
    d2=[(t.get_distance(f1,x),x) for x in nodes]; mx=max(d2)[0]
    rows.append((cl,len(us),mx))
F=pd.DataFrame(rows,columns=["clone","usable_cells","full_diam"]).sort_values("full_diam",ascending=False)
print(F.head(6).to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("global max diameter over ALL 14,831 usable cells: %.4f (clone %s)" % (F.full_diam.max(), F.iloc[0].clone))
print("within_max realised in the npz (73-cell subsample):        1.5076")
print("=> the cross-clone constant would be 2 x %.4f = %.4f if all usable cells were kept, "
      "i.e. %.1f%% larger than the stored %.4f"
      % (F.full_diam.max(), 2*F.full_diam.max(), 100*(2*F.full_diam.max()/3.015271-1), 3.015271))
