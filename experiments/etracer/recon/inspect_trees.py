from pathlib import Path
import re, sys, numpy as np, pandas as pd
from ete3 import Tree

HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER/LineageTracing"
TD = B / "single_cell/sc_trees_LE"

sc = pd.read_csv(B/"single_cell/sc_cell_lineage.csv")
sp = pd.read_csv(B/"spatial/spatial_lineage_LE.csv")
sc_clones = set(sc.lineageGrp.astype(str))
sp_clones = set(sp.lineageGrp.astype(str)) - {"NA","nan"}
shared = sc_clones & sp_clones
print(f"sc clones {len(sc_clones)}  spatial LE clones {len(sp_clones)}  shared {len(shared)}")

tree_files = sorted(TD.glob("*.phy.treefile"))
tf_clone = {}
for f in tree_files:
    c = f.name.split(".")[0]
    clone, num = c.rsplit("-",1)
    tf_clone[clone] = (f, int(num))
print(f"tree files {len(tree_files)}  clones with trees {len(tf_clone)}")
print("shared but NO tree:", sorted(shared - set(tf_clone)))
print("tree but not shared:", sorted(set(tf_clone) - shared))

# prefix stats of sc cellBC
pref = sc.cellBC.str.split("_").str[0].value_counts()
print("sc cellBC prefixes:", pref.to_dict())

rows=[]
fail0=[]; fail1=[]
allleaf_in_sc = 0; allleaf_tot = 0
for clone in sorted(tf_clone, key=lambda c:int(c[1:])):
    f, num = tf_clone[clone]
    r = {"clone":clone, "filename_num":num, "shared":clone in shared}
    # parse tests
    t1=t0=None
    try:
        t1 = Tree(str(f), format=1)
    except Exception as e:
        fail1.append((clone, repr(e)[:120]))
    try:
        t0 = Tree(str(f), format=0)
    except Exception as e:
        fail0.append((clone, repr(e)[:120]))
    t = t1 if t1 is not None else t0
    if t is None:
        rows.append(r); continue
    lv = [l.name for l in t.iter_leaves()]
    r["n_leaves"]=len(lv); r["n_unique_leaves"]=len(set(lv))
    r["n_nodes"]=sum(1 for _ in t.traverse())
    r["root_children"]=len(t.children)
    r["max_depth"]=max(len(n.get_ancestors()) for n in t.iter_leaves())
    d=[n.dist for n in t.traverse() if not n.is_root()]
    d=np.array(d)
    r["bl_zero_frac"]=float((d<1e-6).mean()); r["bl_neg"]=int((d<0).sum())
    r["bl_min"]=float(d.min()); r["bl_med"]=float(np.median(d)); r["bl_max"]=float(d.max())
    r["bl_sum"]=float(d.sum())
    # naming: leaves vs sc_cell_lineage cellBC for this clone
    scbc = set(sc.loc[sc.lineageGrp==clone,"cellBC"])
    r["n_sc_cells"]=len(scbc)
    r["leaves_in_sc"]=len(set(lv)&scbc)
    r["leaves_not_in_sc"]=len(set(lv)-scbc)
    # are leaves in sc but assigned to a DIFFERENT clone?
    r["leaves_in_other_clone"]=len((set(lv)-scbc) & set(sc.cellBC))
    r["leaf_prefixes"]="|".join(sorted({x.split("_")[0] for x in lv}))
    r["leaf_nfields"]="|".join(sorted({str(x.count("_")) for x in lv}))
    allleaf_in_sc += r["leaves_in_sc"]; allleaf_tot += len(lv)
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(HERE/"recon/tree_inventory.csv", index=False)
print("\nparse failures format=1:", fail1)
print("parse failures format=0:", fail0)
print(f"\nleaves matching sc cellBC of same clone: {allleaf_in_sc}/{allleaf_tot}")
print("\nfilename_num == n_leaves for all?", bool((df.filename_num==df.n_leaves).all()),
      " mismatches:", df.loc[df.filename_num!=df.n_leaves,["clone","filename_num","n_leaves"]].to_dict("records"))
print("\nduplicate leaf names anywhere?", bool((df.n_leaves!=df.n_unique_leaves).any()))
print("\n--- 45 shared clones ---")
s = df[df.shared].sort_values("n_leaves", ascending=False)
print(s[["clone","n_leaves","n_sc_cells","leaves_not_in_sc","root_children","max_depth","bl_zero_frac","bl_med","bl_max","bl_sum"]].to_string(index=False))
print("\nshared leaves total:", int(s.n_leaves.sum()), " min/median/max:", int(s.n_leaves.min()), float(s.n_leaves.median()), int(s.n_leaves.max()))
print("\nBL global: med of medians %.5f, max of max %.4f, min nonzero %.3g" % (
    s.bl_med.median(), s.bl_max.max(), df.bl_min.min()))
