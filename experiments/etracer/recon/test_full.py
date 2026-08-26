import re, collections
from pathlib import Path
import pandas as pd
from ete3 import Tree
HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE/"Tumor_Immune_Evasion_eTRACER"/"LineageTracing"

def key_mat(x):
    x = re.sub(r"-\d+$", "", x); p = x.split("_")
    if len(p) > 1 and p[0] == p[1]: p = p[1:]
    a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])
def key_lin(x):
    p = x.split("_"); a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])

bcs=[l.strip() for l in open(HERE/"export/sc/barcodes.tsv") if l.strip()]
mk={key_mat(b):b for b in bcs}
lin=pd.read_csv(B/"single_cell/sc_cell_lineage.csv"); lin["lineageGrp"]=lin.lineageGrp.astype(str)
lin["key"]=lin.cellBC.map(key_lin)
print("lineage keys unique?", lin.key.nunique(), "of", len(lin))
lin["matrix_id"]=lin.key.map(mk)

# --- why do the 171 fail?
un=lin[lin.matrix_id.isna()]
by_ab = collections.defaultdict(set)   # (animal,bc16) -> prefixes present in matrix
by_bc = collections.defaultdict(set)
for k in mk: by_ab[(k[1],k[2])].add(k[0]); by_bc[k[2]].add((k[0],k[1]))
n_prefix_only = sum(1 for k in un.key if (k[1],k[2]) in by_ab)
n_bc_only     = sum(1 for k in un.key if k[2] in by_bc)
print(f"unresolved={len(un)}: same(animal,16bp) exists under a DIFFERENT tumour prefix: {n_prefix_only}; "
      f"16bp exists anywhere in matrix: {n_bc_only}")
# do middle tokens matter? does relaxing to (animal,16bp) create ambiguity?
amb=sum(1 for k,v in by_ab.items() if len(v)>1)
print(f"(animal,16bp) pairs that appear under >1 tumour prefix in the matrix: {amb} of {len(by_ab)}")
print(f"16bp values shared across >1 (prefix,animal): {sum(1 for k,v in by_bc.items() if len(v)>1)} of {len(by_bc)}")
# middle-token check: strip animal+prefix, are middle tokens ever needed?
mid=collections.Counter("_".join(x.split("_")[2:-1]) for x in lin.cellBC)
print("lineage cellBC middle tokens (between animal and 16bp):", dict(mid.most_common()))

sc_trees={f.name.split("-")[0]:f for f in (B/"single_cell/sc_trees_LE").glob("*.phy.treefile")}
sp=pd.read_csv(B/"spatial/spatial_lineage_LE.csv").dropna(subset=["lineageGrp"]).copy()
sp["lineageGrp"]=sp.lineageGrp.astype(str)
shared=sorted(set(sc_trees)&set(sp.lineageGrp), key=lambda c:int(c[1:]))
print("\nSHARED-45:", len(shared))
lin["is_shared"]=lin.lineageGrp.isin(shared)
for k,g in lin.groupby("is_shared"):
    print(f"  {'shared-45' if k else 'non-shared':12s} cells={len(g):6d} resolved={g.matrix_id.notna().sum():6d} "
          f"({100*g.matrix_id.notna().mean():.2f}%) clones={g.lineageGrp.nunique()}")

rows=[]
for c in shared:
    g=lin[lin.lineageGrp==c]
    lv={l.name for l in Tree(str(sc_trees[c]),format=1).iter_leaves()}
    lv.discard("synthetic")
    gr=g[g.matrix_id.notna()]
    rows.append(dict(clone=c, sp_bins=int((sp.lineageGrp==c).sum()), lin_cells=len(g),
                     resolved=int(g.matrix_id.notna().sum()), tree_leaves=len(lv),
                     in_tree=int(g.cellBC.isin(lv).sum()), usable=int(gr.cellBC.isin(lv).sum())))
df=pd.DataFrame(rows)
df["leaf_not_in_lin"]=[len({l.name for l in Tree(str(sc_trees[c]),format=1).iter_leaves()}-{"synthetic"}-set(lin.cellBC)) for c in shared]
df=df.sort_values("usable",ascending=False)
df.to_csv(HERE/"recon/usable_cells_per_clone.csv",index=False)
print("\n=== 45 SHARED LE CLONES ===")
print(df.to_string(index=False))
print(f"\nTOTALS sp_bins={df.sp_bins.sum()} lin_cells={df.lin_cells.sum()} resolved={df.resolved.sum()} "
      f"tree_leaves={df.tree_leaves.sum()} in_tree={df.in_tree.sum()} usable={df.usable.sum()}")
for t in (4,8,20,50,100):
    print(f"  clones with usable>={t}: {(df.usable>=t).sum()}")
print("  clones usable==0:", df[df.usable==0].clone.tolist())
# bins covered by clones passing thresholds
for t in (0,8,20):
    keep=set(df[df.usable>=t].clone)
    print(f"  usable>={t}: {len(keep)} clones, {df[df.clone.isin(keep)].sp_bins.sum()} labelled bins "
          f"({100*df[df.clone.isin(keep)].sp_bins.sum()/7568:.1f}% of 35,835-bin mask denominator 7568)")
