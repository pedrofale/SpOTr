"""Test key_mat/key_lin ID matching between sc_cell_lineage.csv and export/sc/barcodes.tsv."""
import re, sys, collections
from pathlib import Path
import pandas as pd
from ete3 import Tree

HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"

def key_mat(x):
    x = re.sub(r"-\d+$", "", x); p = x.split("_")
    if len(p) > 1 and p[0] == p[1]: p = p[1:]
    a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])

def key_lin(x):
    p = x.split("_"); a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])

bcs = [l.strip() for l in open(HERE/"export/sc/barcodes.tsv") if l.strip()]
print(f"barcodes.tsv: {len(bcs)} entries, {len(set(bcs))} unique")

# --- structural survey of barcode formats
def shape(x, f):
    return f"{len(f(x))}tok"
mat_tok = collections.Counter(len(re.sub(r'-\d+$','',b).split('_')) for b in bcs)
print("barcodes.tsv token-count distribution:", dict(sorted(mat_tok.items())))
mat_dupdouble = sum(1 for b in bcs if (lambda p: len(p)>1 and p[0]==p[1])(re.sub(r'-\d+$','',b).split('_')))
print(f"  barcodes with doubled first token (T1_T1_...): {mat_dupdouble}")
suffix = collections.Counter(re.search(r'-(\d+)$', b).group(1) if re.search(r'-(\d+)$', b) else "NONE" for b in bcs)
print("  trailing -N suffix distribution:", dict(suffix))

mk_keys = [key_mat(b) for b in bcs]
mk = {}
dupmat = collections.Counter(mk_keys)
ncoll = sum(v-1 for v in dupmat.values() if v > 1)
print(f"key_mat: {len(set(mk_keys))} unique keys from {len(bcs)} barcodes -> {ncoll} barcodes lost to key collision")
if ncoll:
    ex = [k for k,v in dupmat.items() if v>1][:3]
    for k in ex:
        print("   collision example:", k, [b for b in bcs if key_mat(b)==k][:4])
for b in bcs: mk[key_mat(b)] = b

lin = pd.read_csv(B/"single_cell/sc_cell_lineage.csv")
lin["lineageGrp"] = lin.lineageGrp.astype(str)
print(f"\nsc_cell_lineage.csv: {len(lin)} rows, {lin.cellBC.nunique()} unique cellBC, {lin.lineageGrp.nunique()} clones")
lin_tok = collections.Counter(len(x.split('_')) for x in lin.cellBC)
print("cellBC token-count distribution:", dict(sorted(lin_tok.items())))

lin["key"] = lin.cellBC.map(key_lin)
lin["matrix_id"] = lin.key.map(mk)
res = lin.matrix_id.notna()
print(f"\nRESOLVE: {res.sum()} / {len(lin)} lineage cells resolve to a matrix column ({100*res.mean():.2f}%)")
print(f"  unique matrix columns hit: {lin.matrix_id.nunique()}")
mm = lin[res].matrix_id.value_counts()
print(f"  matrix columns claimed by >1 lineage cell: {(mm>1).sum()} (max {mm.max()})")
if (mm>1).sum():
    bad = mm[mm>1].index[:3]
    for b in bad:
        print("   ", b, "<-", lin[lin.matrix_id==b].cellBC.tolist()[:5], "clones", lin[lin.matrix_id==b].lineageGrp.tolist()[:5])

# --- shared clones
sc_trees = {f.name.split("-")[0]: f for f in (B/"single_cell/sc_trees_LE").glob("*.phy.treefile")}
sp_trees = {f.name.split("-")[0] for f in (B/"spatial/spatial_trees_LE").glob("*.phy.treefile")}
shared = sorted(set(sc_trees) & sp_trees, key=lambda c: int(c[1:]))
print(f"\nsc LE tree clones: {len(sc_trees)}; spatial LE tree clones: {len(sp_trees)}; shared: {len(shared)}")

# cross-check shared against spatial_lineage_LE.csv clone labels
sp = pd.read_csv(B/"spatial/spatial_lineage_LE.csv").dropna(subset=["lineageGrp","x","y"])
sp["lineageGrp"] = sp.lineageGrp.astype(str)
print(f"spatial_lineage_LE.csv: {len(sp)} labelled bins, {sp.lineageGrp.nunique()} distinct clones; "
      f"clones with a spatial tree AND an sc tree: {len(set(sp.lineageGrp)&set(shared))}")

lin["is_shared"] = lin.lineageGrp.isin(shared)
tab = lin.groupby("is_shared").agg(cells=("cellBC","size"), resolved=("matrix_id", lambda s: s.notna().sum()))
print("\nResolution by shared-clone membership:")
for k,r in tab.iterrows():
    lbl = "shared-45 clones" if k else "other clones"
    print(f"  {lbl:22s} cells={r.cells:6d} resolved={r.resolved:6d} ({100*r.resolved/r.cells:.2f}%) unresolved={r.cells-r.resolved}")

# unresolved diagnostics
unres = lin[lin.matrix_id.isna()]
print(f"\nUnresolved {len(unres)}: sample cellBCs {unres.cellBC.head(5).tolist()}")
print("  unresolved by first token:", dict(collections.Counter(x.split('_')[0] for x in unres.cellBC).most_common(10)))
print("  matrix first tokens:", dict(collections.Counter(key_mat(b)[0] for b in bcs).most_common(10)))
print("  unresolved animal tokens:", dict(collections.Counter(k[1] for k in unres.key).most_common(10)))
print("  matrix animal tokens:", dict(collections.Counter(k[1] for k in mk).most_common(15)))

# --- per-clone usable = resolved AND leaf of that clone's LE sc tree
rows = []
for clone in shared:
    g = lin[lin.lineageGrp == clone]
    leaves = {l.name for l in Tree(str(sc_trees[clone]), format=1).iter_leaves()}
    gr = g[g.matrix_id.notna()]
    inleaf = g[g.cellBC.isin(leaves)]
    usable = gr[gr.cellBC.isin(leaves)]
    rows.append(dict(clone=clone, tree_leaves=len(leaves), lin_cells=len(g),
                     resolved=len(gr), in_tree=len(inleaf), usable=len(usable),
                     sp_bins=int((sp.lineageGrp==clone).sum())))
df = pd.DataFrame(rows).sort_values("usable", ascending=False)
df.to_csv(HERE/"recon/usable_cells_per_clone.csv", index=False)
print("\n=== PER-CLONE (45 shared LE clones) ===")
print(df.to_string(index=False))
print(f"\nTOTALS: tree_leaves={df.tree_leaves.sum()} lin_cells={df.lin_cells.sum()} "
      f"resolved={df.resolved.sum()} in_tree={df.in_tree.sum()} usable={df.usable.sum()} sp_bins={df.sp_bins.sum()}")
print(f"clones with usable>=8: {(df.usable>=8).sum()}; >=20: {(df.usable>=20).sum()}; >=50: {(df.usable>=50).sum()}; ==0: {(df.usable==0).sum()}")
# are tree leaves a subset of lineage cellBCs?
allbc = set(lin.cellBC)
miss = []
for clone in shared[:len(shared)]:
    leaves = {l.name for l in Tree(str(sc_trees[clone]), format=1).iter_leaves()}
    miss.append(len(leaves - allbc))
print(f"tree leaves not found anywhere in sc_cell_lineage cellBC: total {sum(miss)}")
