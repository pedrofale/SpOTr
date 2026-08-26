import re, itertools, collections, numpy as np, pandas as pd
from pathlib import Path
from ete3 import Tree
HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
def clone_of(p): return re.sub(r"-\d+$", "", p.name.replace(".phy.treefile", ""))
files = sorted((B/"single_cell/sc_trees_LE").glob("*.phy.treefile"))
print("treefiles:", len(files))
cnt = collections.Counter(clone_of(f) for f in files)
print("clone_of collisions (2 files -> same clone):", {k:v for k,v in cnt.items() if v>1} or "NONE")

cells = pd.read_csv(HERE/"recon/audit_cells_order.csv")
used = sorted(cells.clone.unique(), key=lambda c:int(c[1:]))
tf = {clone_of(f): f for f in files}

leafsets, negb, dupl, rootdist = {}, {}, {}, {}
for c in used:
    t = Tree(str(tf[c]), format=1)
    names = t.get_leaf_names()
    dupl[c] = len(names)-len(set(names))
    leafsets[c] = set(names) - {"synthetic"}
    negb[c] = sum(1 for x in t.traverse() if x.up is not None and x.dist < 0)
    rootdist[c] = t.dist
print("trees with duplicate leaf names:", {k:v for k,v in dupl.items() if v} or "NONE")
print("trees with negative branch lengths (would be clamped by max(dist,0)):",
      {k:v for k,v in negb.items() if v} or "NONE")
print("distinct root .dist values (unused by both implementations):", sorted(set(rootdist.values())))

# leaf-name overlap ACROSS clone trees -> cells that are relatives but get the cross constant
ov = []
for a,b in itertools.combinations(used,2):
    k = leafsets[a] & leafsets[b]
    if k: ov.append((a,b,len(k)))
print("clone-tree leaf-name overlaps:", ov[:10], "... total pairs overlapping:", len(ov))
sel = dict(zip(cells.cellBC, cells.clone))
bad = [(cb, cl, [c for c in used if cb in leafsets[c] and c != cl]) for cb, cl in sel.items()]
bad = [x for x in bad if x[2]]
print("selected cells that are ALSO a leaf of another clone's tree:", len(bad), bad[:5])

# ---- raw-newick hand check on the smallest tree ----
c = "C135"; raw = open(tf[c]).read().strip()
print(f"\n[{c}] raw newick length {len(raw)}; first 300 chars:\n", raw[:300])
t = Tree(str(tf[c]), format=1)
# find a cherry (two leaves sharing a parent) among selected cells
g = cells[cells.clone==c]; names=set(g.cellBC)
d = np.load(HERE/"etracer_LE_full.npz")["C_tree"].astype(np.float64)
for nd in t.traverse():
    ch = nd.children
    if len(ch)>=2 and all(x.is_leaf() for x in ch[:2]) and ch[0].name in names and ch[1].name in names:
        a,b = ch[0], ch[1]
        i = g.index[g.cellBC==a.name][0]; j = g.index[g.cellBC==b.name][0]
        # literal substring from the newick showing both leaves and their lengths
        pa, pb = raw.find(a.name), raw.find(b.name)
        lo,hi = min(pa,pb), max(pa,pb)
        print(f"\ncherry from raw text: ...{raw[max(0,lo-3):hi+len(b.name)+14]}...")
        print(f"  ete3 dists: {a.name} -> {a.dist:.9f} ; {b.name} -> {b.dist:.9f}")
        print(f"  hand sum a.dist+b.dist = {a.dist+b.dist:.9f}")
        print(f"  C_tree[{i},{j}]         = {d[i,j]:.9f}   diff {abs(a.dist+b.dist-d[i,j]):.2e}")
        break

# ---- tree-size vs diameter confound ----
R = pd.read_csv(HERE/"recon/audit_ctree_perclone.csv")
lo = np.log10(R.tree_leaves)
print("\nSpearman(tree_leaves, within-clone diameter) = %.3f  Pearson(log10 leaves, diam) = %.3f"
      % (R[["tree_leaves","max"]].corr(method="spearman").iloc[0,1], np.corrcoef(lo, R["max"])[0,1]))
print("Spearman(tree_leaves, median within dist) = %.3f"
      % R[["tree_leaves","med"]].corr(method="spearman").iloc[0,1])
print("diameter spread: max/min = %.1fx ; median-dist spread max/min = %.1fx"
      % (R["max"].max()/R["max"].min(), R["med"].max()/R["med"].min()))
print("\nsmallest trees (<=65 leaves): median diam %.3f | largest trees (>=300 leaves): median diam %.3f"
      % (R[R.tree_leaves<=65]["max"].median(), R[R.tree_leaves>=300]["max"].median()))
