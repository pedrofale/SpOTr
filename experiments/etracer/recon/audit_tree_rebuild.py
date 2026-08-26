"""Independently reconstruct the cell ORDER used by prep_le_full.py (steps 1-4 only,
no matrix I/O), so C_tree rows can be joined back to leaf names."""
import re, numpy as np, pandas as pd
from pathlib import Path
from ete3 import Tree

HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
seed = 0

def key_mat(x):
    x = re.sub(r"-\d+$", "", x); p = x.split("_")
    if len(p) > 1 and p[0] == p[1]: p = p[1:]
    a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])
def key_lin(x):
    p = x.split("_"); a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])
def clone_of(path): return re.sub(r"-\d+$", "", path.name.replace(".phy.treefile", ""))

lin_sp = pd.read_csv(B / "spatial/spatial_lineage_LE.csv", keep_default_na=False, na_values=[])
lin_sp = lin_sp[lin_sp.lineageGrp != "NA"]
tree_files = {clone_of(f): f for f in (B / "single_cell/sc_trees_LE").glob("*.phy.treefile")}
shared = sorted(set(tree_files) & set(lin_sp.lineageGrp.astype(str)), key=lambda c: int(c[1:]))

bcs = [l.strip() for l in open(HERE / "export/sc/barcodes.tsv")]
mk = {key_mat(b): i + 1 for i, b in enumerate(bcs)}
lin_sc = pd.read_csv(B / "single_cell/sc_cell_lineage.csv")
lin_sc["lineageGrp"] = lin_sc.lineageGrp.astype(str)
lin_sc["matrix_id"] = lin_sc.cellBC.map(key_lin).map(mk)

avail = {}
for c in shared:
    t = Tree(str(tree_files[c]), format=1)
    leaves = {l.name for l in t.iter_leaves()} - {"synthetic"}
    g = lin_sc[(lin_sc.lineageGrp == c) & lin_sc.cellBC.isin(leaves) & lin_sc.matrix_id.notna()]
    avail[c] = g.drop_duplicates("cellBC")
n_av = {c: len(g) for c, g in avail.items()}
caps = np.arange(1, max(n_av.values()) + 1)
tot = np.array([sum(min(v, int(k)) for v in n_av.values()) for k in caps])
ok = np.where(tot <= 3000)[0]
cap = int(caps[ok[-1]]) if len(ok) else int(caps[0])
rows = []
for c in shared:
    g = avail[c]
    rows.append((g if len(g) <= cap else g.sample(cap, random_state=seed)).assign(clone=c))
cells = pd.concat(rows, ignore_index=True)
cells["matrix_id"] = cells.matrix_id.astype(int)
print("cap", cap, "n cells", len(cells))
cells[["cellBC", "clone", "matrix_id"]].to_csv(HERE / "recon/audit_cells_order.csv", index=False)

d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
cc = d["cell_clone"]
print("reconstructed clone vector matches npz cell_clone:", bool((cells.clone.to_numpy().astype(str) == cc).all()))
print("n cells npz", len(cc), "recon", len(cells))
# duplicate cellBC across clones? (would corrupt `pos` in prep)
dup = cells.cellBC.duplicated().sum()
print("duplicated cellBC across the whole selection:", dup)
dup_all = lin_sc.drop_duplicates(["cellBC","lineageGrp"]).cellBC.duplicated().sum()
print("cellBC assigned to >1 lineageGrp in sc_cell_lineage.csv:", dup_all)
# duplicate matrix_id (two cellBCs -> same matrix column)?
print("duplicated matrix_id in selection:", int(cells.matrix_id.duplicated().sum()))
