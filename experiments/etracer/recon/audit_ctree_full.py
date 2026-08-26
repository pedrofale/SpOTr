"""Full independent recomputation of every within-clone block of C_tree, plus per-clone stats."""
import re, numpy as np, pandas as pd
from pathlib import Path
from ete3 import Tree

HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
def clone_of(p): return re.sub(r"-\d+$", "", p.name.replace(".phy.treefile", ""))
tree_files = {clone_of(f): f for f in (B / "single_cell/sc_trees_LE").glob("*.phy.treefile")}

d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
C = d["C_tree"].astype(np.float64)
cells = pd.read_csv(HERE / "recon/audit_cells_order.csv")

def cophenetic_indep(t, names):
    """Ancestor-chain implementation: for each leaf, dict {id(anc): cumdist}; pair distance
    = first shared ancestor. Independent of the postorder code in prep_le_full.py."""
    lm = {}
    for l in t.iter_leaves():
        lm.setdefault(l.name, l)
    chains = []
    for nm in names:
        cur, acc, ch = lm[nm], 0.0, {}
        while cur is not None:
            ch[id(cur)] = acc
            if cur.up is not None: acc += cur.dist
            cur = cur.up
        chains.append((ch, lm[nm]))
    n = len(names); D = np.zeros((n, n))
    for a in range(n):
        cha = chains[a][0]
        for b in range(a + 1, n):
            cur, acc = chains[b][1], 0.0
            while cur is not None:
                if id(cur) in cha:
                    D[a, b] = D[b, a] = cha[id(cur)] + acc; break
                acc += cur.dist; cur = cur.up
            else:
                D[a, b] = D[b, a] = np.nan   # disconnected -> should never happen
    return D

recs, maxerr_all, zero_pairs = [], 0.0, []
for cl, g in cells.groupby("clone", sort=False):
    t = Tree(str(tree_files[cl]), format=1)
    names = g.cellBC.to_numpy(); ii = g.index.to_numpy()
    M = cophenetic_indep(t, names)
    sub = C[np.ix_(ii, ii)]
    err = np.nanmax(np.abs(M - sub))
    maxerr_all = max(maxerr_all, err)
    ut = np.triu_indices(len(names), 1)
    v = sub[ut]
    nz = int((v == 0).sum())
    if nz: zero_pairs.append((cl, nz, len(v)))
    allleaf = t.get_leaf_names()
    recs.append(dict(clone=cl, cells=len(names), tree_leaves=len(allleaf),
                     has_synth=("synthetic" in allleaf),
                     pairs=len(v), min=v.min(), q25=np.quantile(v,.25), med=np.median(v),
                     q75=np.quantile(v,.75), max=v.max(), mean=v.mean(), sd=v.std(),
                     n_unique=len(np.unique(np.round(v,9))), zero_pairs=nz,
                     maxerr=err))
R = pd.DataFrame(recs).sort_values("max", ascending=False)
pd.set_option("display.width", 200, "display.max_rows", 60)
print("MAX abs error over ALL 45 within-clone blocks (independent recompute vs C_tree): %.3e" % maxerr_all)
print("blocks with NaN (unreachable pair):", int(R.maxerr.isna().sum()))
print()
print(R[["clone","cells","tree_leaves","has_synth","pairs","min","med","max","sd","n_unique","zero_pairs","maxerr"]]
      .to_string(index=False, float_format=lambda x: f"{x:.6g}"))
R.to_csv(HERE/"recon/audit_ctree_perclone.csv", index=False)
print("\n--- degeneracy screen ---")
deg = R[(R["max"] == 0) | (R.n_unique <= 1)]
print("clones with all-zero or single-valued within-clone distances:", list(deg.clone) or "NONE")
print("clones with any zero-distance distinct-cell pair:", zero_pairs)
print("\nwithin-clone DIAMETER (max) summary across 45 clones: min %.4f med %.4f max %.4f"
      % (R["max"].min(), R["max"].median(), R["max"].max()))
print("clone driving global within_max:", R.iloc[0].clone, "=", R.iloc[0]["max"])
print("cross-clone constant in C_tree = %.6f ; ratio to MEDIAN clone diameter = %.2fx ; to 2nd-largest = %.2fx"
      % (3.015270948, 3.015270948/R["max"].median(), 3.015270948/R["max"].iloc[1]))
