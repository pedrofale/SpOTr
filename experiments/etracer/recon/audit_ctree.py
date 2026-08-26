"""Independent verification of C_tree in etracer_LE_full.npz."""
import re, itertools, numpy as np, pandas as pd
from pathlib import Path
from ete3 import Tree

HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer")
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
def clone_of(p): return re.sub(r"-\d+$", "", p.name.replace(".phy.treefile", ""))
tree_files = {clone_of(f): f for f in (B / "single_cell/sc_trees_LE").glob("*.phy.treefile")}

d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
C = d["C_tree"].astype(np.float64)
cell_clone = d["cell_clone"]
cells = pd.read_csv(HERE / "recon/audit_cells_order.csv")
assert (cells.clone.to_numpy().astype(str) == cell_clone).all()
n = len(cells)

# ---------- global structural checks ----------
print("=== structural ===")
print("shape", C.shape, "dtype", d["C_tree"].dtype)
print("symmetric exactly:", bool((C == C.T).all()))
print("zero diagonal:", bool((np.diag(C) == 0).all()))
print("all finite:", bool(np.isfinite(C).all()))
print("min value:", C.min(), " any negative:", bool((C < 0).any()))
print("max value:", C.max())

same = cell_clone[:, None] == cell_clone[None, :]
off = C[~same]
print("cross-clone entries: n=%d unique=%d min=%.10g max=%.10g" %
      (off.size, len(np.unique(off)), off.min(), off.max()))
win = C[same & ~np.eye(n, dtype=bool)]
print("within-clone (off-diag): n=%d min=%.10g max=%.10g" % (win.size, win.min(), win.max()))
print("cross constant > every within distance:", bool(off.min() > win.max()))
print("ratio cross/within_max = %.4f" % (off.min() / win.max()))
print("within-clone zeros (off-diag pairs at distance 0):", int((win == 0).sum()))

# triangle inequality spot check on a random subsample
rng = np.random.default_rng(0)
idx = rng.choice(n, 300, replace=False)
S = C[np.ix_(idx, idx)]
viol = (S[:, :, None] > S[:, None, :] + S[None, :, :] + 1e-6).sum()
print("triangle-inequality violations in 300x300x300 subsample:", int(viol))

# ---------- independent patristic recomputation ----------
def anc_dists(node):
    """{node_object: cumulative branch length from `node` up to it}, using RAW dists."""
    out, cur, acc = {}, node, 0.0
    while cur is not None:
        out[id(cur)] = (acc, cur)
        if cur.up is not None: acc += cur.dist
        cur = cur.up
    return out

def patristic_manual(t, a, b, leafmap):
    da = anc_dists(leafmap[a]); db = anc_dists(leafmap[b])
    best = None
    cur = leafmap[b]; acc = 0.0
    while cur is not None:
        if id(cur) in da:
            best = da[id(cur)][0] + acc; break
        acc += cur.dist; cur = cur.up
    return best

print("\n=== per-pair independent patristic (manual path sum vs C_tree) ===")
test_clones = ["C2", "C4", "C31", "C143"]
for cl in test_clones:
    t = Tree(str(tree_files[cl]), format=1)
    lm = {}
    dupnames = []
    for l in t.iter_leaves():
        if l.name in lm: dupnames.append(l.name)
        lm[l.name] = l
    negd = [x.dist for x in t.traverse() if x.dist < 0]
    g = cells[cells.clone == cl]
    ii = g.index.to_numpy(); names = g.cellBC.to_numpy()
    pairs = list(itertools.combinations(range(len(names)), 2))
    rng2 = np.random.default_rng(1)
    sel = [pairs[k] for k in rng2.choice(len(pairs), min(12, len(pairs)), replace=False)]
    errs, errs_ete = [], []
    rowsout = []
    for a, b in sel:
        man = patristic_manual(t, names[a], names[b], lm)
        ete = t.get_distance(lm[names[a]], lm[names[b]])
        got = C[ii[a], ii[b]]
        errs.append(abs(man - got)); errs_ete.append(abs(ete - got))
        rowsout.append((names[a][:14], names[b][:14], man, ete, got))
    print(f"\n[{cl}]  leaves in tree={len(lm)}  dup leaf names={len(dupnames)}  "
          f"negative branch lengths={len(negd)}  cells used={len(names)}")
    for r in rowsout[:6]:
        print("   %-14s %-14s manual=%.9f ete3=%.9f C_tree=%.9f" % r)
    print("   max |manual - C_tree| over %d pairs = %.3e ; max |ete3 - C_tree| = %.3e"
          % (len(sel), max(errs), max(errs_ete)))
    # FULL submatrix check via ete3 for small clones
    if len(names) <= 45:
        M = np.zeros((len(names), len(names)))
        for a in range(len(names)):
            for b in range(a+1, len(names)):
                M[a, b] = M[b, a] = t.get_distance(lm[names[a]], lm[names[b]])
        sub = C[np.ix_(ii, ii)]
        print("   FULL %dx%d submatrix max abs err vs ete3 = %.3e" % (len(names), len(names), np.abs(M-sub).max()))
