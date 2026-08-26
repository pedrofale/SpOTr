"""Does the lineage tree predict spatial position? Tested on eTRACER.

This is the cleanest test of SpOTr's premise available anywhere so far:

  - The tree is a real LINEAGE phylogeny (IQ-TREE2 ML on indel alleles written into
    endogenous 3'UTR target sites), NOT a CNV-derived clone graph.
  - It is reconstructed from barcodes only, so it is not informed by position -- no
    circularity, unlike KP-Tracer (spatially imputed) or BaSISS (clone = smoothed field).
  - Its leaves ARE Stereo-seq bins, labelled by their own coordinates ("10800_1700"),
    so tree metric and spatial metric live on identical units at SpOTr's own resolution.

Test per clone: does tree distance between two bins predict their spatial distance?
Null: permute the leaf->coordinate assignment within the clone.
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from ete3 import Tree

HERE = Path(__file__).resolve().parent
BASE = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing" / "spatial"
MAX_LEAVES = 400
N_PERM = 999
rng = np.random.default_rng(0)


def cophenetic(t, leaves):
    """Pairwise topological distance between the given leaves (unit edges).

    Runs on the FULL tree and simply ignores leaves outside `leaves`, so no pruning is
    needed -- ete3's prune() raises TreeError on the large trees here.
    """
    idx = {l: i for i, l in enumerate(leaves)}
    n = len(leaves)
    D = np.zeros((n, n))
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node._acc = [(idx[node.name], 0.0)] if node.name in idx else []
            continue
        kids = [c._acc for c in node.children]
        for a in range(len(kids)):
            for b in range(a + 1, len(kids)):
                for i, di in kids[a]:
                    for j, dj in kids[b]:
                        d = di + dj + 2.0
                        D[i, j] = D[j, i] = d
        node._acc = [(i, d + 1.0) for k in kids for i, d in k]
    return D


def analyse(tf):
    t = Tree(str(tf), format=1)
    leaves = [l.name for l in t.iter_leaves() if re.fullmatch(r"\d+_\d+", l.name)]
    if len(leaves) < 12:
        return None
    if len(leaves) > MAX_LEAVES:
        leaves = sorted(rng.choice(leaves, MAX_LEAVES, replace=False))
    xy = np.array([[int(v) for v in l.split("_")] for l in leaves], dtype=float)

    Dt = cophenetic(t, leaves)
    Ds = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    iu = np.triu_indices(len(leaves), 1)
    dt, ds = Dt[iu], Ds[iu]
    if dt.std() == 0 or ds.std() == 0:
        return None
    r = np.corrcoef(dt, ds)[0, 1]

    null = np.empty(N_PERM)
    for k in range(N_PERM):
        p = rng.permutation(len(leaves))
        null[k] = np.corrcoef(dt, Ds[np.ix_(p, p)][iu])[0, 1]
    pv = (np.sum(null >= r) + 1) / (N_PERM + 1)          # one-sided: tree predicts space
    return dict(clone=tf.name.split(".")[0], n_leaves=len(leaves), r=r,
                null_mean=float(null.mean()), null_sd=float(null.std()), p=pv)


def main():
    rows = []
    for stage in ["BL", "EE", "LE"]:
        d = BASE / f"spatial_trees_{stage}"
        for tf in sorted(d.glob("*.phy.treefile")):
            try:
                res = analyse(tf)
            except Exception as e:
                print(f"  {tf.name}: FAILED {type(e).__name__}"); continue
            if res:
                res["stage"] = stage
                rows.append(res)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "etracer_premise.csv", index=False)

    print(f"\n{'stage':>6} {'clones':>7} {'mean r':>8} {'median r':>9} "
          f"{'r>0':>6} {'p<0.05':>7}")
    print("-" * 50)
    for stage, s in df.groupby("stage"):
        print(f"{stage:>6} {len(s):>7} {s['r'].mean():>8.3f} {s['r'].median():>9.3f} "
              f"{(s['r']>0).sum():>4}/{len(s):<2} {(s['p']<0.05).sum():>4}/{len(s):<2}")
    print("-" * 50)
    print(f"{'ALL':>6} {len(df):>7} {df['r'].mean():>8.3f} {df['r'].median():>9.3f} "
          f"{(df['r']>0).sum():>4}/{len(df):<2} {(df['p']<0.05).sum():>4}/{len(df):<2}")
    print(f"\nnull is centred at {df['null_mean'].mean():+.4f} (sanity: should be ~0)")
    print("\ntop clones by correlation:")
    print(df.nlargest(8, "r")[["stage","clone","n_leaves","r","p"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
