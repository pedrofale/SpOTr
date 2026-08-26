"""Does the SS3 phylogeny carry information about spatial arrangement in the Visium?

This is SpOTr's core premise, stated as something testable without running SpOTr:
the Gromov-Wasserstein term aligns tree distances among cells with spatial distances
among spots, which only helps if spots whose clones are close in the tree are also
close in space.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def clone_tree_distances(edges_csv):
    """Path distance between clones in the Numbat clone tree, weighted by n_events."""
    e = pd.read_csv(edges_csv)
    nodes = sorted(set(e["from_clone"]) | set(e["to_clone"]))
    idx = {c: i for i, c in enumerate(nodes)}
    K = len(nodes)
    D = np.full((K, K), np.inf)
    np.fill_diagonal(D, 0.0)
    for _, r in e.iterrows():
        i, j, w = idx[r["from_clone"]], idx[r["to_clone"]], float(r["n_events"])
        D[i, j] = D[j, i] = w
    for k in range(K):                      # Floyd-Warshall
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return np.array(nodes), D


def mantel(x, y, n_perm=2000, labels=None):
    """Correlation between two distance vectors with a label-permutation null."""
    r = np.corrcoef(x, y)[0, 1]
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(len(labels))
        null[t] = np.corrcoef(x[perm][:, None].ravel() if x.ndim > 1 else x, y)[0, 1]
    return r, null


def main(section="B1"):
    d = np.load(HERE / f"bcsa3_{section}_inputs.npz", allow_pickle=True)
    xy, truth = d["xy"], d["spot_clone_numbat"]
    nodes, Dtree = clone_tree_distances(HERE / "bcsa3_clone_edges.csv")
    pos = {c: i for i, c in enumerate(nodes)}

    keep = np.array([c in pos for c in truth])
    xy, truth = xy[keep], truth[keep]
    ti = np.array([pos[c] for c in truth])

    print(f"BCSA3/{section}: {len(truth)} spots, {len(np.unique(truth))} clones\n")

    # --- Test 1: spot-pair level -------------------------------------------------
    # subsample pairs; all-pairs is 2.6M and unnecessary
    n = len(truth)
    ii = rng.integers(0, n, 300_000); jj = rng.integers(0, n, 300_000)
    ok = ii != jj
    ii, jj = ii[ok], jj[ok]
    dt = Dtree[ti[ii], ti[jj]]
    ds = np.linalg.norm(xy[ii] - xy[jj], axis=1)
    r_obs = np.corrcoef(dt, ds)[0, 1]

    # null: permute the clone labels over spots, preserving both marginals
    null = np.empty(500)
    for t in range(500):
        p = rng.permutation(n)
        null[t] = np.corrcoef(Dtree[ti[p][ii], ti[p][jj]], ds)[0, 1]
    z = (r_obs - null.mean()) / null.std()
    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (len(null) + 1)
    print("Test 1 - spot pairs: corr(tree distance between clones, spatial distance)")
    print(f"  observed r = {r_obs:+.4f}")
    print(f"  null       = {null.mean():+.4f} +/- {null.std():.4f}   (z = {z:+.1f}, p = {p:.4f})")
    print(f"  -> {'phylogeny predicts spatial arrangement' if p < 0.05 and r_obs > 0 else 'NO usable tree-space agreement'}\n")

    # --- Test 2: clone-territory level -------------------------------------------
    cl = [c for c in np.unique(truth) if (truth == c).sum() >= 20]
    cent = np.stack([xy[truth == c].mean(0) for c in cl])
    Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
    Dt = Dtree[np.ix_([pos[c] for c in cl], [pos[c] for c in cl])]
    iu = np.triu_indices(len(cl), 1)
    r2 = np.corrcoef(Dt[iu], Ds[iu])[0, 1]
    null2 = np.empty(5000)
    for t in range(5000):
        p2 = rng.permutation(len(cl))
        null2[t] = np.corrcoef(Dt[np.ix_(p2, p2)][iu], Ds[iu])[0, 1]
    pv = (np.sum(np.abs(null2) >= abs(r2)) + 1) / (len(null2) + 1)
    print(f"Test 2 - clone territories ({len(cl)} clones with >=20 spots):")
    print(f"  Mantel r(tree distance, centroid distance) = {r2:+.3f}   (p = {pv:.4f})")
    print(f"  -> {'territories follow the phylogeny' if pv < 0.05 and r2 > 0 else 'territory layout is unrelated to the phylogeny'}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["B1"]))
