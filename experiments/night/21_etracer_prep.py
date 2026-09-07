"""A SECOND DATASET, AND A WEAKER REQUIREMENT ON THE REFERENCE.

PEtracer gives matched single cells: expression, tree AND position for the same cell.  That is a
lot to ask.  eTRACER gives something weaker and much more common -- an in-situ readout that labels
each spatial BIN with a clone, plus dissociated cells with expression and a tree.  No cell has a
position.

The relationship can still be estimated.  For a pair of spatial bins with clones (k, l), the tree
distance between the cells they contain is not known but its DISTRIBUTION is: it is the empirical
distribution of patristic distances between cells of clone k and cells of clone l, which the
dissociated data provides.  Sampling from it turns bin pairs into (tree distance, spatial
distance) draws, and everything downstream is unchanged.

This script builds the inputs: bins aggregated into spots by a k-d partition, exactly as the
PEtracer pseudo-spots are built.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
ET = HERE.parent / "etracer"


def kd_split(xy, idx, depth):
    if depth == 0:
        return [idx]
    ax = int(np.argmax(xy[idx].max(0) - xy[idx].min(0)))
    o = idx[np.argsort(xy[idx, ax])]
    h = len(o) // 2
    return kd_split(xy, o[:h], depth - 1) + kd_split(xy, o[h:], depth - 1)


for TAG in ["LE", "EE"]:
    d = np.load(ET / f"etracer_inputs_{TAG}.npz", allow_pickle=True)
    Xb = np.asarray(d["X_st"], float)             # bin expression
    xyb = np.asarray(d["xy"], float)
    binc = np.asarray(d["bin_clone"]).astype(str)
    Xc = np.asarray(d["X_sc"], float)
    CT = np.asarray(d["C_tree"], float)
    cellc = np.asarray(d["cell_clone"]).astype(str)
    clones = np.asarray(d["clones"]).astype(str)

    groups = kd_split(xyb, np.arange(len(xyb)), 9)          # 512 spots
    m = len(groups)
    X_st = np.array([Xb[g].mean(0) for g in groups])
    xy_st = np.array([xyb[g].mean(0) for g in groups])
    comp = np.zeros((m, len(clones)))
    ci = {c: i for i, c in enumerate(clones)}
    for j, g in enumerate(groups):
        for b in g:
            if binc[b] in ci:
                comp[j, ci[binc[b]]] += 1

    np.savez_compressed(HERE / f"etracer_spots_{TAG}.npz", X_sc=Xc, X_st=X_st, C_tree=CT,
                        xy_st=xy_st, cell_clone=cellc, clones=clones, spot_comp=comp,
                        n_bins=np.array([len(g) for g in groups]))
    print(f"{TAG}: {len(Xc)} cells x {m} spots; bins/spot {np.mean([len(g) for g in groups]):.1f}; "
          f"spot purity (largest clone share) {np.mean(comp.max(1)/comp.sum(1)):.2f}; "
          f"tissue {np.ptp(xy_st, 0).round(0)} um")
