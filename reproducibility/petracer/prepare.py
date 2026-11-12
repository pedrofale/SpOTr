"""PEtracer (Koblan et al. 2025), with synthetic Visium spots from one half of a tumor section and dissociated cells from the other half.

The data are the mouse 4T1 lung metastases of the PEtracer study, measured by MERFISH (175 genes) with the lineage of each tumor cell read in
situ, as pycea.datasets.koblan25(experiment="tumor") provides them (downloaded from Zenodo record 15750529 on first use and cached in
~/.treedata). They contain raw counts, positions, 18 cell subtypes (Malignant and 17 others), one time-calibrated tree over the traced malignant
cells with its leaves at time 35, and 18 clades.

We split the cells of one section (default M3-4, without doublets) at random into two halves (default_rng of numpy with seed 0).
  spots        We measure the first half as by Visium, with spots of 55 um on a hexagonal lattice 100 um apart, each containing the cells whose
               centers are within it. The cells between spots are not measured, and the spots without a cell are dropped. The counts of each spot
               are the sum of the counts of its cells.
  dissociated  We sample 8,000 cells of every type without replacement from the other half, so that the proportions of traced cells and of cell
               types are kept.
The label of each dissociated cell is its subtype, and the clone of each traced malignant cell is its clade ("clade 1" .. "clade 18"), or
"pooled" for the traced cells outside the 18 clades. The tree of the mapped cells is the subtree of the time-calibrated tree on their leaves.

Writes $SPOTR_DATA/petracer/<section>/
  sc.h5ad      The dissociated cells, with raw counts, the columns cell_type, clone (missing for the cells without a lineage readout and the
               non-tumor cells), x and y (in situ positions) of obs, and the tree of the mapped cells in Newick format in uns["tree"] (branch
               lengths in the time of the tree).
  st.h5ad      The spots, with raw counts, the spot centers (um) in obsm["spatial"], n_cells and n_tumor in obs, and the truth in
               obsm["clone_counts"] (the traced malignant cells of each clone in each spot) and obsm["type_counts"] (the non-tumor cells of each
               subtype in each spot).
  insitu.npz   The cells of the measured half, with xy and label (the clone for traced malignant cells, "Malignant" for the other malignant cells
               and the subtype otherwise).

Usage: python prepare.py [section]   (environment benchmark_env, pycea 0.2.0)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402

SPOT_D, SPOT_A = 55.0, 100.0                        # spot diameter and center-to-center spacing of Visium (um)
N_DISS, SEED = 8000, 0
CANCER = "Malignant"


def visium_spots(xy):
    """Return the spot centers on a hexagonal lattice over the bounding box of the cells, and the spot of each cell, or -1 for a cell between spots.
    """
    from scipy.spatial import cKDTree
    lo, hi = xy.min(0), xy.max(0); a = SPOT_A; rows = np.arange(lo[1] + a / 4, hi[1], a * np.sqrt(3) / 2)
    grid = np.array([(x, y) for i, y in enumerate(rows) for x in np.arange(lo[0] + a / 4 + (a / 2) * (i % 2), hi[0], a)])
    dist, near = cKDTree(grid).query(xy); hit = dist <= SPOT_D / 2
    used, inv = np.unique(near[hit], return_inverse=True); spot = np.full(len(xy), -1); spot[hit] = inv.ravel()
    return grid[used], spot


def newick(parent, length, names):
    """Return a Newick string with branch lengths at full precision."""
    kids = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p >= 0:
            kids[p].append(v)
    def rec(v):
        s = f"({','.join(rec(c) for c in kids[v])}){names[v]}" if kids[v] else names[v]
        return s + (f":{float(length[v])!r}" if parent[v] >= 0 else "")
    return rec(int(np.where(np.asarray(parent) < 0)[0][0])) + ";"


def main(section="M3-4"):
    import anndata as ad, pandas as pd, pycea, scipy.sparse as sp
    from spotr import tree as sptree
    t = pycea.datasets.koblan25(experiment="tumor"); o = t.obs
    sub_all = o["cell_subtype"].astype(str).to_numpy(); clade_ids = sorted({c for c in o["clade"].astype(str) if c != "nan"}, key=int)
    others = sorted(set(sub_all) - {CANCER})
    idx = np.where(((o["sample"].astype(str) == section) & (o["type"].astype(str) != "doublet")).to_numpy())[0]
    X = t.X[idx]; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.int32)
    xy = np.asarray(t.obsm["spatial"])[idx].astype(np.float32).astype(float); names = np.asarray(t.obs_names[idx]).astype(str)
    sub = sub_all[idx]; mal = sub == CANCER; traced = o["type"].astype(str).to_numpy()[idx] == "tracing"
    clade = o["clade"].astype(str).to_numpy()[idx]
    clone = np.array([("pooled" if c == "nan" else f"clade {c}") if (m and tr) else "" for c, m, tr in zip(clade, mal, traced)], dtype=object)
    label = np.where(clone != "", clone, sub)                                                  # in situ label, the clone or otherwise the subtype
    # the two halves
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(idx)); side, pool = perm[: len(idx) // 2], perm[len(idx) // 2:]
    diss = np.sort(rng.choice(pool, min(N_DISS, len(pool)), replace=False))
    # spots from the first half
    centres, spot = visium_spots(xy[side]); inside = spot >= 0; cells, s_of = side[inside], spot[inside]; M = len(centres)
    Y = np.zeros((M, X.shape[1]), np.int64); np.add.at(Y, s_of, X[cells])
    clones = [f"clade {c}" for c in clade_ids] + ["pooled"]
    Ncl = np.zeros((M, len(clones))); tc = clone[cells] != ""; np.add.at(Ncl, (s_of[tc], [clones.index(c) for c in clone[cells][tc]]), 1.0)
    Nme = np.zeros((M, len(others))); nm = ~mal[cells]; np.add.at(Nme, (s_of[nm], [others.index(s) for s in sub[cells][nm]]), 1.0)
    genes = pd.Index(np.asarray(t.var_names).astype(str))
    spots_ = pd.Index([f"spot{i}" for i in range(M)])
    st = ad.AnnData(X=sp.csr_matrix(Y.astype(np.float32)), obs=pd.DataFrame(dict(n_cells=np.bincount(s_of, minlength=M).astype(float),
                                                                                 n_tumor=np.bincount(s_of[mal[cells]], minlength=M).astype(float)), index=spots_),
                    var=pd.DataFrame(index=genes))
    st.obsm["spatial"] = centres.astype(float)
    st.obsm["clone_counts"] = pd.DataFrame(Ncl, index=spots_, columns=clones); st.obsm["type_counts"] = pd.DataFrame(Nme, index=spots_, columns=others)
    # the dissociated cells, with the tree of the mapped ones
    cl = pd.Categorical([c if c else np.nan for c in clone[diss]], categories=clones)
    sc = ad.AnnData(X=sp.csr_matrix(X[diss].astype(np.float32)), obs=pd.DataFrame(dict(cell_type=pd.Categorical(sub[diss]), clone=cl, x=xy[diss, 0], y=xy[diss, 1]),
                                                                                    index=pd.Index(names[diss])), var=pd.DataFrame(index=genes))
    nn, parent, length = sptree.from_networkx(t.obst["tree"], depth_key="time"); where = {n: i for i, n in enumerate(nn)}
    keep = {where[n]: n for n in names[diss][clone[diss] != ""]}
    sub_names, sub_parent, sub_length = sptree.subtree(parent, length, keep); sc.uns["tree"] = newick(sub_parent, sub_length, sub_names)
    out = paths.DATA / "petracer" / section; out.mkdir(parents=True, exist_ok=True)
    sc.write_h5ad(out / "sc.h5ad", compression="gzip"); st.write_h5ad(out / "st.h5ad", compression="gzip")
    np.savez_compressed(out / "insitu.npz", xy=xy[side].astype(np.float32), label=label[side].astype(str))
    print(f"PEtracer {section}: {len(diss)} dissociated cells ({int((clone[diss] != '').sum())} with a clone, "
          f"{int((mal[diss] & (clone[diss] == '')).sum())} malignant without), {M} spots ({np.bincount(s_of).mean():.1f} cells each) -> {out}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main(*sys.argv[1:2])
