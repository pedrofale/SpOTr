"""Rebuild the pseudo-spots so a spot is the CENTROID of its cells in the cost space.

CURRENT construction (prep_petracer.py):
    spot = log1p(normalise(SUM of the member cells' raw counts))
    cell = log1p(normalise(that cell's raw counts))
Two problems. log1p is concave, so log1p(mean) >= mean(log1p): a spot's profile is inflated
relative to the average of its own cells' profiles in the very space the cost is computed in.
And summing raw counts weights cells by library size, so a high-count cell dominates its spot.
Neither is a property of the biology -- a cell is not closest to its own spot even with no noise.

FAIR construction:
    normalise and log1p each spot-side cell FIRST, then average within the spot.
The spot is then exactly the centroid of its members in the same space as X_sc, so the expression
cost is minimised at the truth by construction.

Everything else -- the disjoint 50/50 split, the lattice, the tree -- is byte-identical, so any
difference is attributable to the aggregation alone.
"""
import sys, numpy as np, scanpy as sc
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_petracer as pp

sc.settings.verbosity = 0
import pycea
t = pycea.datasets.koblan25("tumor")

for SEC in ["M3-4", "M3-1", "M3-2"]:
    old = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    names_sc = np.asarray(old["cell_names"]).astype(str)
    names_sp = np.asarray(old["spot_cell_names"]).astype(str)
    spot_of = np.asarray(old["spot_of_cell"]).astype(int)     # spot index of each spot-side cell
    M = int(old["X_st"].shape[0])
    idx = {nm: i for i, nm in enumerate(np.asarray(t.obs.index, dtype=object))}
    row_sp = np.array([idx[nm] for nm in names_sp])
    Xsp_raw = t.X[row_sp]

    # normalise + log1p EACH spot-side cell, then average within the spot
    a = sc.AnnData(np.asarray(Xsp_raw.todense() if hasattr(Xsp_raw, "todense") else Xsp_raw,
                              dtype=np.float32))
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    Xcells = np.asarray(a.X)
    X_st_fair = np.zeros((M, Xcells.shape[1]), np.float32)
    for s in range(M):
        X_st_fair[s] = Xcells[spot_of == s].mean(0)

    out = {k: old[k] for k in old.files}
    out["X_st"] = X_st_fair
    np.savez_compressed(HERE / f"petracerFAIR_{SEC}.npz", **out)

    # how much does this change the expression cost?
    import ot
    X_sc = np.asarray(old["X_sc"], float)
    tru = None
    for tag, X_st in [("current (log of sum)", np.asarray(old["X_st"], float)),
                      ("fair (mean of logs)", X_st_fair.astype(float))]:
        Mc = ot.dist(X_sc, X_st)
        xy_st = np.asarray(old["xy"], float); xy_sc = np.asarray(old["xy_sc"], float)
        nn = Mc.argmin(1)
        err = float(np.median(np.linalg.norm(xy_st[nn] - xy_sc, axis=1)))
        # is a spot-side cell closest to its OWN spot? (sanity check of the construction)
        Mown = ot.dist(Xcells, X_st)
        selfhit = float(np.mean(Mown.argmin(1) == spot_of))
        print(f"{SEC:<6} {tag:<22} expr-NN placement {err:7.1f} um   "
              f"spot-side cell finds its own spot: {selfhit:.1%}")
    print()
