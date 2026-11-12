"""The other methods on one simulated tumor (environment benchmark_env, with cell2location in its own environment through common/c2l.py).

The methods that take labeled dissociated cells (cell2location, DestVI and scDOT) get world.labels, with one label for each non-tumor type, one
for each clone (with the smaller clades pooled as one label) and one for the tumor cells without a lineage readout.

  c2l       cell2location (common/c2l.py, deconvolve), with N_cells_per_location the mean number of cells per spot. Writes the abundance of each
            label in each spot.
  destvi    DestVI (common/baselines.py). Writes the proportion of each label in each spot.
  scdot     scDOT. Writes its map of the dissociated cells.
  tangram   Tangram in mode "cells", with the true numbers of cells in the spots as its density prior.
  exact_ot  Exact optimal transport under the cosine distance between log-normalized profiles, with the true numbers of cells in the spots as
            their masses.
  fgw       Fused Gromov-Wasserstein (POT, square loss, alpha 0.5) of the traced cells against the spots. The feature cost is the cosine distance
            between log-normalized profiles, the structures are the tree distances and the Euclidean distances between spots, each divided by its
            maximum, and the masses are uniform. The rows of the other cells are zero.

Writes w and types (abundances or proportions, spots x labels) for c2l and DestVI, and P (dissociated cells x spots) for the maps.

Usage: python fit_baselines.py <method> <seed folder> <seed> <out.npz> [--epochs N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths, baselines, c2l  # noqa: E402
import world  # noqa: E402


def fgw(q, t):
    import ot
    from scipy.spatial.distance import pdist, squareform
    tr = q["traced"].astype(bool); a, b = baselines.lognorm(q["Xc"][tr]), baselines.lognorm(q["Y"])
    Mf = 1.0 - (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)) @ (b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)).T
    d = np.asarray(t["D_true"], float); CS = squareform(pdist(q["spots"].astype(float))); n, S = len(a), len(b)
    G = ot.gromov.fused_gromov_wasserstein(Mf, d / d.max(), CS / CS.max(), np.full(n, 1.0 / n), np.full(S, 1.0 / S), loss_fun="square_loss", alpha=0.5)
    P = np.zeros((len(tr), S)); P[np.where(tr)[0]] = np.asarray(G, float) * n; return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("method", choices=list(world.METHODS)); ap.add_argument("folder"); ap.add_argument("seed", type=int)
    ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=None); a = ap.parse_args()
    d = Path(a.folder); q, t = np.load(d / "problem.npz", allow_pickle=True), np.load(d / "trees.npz")
    Xc, Y, lab, mass = q["Xc"].astype(float), q["Y"].astype(float), world.labels(q, t), q["ncell"].astype(float)
    if a.method == "c2l":
        w, types = c2l.deconvolve(Xc, lab, Y, float(q["ncell"].mean()), a.out, epochs=a.epochs or 30000); print(f"cell2location: {len(types)} labels"); return
    if a.method == "destvi":
        w, types = baselines.destvi(Xc, lab, Y, seed=a.seed); np.savez_compressed(a.out, w=w.astype(np.float32), types=np.array(types)); return
    P = {"scdot": lambda: baselines.scdot(Xc, lab, Y, seed=a.seed), "tangram": lambda: baselines.tangram(Xc, Y, mass, epochs=a.epochs or 1000, seed=a.seed),
         "exact_ot": lambda: baselines.exact_ot(Xc, Y, mass), "fgw": lambda: fgw(q, t)}[a.method]()
    np.savez_compressed(a.out, P=np.asarray(P, np.float32)); print(f"{a.method}: {P.shape}")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
