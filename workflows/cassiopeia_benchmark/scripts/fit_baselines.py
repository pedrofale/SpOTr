"""Expression-only baselines on one simulated problem (run in the spotr environment): exact optimal transport and Tangram.

Both map all dissociated cells to the spots, with spot masses proportional to the spots' true numbers of cells.  Exact OT: POT's emd under the
cosine distance between log-normalised profiles (cells and spots each normalised to 10^4 counts), SpOTr's expression cost as spotr.tex states
(squared Euclidean, POT's default, until 2026-11-03).  Tangram: map_cells_to_space,
mode "cells", density prior from those masses (as experiments/night/73_tangram.py).

Usage: python fit_baselines.py <problem.npz> <method: exact_ot|tangram> <epochs> <seed> <out.npz>
"""
import os, sys
import numpy as np


def main(prob, method, epochs, seed, out):
    q = np.load(prob); Xc, Y = q["Xc"].astype(float), q["Y"].astype(float); mass = q["ncell"].astype(float); mass /= mass.sum()
    if method == "exact_ot":
        import ot
        lx = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4); ly = np.log1p(Y / np.maximum(Y.sum(1, keepdims=True), 1) * 1e4)
        P = ot.emd(np.full(len(Xc), 1.0 / len(Xc)), mass, ot.dist(lx, ly, metric="cosine"), numItermax=10 ** 7) * len(Xc)
    else:
        import anndata as ad, tangram as tg
        genes = [f"g{i}" for i in range(Xc.shape[1])]; a_sc = ad.AnnData(Xc.astype(np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(Y.astype(np.float32)); a_sp.var_names = genes
        tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = mass
        P = np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=int(epochs),
                                             device=os.environ.get("TANGRAM_DEVICE", "cuda"), random_state=int(seed), verbose=False).X, float)
    np.savez_compressed(out, P=P.astype(np.float32)); print(f"{method}: {P.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
