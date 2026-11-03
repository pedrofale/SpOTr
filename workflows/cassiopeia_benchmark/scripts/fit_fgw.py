"""Fused Gromov-Wasserstein on one simulated problem, with SpOTr's distance matrices (decision-paper-results-structure: "FGW using the same
distance matrices as SpOTr", uniform spot capacities).  Run in the spotr environment (CPU).

The traced dissociated cells (the structure needs their tree) against every spot: feature cost the cosine distance between the cells' and the
spots' log-normalised profiles (SpOTr's expression metric, on the whole spot since FGW has no deconvolution); structures the tree's distances
between the cells (true or reconstructed, as given to SpOTr) and the Euclidean distances between the spots, each divided by its maximum; uniform
masses on cells and spots; POT's fused_gromov_wasserstein, square loss, alpha 0.5 (POT's default).  Rows of untraced cells are zero, so that
score.py reads it as any other map of cells.

Usage: python fit_fgw.py <params.json> <problem.npz> <trees.npz> <out.npz>
"""
import sys, json
import numpy as np


def main(pfile, prob, trees, out):
    import ot
    from scipy.spatial.distance import pdist, squareform
    p = json.load(open(pfile)); q, t = np.load(prob), np.load(trees); tr = q["traced"].astype(bool)
    d = np.asarray(t["D_rec"] if p["tree"] == "reconstructed" else t["D_true"], float); Xc, Y = q["Xc"].astype(float)[tr], q["Y"].astype(float)
    ln = lambda A: np.log1p(A / np.maximum(A.sum(1, keepdims=True), 1) * 1e4); a, b = ln(Xc), ln(Y)
    Mf = 1.0 - (a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)) @ (b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)).T
    CS = squareform(pdist(q["spots"].astype(float))); C1, C2 = d / d.max(), CS / CS.max()
    G = ot.gromov.fused_gromov_wasserstein(Mf, C1, C2, np.full(len(Xc), 1.0 / len(Xc)), np.full(len(Y), 1.0 / len(Y)), loss_fun="square_loss", alpha=0.5)
    P = np.zeros((len(tr), len(Y)), np.float32); P[np.where(tr)[0]] = np.asarray(G, float) * len(Xc); np.savez_compressed(out, P=P); print(f"FGW: {G.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
