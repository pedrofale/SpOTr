"""scDOT on one simulated problem, from its upstream code unmodified (benchmark/packages.py: github.com/namtk/scDOT at da74a8a, README's
training loop, its own map read out).  Cells labelled as for DestVI and cell2location (each clone of the tree given to SpOTr, pooled, untraced,
each microenvironment type).  Run in the spotr environment (CPU).

Usage: python fit_scdot.py <params.json> <problem.npz> <trees.npz> <seed> <out.npz>   (writes P: cells x spots, each cell's distribution)
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark"))


def main(pfile, prob, trees, seed, out):
    import packages
    p = json.load(open(pfile)); q, t = np.load(prob), np.load(trees); rec = p["tree"] == "reconstructed"
    clone = (t["clone_rec_tree"] if rec else t["clone_true_tree"]).astype(int); kind, me, tr = q["kind"], q["me_type"], q["traced"].astype(bool)
    labs = np.array([f"me{e}" for e in me], dtype=object); labs[(kind == 0) & ~tr] = "untraced"; labs[tr] = [f"c{c}" if c >= 0 else "pooled" for c in clone]
    P = packages.scdot(q["Xc"].astype(float), labs, q["Y"].astype(float), seed=int(seed)); np.savez_compressed(out, P=P.astype(np.float32)); print(f"scDOT: {P.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
