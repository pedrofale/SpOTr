"""Does the lineage tree predict spatial position? Tested on PEtracer 4T1 tumours.

koblan25 (Koblan et al. 2025) reads a prime-editing lineage barcode out IN SITU, so lineage
and position are measured in the SAME cells. That makes it the positive control for SpOTr's
core premise -- the Gromov-Wasserstein term only helps if tree distance tracks spatial
distance. Same two tests used on BCSA and the GBM cohort, so numbers are comparable.
"""
import numpy as np, pandas as pd, networkx as nx
import pycea as py
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

rng = np.random.default_rng(0)


def coherence(xy, lab, k=12):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    same = (lab[nbrs] == lab[:, None]).mean()
    p = np.array([(lab == c).mean() for c in np.unique(lab)])
    chance = (p ** 2).sum()
    return (same - chance) / (1 - chance)


def main():
    t = py.datasets.koblan25("tumor")
    xy_all = np.asarray(t.obsm["spatial"], dtype=float)
    clade = t.obs["clade"].astype(str).to_numpy()
    has = (clade != "nan") & (clade != "None") & pd.notna(t.obs["clade"]).to_numpy()
    print(f"[data] {t.n_obs} cells total, {has.sum()} with a lineage clade, "
          f"{len(np.unique(clade[has]))} clades")

    xy, lab = xy_all[has], clade[has]
    names = np.asarray(t.obs_names)[has]

    # ---- tree distances between clades, from the actual lineage tree ----
    G = t.obst["tree"]
    U = G.to_undirected()
    leaf_of = {n: n for n in U.nodes}
    clades = sorted(np.unique(lab), key=lambda s: int(s) if s.isdigit() else 1e9)
    # representative leaves per clade (cap for tractability)
    reps = {c: [n for n in names[lab == c] if n in U][:40] for c in clades}
    reps = {c: v for c, v in reps.items() if len(v) >= 5}
    clades = [c for c in clades if c in reps]
    print(f"[tree] {len(clades)} clades with >=5 mapped leaves")

    D = np.zeros((len(clades), len(clades)))
    for i, a in enumerate(clades):
        lengths = {}
        for src in reps[a]:
            lengths[src] = nx.single_source_shortest_path_length(U, src, cutoff=200)
        for j, b in enumerate(clades):
            if j < i:
                D[i, j] = D[j, i]; continue
            vals = [lengths[s][d] for s in reps[a] for d in reps[b] if d in lengths[s]]
            D[i, j] = np.mean(vals) if vals else np.nan
    ok = ~np.isnan(D).any(1)
    clades = [c for c, k in zip(clades, ok) if k]
    D = D[np.ix_(ok, ok)]
    print(f"[tree] clade distance matrix {D.shape}, range {np.nanmin(D):.1f}-{np.nanmax(D):.1f}")

    keep = np.isin(lab, clades)
    xy, lab = xy[keep], lab[keep]

    # ---- Test A: spatial coherence of clade labels ----
    coh = coherence(xy, lab)
    print(f"\n[Test A] clade spatial coherence = {coh:.3f}   "
          f"(BCSA 0.03-0.17, GBM best 0.35-0.80)")

    # ---- Test B: Mantel, clade tree distance vs clade territory distance ----
    big = [c for c in clades if (lab == c).sum() >= 20]
    idx = [clades.index(c) for c in big]
    cent = np.stack([xy[lab == c].mean(0) for c in big])
    Ds = squareform(pdist(cent)); Dt = D[np.ix_(idx, idx)]
    iu = np.triu_indices(len(big), 1)
    r = np.corrcoef(Dt[iu], Ds[iu])[0, 1]
    null = np.array([np.corrcoef(Dt[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                     for p in (rng.permutation(len(big)) for _ in range(20000))])
    pv = (np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)
    print(f"[Test B] Mantel r(tree, territory) = {r:+.3f}, p = {pv:.4f}  ({len(big)} clades)")

    # ---- Test C: cell pairs ----
    n = len(lab)
    ii, jj = rng.integers(0, n, 400_000), rng.integers(0, n, 400_000)
    m = ii != jj; ii, jj = ii[m], jj[m]
    ti = np.array([clades.index(c) for c in lab])
    dt = D[ti[ii], ti[jj]]
    ds = np.linalg.norm(xy[ii] - xy[jj], axis=1)
    rc = np.corrcoef(dt, ds)[0, 1]
    nullc = np.array([np.corrcoef(D[ti[p][ii], ti[p][jj]], ds)[0, 1]
                      for p in (rng.permutation(n) for _ in range(200))])
    pc = (np.sum(np.abs(nullc) >= abs(rc)) + 1) / (len(nullc) + 1)
    print(f"[Test C] cell pairs r(tree, spatial) = {rc:+.4f}, p = {pc:.4f}  "
          f"(BCSA 0.015-0.115)")

    pd.DataFrame([dict(dataset="PEtracer_4T1_tumor", n_cells=int(len(lab)), n_clades=len(clades),
                       coherence=coh, mantel_r=r, mantel_p=pv, pair_r=rc, pair_p=pc)]
                 ).to_csv("petracer_premise.csv", index=False)
    np.savez_compressed("petracer_xy.npz", xy=xy, lab=lab, clades=np.array(clades), D=D)
    print("\nwrote petracer_premise.csv")


if __name__ == "__main__":
    main()
