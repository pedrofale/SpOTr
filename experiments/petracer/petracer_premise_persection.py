"""PEtracer premise, recomputed PER SECTION.

petracer_premise.py used t.obsm["spatial"] as one coordinate frame. That frame holds three
tiled sections - DBSCAN on the saved coordinates finds three disjoint components of 2858 /
1828 / 1267 cells - so the published numbers (Mantel r = +0.199, p = 0.103; cell-pair
r = +0.092) were computed across sections. Both tests are invalidated by that:

  Test B correlates clade TERRITORY distance with tree distance, but a clade spanning two
  sections has its centroid in the gap between them. Clade centroid spread (662) is BELOW mean
  within-clade spread (1086), which is the signature of exactly this.
  Test C samples random cell pairs; most pairs are cross-section, where spatial distance
  measures tile layout rather than tissue.

Recomputed within each section separately, which is the only frame in which spatial distance
means anything.
"""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

rng = np.random.default_rng(0)
d = np.load("petracer_xy.npz", allow_pickle=True)
xy, lab, clades, D = d["xy"], d["lab"].astype(str), d["clades"].astype(str), d["D"]
cidx = {c: i for i, c in enumerate(clades)}

scale = np.ptp(xy, axis=0).max()
sec = DBSCAN(eps=0.02 * scale, min_samples=20).fit_predict(xy)
print(f"{len(xy)} cells -> sections {sorted(set(sec) - {-1})}, "
      f"sizes {[int((sec==s).sum()) for s in sorted(set(sec) - {-1})]}, "
      f"unassigned {(sec==-1).sum()}")


def coherence(P, L, k=12):
    k = min(k, len(P) - 1)
    nb = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    same = (L[nb] == L[:, None]).mean()
    p = np.array([(L == c).mean() for c in np.unique(L)])
    ch = (p ** 2).sum()
    return (same - ch) / (1 - ch)


rows = []
for s in sorted(set(sec) - {-1}):
    m = sec == s
    P, L = xy[m], lab[m]
    big = [c for c in np.unique(L) if (L == c).sum() >= 20]
    if len(big) < 4:
        print(f"section {s}: only {len(big)} clades with >=20 cells -- skipped"); continue
    keep = np.isin(L, big); P, L = P[keep], L[keep]

    coh = coherence(P, L)
    # permutation null for coherence
    nullc = np.array([coherence(P, rng.permutation(L)) for _ in range(199)])
    p_coh = (np.sum(nullc >= coh) + 1) / (len(nullc) + 1)

    ii = [cidx[c] for c in big]
    cen = np.stack([P[L == c].mean(0) for c in big])
    Ds, Dt = squareform(pdist(cen)), D[np.ix_(ii, ii)]
    iu = np.triu_indices(len(big), 1)
    r = np.corrcoef(Dt[iu], Ds[iu])[0, 1]
    null = np.array([np.corrcoef(Dt[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                     for p in (rng.permutation(len(big)) for _ in range(9999))])
    pv = (np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)

    n = len(L)
    a, b_ = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    ok = a != b_; a, b_ = a[ok], b_[ok]
    ti = np.array([cidx[c] for c in L])
    dt, ds = D[ti[a], ti[b_]], np.linalg.norm(P[a] - P[b_], axis=1)
    rc = np.corrcoef(dt, ds)[0, 1]
    nc = np.array([np.corrcoef(D[ti[p][a], ti[p][b_]], ds)[0, 1]
                   for p in (rng.permutation(n) for _ in range(199))])
    pc = (np.sum(np.abs(nc) >= abs(rc)) + 1) / (len(nc) + 1)

    rows.append(dict(section=int(s), n_cells=int(len(L)), n_clades=len(big),
                     coherence=coh, coh_p=p_coh, mantel_r=r, mantel_p=pv,
                     pair_r=rc, pair_p=pc))
    print(f"section {s}: n={len(L):>4} clades={len(big):>2}  coherence {coh:.3f} (p={p_coh:.3f})"
          f"  Mantel {r:+.3f} (p={pv:.4f})  cell-pair {rc:+.4f} (p={pc:.3f})")

r = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nmean coherence {r.coherence.mean():.3f}   mean Mantel {r.mantel_r.mean():+.3f}   "
      f"sections with Mantel p<0.05: {(r.mantel_p < 0.05).sum()}/{len(r)}")
print(f"sign agreement across sections: {'YES' if (np.sign(r.mantel_r) == np.sign(r.mantel_r.iloc[0])).all() else 'NO'}"
      "   <- BCSA failed exactly here")
print("\nfor reference:")
print("  pooled across tiled sections (the old, invalid number): coherence 0.310, "
      "Mantel +0.199 p=0.103, cell-pair +0.092")
print("  eTRACER late escape: coherence 0.309, mean per-clone r 0.054, 18/54 clones significant")
print("  BCSA (best): coherence 0.064, does not replicate across sections")
r.to_csv("petracer_premise_persection.csv", index=False)
