"""Does PEtracer per-clade spatial structure REPLICATE across serial sections?

Claim (b) needs alpha_k to be a stable property of clade k. The three M3 sections are
disjoint cell sets sharing the SAME 18 clade labels (verified), so they are independent
measurements of the same quantity. If per-clade coherence does not correlate across
sections, no objective can learn alpha_k, because there is nothing stable to learn.

This is the serial-section test NOTES flagged as VALID but never ran.
"""
import numpy as np, pandas as pd, itertools
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

rng = np.random.default_rng(0)
d = np.load("petracer_xy.npz", allow_pickle=True)
xy, lab, clades, D = d["xy"], d["lab"].astype(str), d["clades"].astype(str), d["D"]

scale = np.ptp(xy, axis=0).max()
sec = DBSCAN(eps=0.02 * scale, min_samples=20).fit_predict(xy)
secs = sorted(set(sec) - {-1})
NAME = {0: "M3-1", 1: "M3-2", 2: "M3-4"}
print(f"{len(xy)} cells -> sections {[int((sec==s).sum()) for s in secs]}")


def perclade_coherence(P, L, k=12):
    """Per-clade kNN same-label excess over chance, normalised."""
    k = min(k, len(P) - 1)
    nb = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    same = (L[nb] == L[:, None]).mean(axis=1)          # per cell
    out = {}
    for c in np.unique(L):
        m = L == c
        p = m.mean()
        out[c] = (same[m].mean() - p) / (1 - p)
    return out


def perclade_spread(P, L):
    """Mean within-clade pairwise distance / mean overall pairwise distance (lower = compact)."""
    allmed = np.median(pdist(P[rng.choice(len(P), min(len(P), 1500), replace=False)]))
    out = {}
    for c in np.unique(L):
        Q = P[L == c]
        if len(Q) > 1200: Q = Q[rng.choice(len(Q), 1200, replace=False)]
        out[c] = np.median(pdist(Q)) / allmed
    return out


COH, SPR, N = {}, {}, {}
for s in secs:
    m = sec == s
    P, L = xy[m], lab[m]
    big = [c for c in np.unique(L) if (L == c).sum() >= 20]
    keep = np.isin(L, big); P, L = P[keep], L[keep]
    COH[NAME[s]] = perclade_coherence(P, L)
    SPR[NAME[s]] = perclade_spread(P, L)
    N[NAME[s]] = {c: int((L == c).sum()) for c in big}

names = [NAME[s] for s in secs]
shared = sorted(set.intersection(*[set(COH[n]) for n in names]), key=int)
print(f"clades measurable in all three sections: {len(shared)} -> {shared}\n")

tab = pd.DataFrame({"clade": shared})
for n in names:
    tab[f"n_{n}"]   = [N[n][c] for c in shared]
    tab[f"coh_{n}"] = [COH[n][c] for c in shared]
    tab[f"spr_{n}"] = [SPR[n][c] for c in shared]
print("PER-CLADE SPATIAL COHERENCE (kNN same-clade excess over chance)")
print(tab[["clade"] + [f"n_{n}" for n in names] + [f"coh_{n}" for n in names]]
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n" + "=" * 74)
print("CROSS-SECTION REPLICATION of the per-clade vector")
for stat, pre in [("coherence", "coh"), ("spread", "spr")]:
    print(f"\n  {stat}:")
    rs = []
    for a, b in itertools.combinations(names, 2):
        x = np.array([COH[a][c] if pre == "coh" else SPR[a][c] for c in shared])
        y = np.array([COH[b][c] if pre == "coh" else SPR[b][c] for c in shared])
        r, p = pearsonr(x, y); rho, pr = spearmanr(x, y)
        rs.append(r)
        print(f"    {a} vs {b}:  pearson r={r:+.3f} (p={p:.3f})   spearman rho={rho:+.3f} (p={pr:.3f})")
    print(f"    mean pairwise r = {np.mean(rs):+.3f}")

# within-section reliability ceiling: random split-half of the SAME section
print("\n  within-section split-half ceiling (how reliable is the measurement itself):")
for s in secs:
    m = sec == s; P, L = xy[m], lab[m]
    big = [c for c in np.unique(L) if (L == c).sum() >= 40]
    keep = np.isin(L, big); P, L = P[keep], L[keep]
    h = rng.random(len(L)) < 0.5
    a = perclade_coherence(P[h], L[h]); b = perclade_coherence(P[~h], L[~h])
    cc = sorted(set(a) & set(b), key=int)
    r, p = pearsonr([a[c] for c in cc], [b[c] for c in cc])
    print(f"    {NAME[s]}: r={r:+.3f} (p={p:.3f}) over {len(cc)} clades")

tab.to_csv("petracer_perclade_replication.csv", index=False)
print("\nwrote petracer_perclade_replication.csv")
