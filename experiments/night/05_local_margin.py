"""The same margin test, but with the corruption the SOLVER actually has to resolve.

Shuffling cells across the whole section is a caricature: every loss detects it.  The mapping
error we are trying to reduce is ~360 um, which is 1-3 spot pitches, so the question is whether a
loss can tell a cell's true spot from a NEARBY one.  Corrupt by moving a fraction of the cells to
a uniformly chosen spot within a radius r, and read the slope at small displacements.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import fit_kernel, tree_bins, space_bins

SECS = ["M3-4", "M3-1", "M3-2"]
RADII = [0, 150, 300, 500, 800, 1200, 2000, 1e9]
NDRAW = 10
rng = np.random.default_rng(0)


def build_losses(t, s_ref, te, se):
    P, _, _ = np.histogram2d(t, s_ref, bins=[te, se]); P /= P.sum()
    a = P.sum(1)[:, None]; b = P.sum(0)[None, :]
    tc = 0.5 * (te[:-1] + te[1:]); sc = 0.5 * (se[:-1] + se[1:])
    pa = P.sum(1); pb = P.sum(0)
    ts = (tc - (tc*pa).sum()) / np.sqrt(max(((tc**2)*pa).sum() - ((tc*pa).sum())**2, 1e-12))
    ss = (sc - (sc*pb).sum()) / np.sqrt(max(((sc**2)*pb).sum() - ((sc*pb).sum())**2, 1e-12))
    out = {"square loss (FGW)": (ts[:, None] - ss[None, :]) ** 2}
    K = fit_kernel(t, s_ref, ncomp=3, te=te, se=se)
    for R in (1, 2, 3):
        out[f"HGR rank {R}"] = -sum(K["rho"][r] * np.outer(K["phi"][r], K["psi"][r])
                                    for r in range(R))
    out["-PMI"] = -np.log(np.where(P > 0, P / np.maximum(a*b, 1e-300), 1e-6))
    w = a * b
    for k in out:
        L = out[k]; mu = float((L*w).sum())
        out[k] = (L - mu) / float(np.sqrt(max((L**2*w).sum() - mu**2, 1e-30)))
    return out


rows = []
for SEC in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float)
    xy_sc = np.asarray(d["xy_sc"], float); xy_st = np.asarray(d["xy"], float)
    n, m = len(xy_sc), len(xy_st)
    CSpot = squareform(pdist(xy_st))
    iu = np.triu_indices(n, 1); t = CT[iu]
    te = tree_bins(t); se = space_bins(pdist(xy_sc))
    L = build_losses(t, pdist(xy_sc), te, se)
    true_spot = np.argmin(((xy_sc[:, None, :] - xy_st[None, :, :])**2).sum(2), axis=1)
    near = [np.where(CSpot[j] <= r)[0] for r in RADII[1:-1] for j in [0]]  # placeholder

    def score(assign):
        sm = CSpot[assign[iu[0]], assign[iu[1]]]
        H, _, _ = np.histogram2d(t, sm, bins=[te, se]); H /= H.sum()
        return {k: float((v*H).sum()) for k, v in L.items()}

    base = score(true_spot)
    rnd = {k: v for k, v in zip(L, np.mean([[score(rng.permutation(true_spot))[k] for k in L]
                                            for _ in range(6)], axis=0))}
    for r in RADII:
        cand = [np.where(CSpot[j] <= r)[0] for j in range(m)] if r < 1e8 else None
        acc = {k: [] for k in L}; disp = []
        for _ in range(NDRAW if r > 0 else 1):
            a2 = true_spot.copy()
            if r > 0:
                a2 = np.array([rng.choice(cand[j]) if cand is not None else rng.integers(m)
                               for j in true_spot])
            disp.append(np.median(np.linalg.norm(xy_st[a2] - xy_sc, axis=1)))
            sc_ = score(a2)
            for k in L: acc[k].append(sc_[k])
        for k in L:
            rows.append(dict(section=SEC, loss=k, radius=r, disp=np.mean(disp),
                             norm=(np.mean(acc[k]) - base[k]) / (rnd[k] - base[k])))
    print(f"{SEC} done", flush=True)

R = pd.DataFrame(rows); R.to_csv(HERE / "local_margin.csv", index=False)
print("\nLoss increase (1.0 = random map) as every cell is moved to a random spot within radius r")
for SEC in SECS:
    g = R[R.section == SEC]
    p = g.pivot_table(index="disp", columns="loss", values="norm")
    print(f"\n{SEC}   (index = median displacement in um)")
    print(p.round(4).to_string())
    sub = p[p.index < 700]
    if len(sub) > 1:
        print("  slope over the first 700 um, per 100 um:")
        print("   " + "  ".join(f"{c}={np.polyfit(sub.index, sub[c], 1)[0]*100:.4f}"
                                for c in sub.columns))
