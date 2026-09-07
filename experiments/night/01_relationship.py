"""WHAT IS THE RELATIONSHIP BETWEEN TREE DISTANCE AND SPATIAL DISTANCE?

Square-loss GW reads the tree through exactly one scalar: corr(tree distance, spatial distance
between the images).  So the question "how much can the tree possibly help" becomes "how large
can that correlation be made".

Two transforms are free:  phi on the tree side, psi on the spatial side.  The pair maximising
corr(phi(t), psi(s)) is the classical MAXIMAL CORRELATION (Hirschfeld-Gebelein-Renyi), and it is
computed by a single SVD of the normalised joint histogram -- correspondence analysis.

    P        joint histogram of (t, s) over all cell pairs, rows a = p(t), cols b = p(s)
    Q        P / sqrt(a b')       ->   svd(Q) = sum_r sigma_r u_r v_r'
    sigma_0  = 1 with u_0 = sqrt(a), v_0 = sqrt(b)   (the trivial component)
    rho_r    = sigma_r for r >= 1   are the canonical correlations
    phi_r    = u_r / sqrt(a),  psi_r = v_r / sqrt(b)   standardised transforms

rho_1 is the maximal correlation.  Comparing it with the raw Pearson correlation says how much
signal a square loss on the raw distances is leaving on the table.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
PET = HERE.parent / "petracer"
SECS = ["M3-4", "M3-1", "M3-2"]
NT, NS = 24, 24


def tbins(t, nt=NT):
    """tree-distance bin edges.  The saturation ceiling is a point mass -- give it its own bin."""
    cap = t.max()
    below = t[t < cap - 1e-9]
    e = np.unique(np.quantile(below, np.linspace(0, 1, nt)))
    return np.r_[e[:-1], cap - 1e-9, cap + 1e-9]


def joint(t, s, te, se):
    P, _, _ = np.histogram2d(t, s, bins=[te, se])
    P = P / P.sum()
    return P


def ca(P, ncomp=4):
    a = P.sum(1); b = P.sum(0)
    keep_r = a > 0; keep_c = b > 0
    Pk = P[np.ix_(keep_r, keep_c)]; ak = a[keep_r]; bk = b[keep_c]
    Q = Pk / np.sqrt(np.outer(ak, bk))
    U, S, Vt = np.linalg.svd(Q, full_matrices=False)
    out = dict(rho=S[1:1 + ncomp], a=a, b=b, keep_r=keep_r, keep_c=keep_c)
    phi = np.zeros((ncomp, len(a))); psi = np.zeros((ncomp, len(b)))
    for r in range(ncomp):
        phi[r, keep_r] = U[:, r + 1] / np.sqrt(ak)
        psi[r, keep_c] = Vt[r + 1] / np.sqrt(bk)
        if np.corrcoef(phi[r, keep_r], np.arange(keep_r.sum()))[0, 1] < 0:
            phi[r] *= -1; psi[r] *= -1          # sign convention: phi increasing in t
    out["phi"] = phi; out["psi"] = psi
    return out


rows = []
store = {}
for SEC in SECS:
    d = np.load(PET / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    iu = np.triu_indices(len(xy), 1)
    t = CT[iu]; s = pdist(xy)
    te = tbins(t); se = np.r_[np.quantile(s, np.linspace(0, 1, NS)), s.max() + 1]
    se = np.unique(se)
    P = joint(t, s, te, se)
    C = ca(P)
    tc = 0.5 * (te[:-1] + te[1:]); sc = 0.5 * (se[:-1] + se[1:])
    # E[s|t] and sd[s|t]
    a = P.sum(1)
    Es = np.where(a > 0, (P * sc[None, :]).sum(1) / np.maximum(a, 1e-12), np.nan)
    Es2 = np.where(a > 0, (P * sc[None, :] ** 2).sum(1) / np.maximum(a, 1e-12), np.nan)
    SD = np.sqrt(np.maximum(Es2 - Es ** 2, 0))
    raw = np.corrcoef(t, s)[0, 1]
    # correlation achieved by phi = E[s|t] (best transform of t alone, psi = identity)
    phi_mean = np.interp(t, tc, np.nan_to_num(Es, nan=np.nanmean(Es)))
    one_sided = np.corrcoef(phi_mean, s)[0, 1]
    store[SEC] = dict(te=te, se=se, tc=tc, sc=sc, P=P, C=C, Es=Es, SD=SD, a=a,
                      t=t, s=s, raw=raw)
    rows.append(dict(section=SEC, n_pairs=len(t), raw_corr=raw,
                     one_sided_max=one_sided, rho1=C["rho"][0], rho2=C["rho"][1],
                     rho3=C["rho"][2], frac_at_cap=float((t >= t.max() - 1e-9).mean())))
    print(f"{SEC}: raw corr {raw:+.4f} | best phi(t) only {one_sided:+.4f} | "
          f"maximal corr rho1 {C['rho'][0]:.4f}  rho2 {C['rho'][1]:.4f}  rho3 {C['rho'][2]:.4f}")

R = pd.DataFrame(rows); R.to_csv(HERE / "relationship.csv", index=False)
np.save(HERE / "relationship.npy", store, allow_pickle=True)

print("\nCross-section stability of the learned transforms (correlation of phi_1 over shared bins)")
for i in range(len(SECS)):
    for j in range(i + 1, len(SECS)):
        A, B = store[SECS[i]], store[SECS[j]]
        g = np.linspace(0.7, 70, 200)
        pa = np.interp(g, A["tc"], A["C"]["phi"][0]); pb = np.interp(g, B["tc"], B["C"]["phi"][0])
        sa = np.interp(np.linspace(0, 3000, 200), A["sc"], A["C"]["psi"][0])
        sb = np.interp(np.linspace(0, 3000, 200), B["sc"], B["C"]["psi"][0])
        print(f"  {SECS[i]} vs {SECS[j]}:  phi1 corr {np.corrcoef(pa, pb)[0,1]:+.3f}   "
              f"psi1 corr {np.corrcoef(sa, sb)[0,1]:+.3f}")
