"""IS THERE AN INTERPRETABLE LAW BEHIND THE TREE-SPACE RELATIONSHIP?

Pedro's redirect: we do not want a fitted, uninterpretable transform per tumour.  We want a cost
DERIVED from what lineage means, with parameters that carry units and transfer.

The generative statement is not "tree distance equals spatial distance".  It is: two cells that
shared an ancestor tau days ago started at the SAME PLACE and have been moving independently ever
since.  Under a random walk with diffusion coefficient D, in 2 dimensions,

    E || x_i - x_k ||^2  =  4 D (T - tau_ik)  =  2 D t_ik        (t = patristic distance = 2(T-tau))

so it is the SQUARED separation that is linear in patristic distance, not the separation itself --
and the tree predicts a VARIANCE, not a distance.  Under confinement (tumour of finite size) the
walk is Ornstein-Uhlenbeck instead and the variance SATURATES:

    E || x_i - x_k ||^2  =  2 s_inf^2 ( 1 - exp(-t / t_c) )

This script tests both against all three PEtracer sections, and compares them with what GW assumes
(E||x_i - x_k|| linear in t).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
HERE = Path(__file__).resolve().parent

def bins_of(t, nb=24):
    cap = t.max(); below = t[t < cap - 1e-9]
    e = np.linspace(below.min(), below.max(), nb)
    return np.r_[e[:-1], cap - 1e-9, cap + 1e-9]

print("Binned E[s^2 | t] and E[s | t]; R^2 of three models fitted on the BIN MEANS")
print("(weights = number of pairs per bin, so the fit is not dominated by sparse short-divergence")
print(" bins).  brownian: 2*D*t   OU: 2*s2*(1-exp(-t/tc))   GW-style: (a*t+b)^2\n")
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    t = CT[np.triu_indices(len(xy), 1)]; s = pdist(xy)
    e = bins_of(t)
    tc, m2, m1, n = [], [], [], []
    for i in range(len(e) - 1):
        m = (t >= e[i]) & (t < e[i + 1])
        if m.sum() < 200:
            continue
        tc.append(t[m].mean()); m2.append((s[m] ** 2).mean()); m1.append(s[m].mean()); n.append(m.sum())
    tc, m2, m1, n = map(np.array, (tc, m2, m1, n))
    w = n / n.sum()

    def r2(pred):
        return 1 - np.sum(w * (m2 - pred) ** 2) / np.sum(w * (m2 - (w * m2).sum()) ** 2)

    D = np.sum(w * tc * m2) / np.sum(w * tc * tc) / 2.0            # least squares through origin
    r2_bm = r2(2 * D * tc)
    try:
        p, _ = curve_fit(lambda x, s2, tcst: 2 * s2 * (1 - np.exp(-x / tcst)), tc, m2,
                         p0=[m2.max() / 2, 30.0], sigma=1 / np.sqrt(n), maxfev=20000)
        r2_ou = r2(2 * p[0] * (1 - np.exp(-tc / p[1])))
        ou = f"s_inf {np.sqrt(p[0]):.0f} um, t_c {p[1]:.1f} d"
    except Exception as ex:
        r2_ou, ou, p = np.nan, str(ex)[:30], [np.nan, np.nan]
    a, b = np.polyfit(tc, m1, 1, w=np.sqrt(n))
    r2_gw = r2((a * tc + b) ** 2)
    print(f"{SEC}:  brownian R2 {r2_bm:.3f}  (D = {D:.0f} um^2/day)"
          f"   |  OU R2 {r2_ou:.3f}  ({ou})"
          f"   |  GW-style R2 {r2_gw:.3f}")
