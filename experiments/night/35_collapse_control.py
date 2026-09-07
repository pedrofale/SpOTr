"""CONTROL FOR THE COLLAPSE.

s / r(a) is built so that E[(s/r)^2] = 2 in every age bin, so the second moment is matched BY
CONSTRUCTION.  The collapse is only meaningful if distributions that genuinely differ in SHAPE
still separate after the same rescaling.  Compare the observed rescaled law against three
alternatives with the SAME second moment:

  Rayleigh              what free diffusion from a common origin predicts
  two uniform points    in a disc -- what "the tree says nothing" looks like
  half-normal           a heavier-tailed alternative

If these sit far from the observed law while six real/simulated datasets sit at 0.01, the collapse
is a finding rather than an artefact of normalisation.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def rescaled(t, s, nb=22):
    a = t / 2.0; cap = a.max(); lo = a[a < cap - 1e-9]
    e = np.r_[np.linspace(lo.min(), lo.max(), nb), cap - 1e-9, cap + 1e-9]
    ac, r = [], []
    for i in range(len(e) - 1):
        m = (a >= e[i]) & (a < e[i + 1])
        if m.sum() < 200:
            continue
        ac.append(a[m].mean()); r.append(np.sqrt((s[m] ** 2).mean() / 2))
    return s / np.interp(a, np.array(ac), np.array(r))


d = np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)
CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
z = rescaled(CT[np.triu_indices(len(xy), 1)], pdist(xy))
z = rng.choice(z, 40000, replace=False)
m2 = (z ** 2).mean()
print(f"observed rescaled law: n={len(z)}, E[z^2]={m2:.3f} (2 by construction), "
      f"mean {z.mean():.3f}, skew-ish q90/q50 {np.quantile(z,.9)/np.median(z):.3f}\n")

alts = {}
r = rng.rayleigh(1.0, 200000); alts["Rayleigh (free diffusion)"] = r * np.sqrt(m2 / (r**2).mean())
p1 = rng.random((200000, 2)); p2 = rng.random((200000, 2))
th1, u1 = rng.random(200000) * 2 * np.pi, np.sqrt(rng.random(200000))
th2, u2 = rng.random(200000) * 2 * np.pi, np.sqrt(rng.random(200000))
dd = np.hypot(u1*np.cos(th1) - u2*np.cos(th2), u1*np.sin(th1) - u2*np.sin(th2))
alts["two uniform points in a disc"] = dd * np.sqrt(m2 / (dd**2).mean())
h = np.abs(rng.normal(0, 1, 200000)); alts["half-normal"] = h * np.sqrt(m2 / (h**2).mean())

print("KS of each alternative against the observed rescaled law (same second moment):")
for k, v in alts.items():
    print(f"   {k:<30} KS = {ks_2samp(z, v[:40000]).statistic:.3f}")
print("\nfor reference, two halves of the observed law itself:")
a_, b_ = z[:20000], z[20000:]
print(f"   {'split-half of the data':<30} KS = {ks_2samp(a_, b_).statistic:.3f}")
print(f"   {'six real/simulated datasets':<30} KS = 0.010 - 0.016  (from 34_collapse.py)")
