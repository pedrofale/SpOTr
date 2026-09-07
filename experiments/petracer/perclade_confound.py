"""Is per-clade coherence a property of the CLADE, or of its abundance in that section?"""
import numpy as np, pandas as pd, itertools
from scipy.stats import pearsonr, spearmanr

t = pd.read_csv("petracer_perclade_replication.csv")
names = ["M3-1", "M3-2", "M3-4"]

print("1) Within each section: does coherence track abundance?")
for n in names:
    r, p = pearsonr(t[f"n_{n}"], t[f"coh_{n}"])
    rho, pr = spearmanr(t[f"n_{n}"], t[f"coh_{n}"])
    print(f"   {n}: pearson r={r:+.3f} (p={p:.3f})   spearman rho={rho:+.3f} (p={pr:.3f})")

print("\n2) Does clade ABUNDANCE itself replicate across sections?")
for a, b in itertools.combinations(names, 2):
    r, p = pearsonr(t[f"n_{a}"], t[f"n_{b}"])
    rho, _ = spearmanr(t[f"n_{a}"], t[f"n_{b}"])
    print(f"   {a} vs {b}: pearson r={r:+.3f} (p={p:.3f})  spearman rho={rho:+.3f}")

print("\n3) Clonal composition TVD between sections (0 = identical mix):")
F = {n: t[f"n_{n}"].values / t[f"n_{n}"].sum() for n in names}
for a, b in itertools.combinations(names, 2):
    print(f"   {a} vs {b}: TVD = {0.5*np.abs(F[a]-F[b]).sum():.3f}")

print("\n4) Cross-section coherence correlation, PARTIALLING OUT log-abundance in both sections:")
def partial(x, y, Z):
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(z, float) for z in Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(rx, ry)
for a, b in itertools.combinations(names, 2):
    x, y = t[f"coh_{a}"].values, t[f"coh_{b}"].values
    r0, p0 = pearsonr(x, y)
    r1, p1 = partial(x, y, [np.log(t[f"n_{a}"]), np.log(t[f"n_{b}"])])
    print(f"   {a} vs {b}: raw r={r0:+.3f} (p={p0:.3f})  ->  partial r={r1:+.3f} (p={p1:.3f})")

print("\n5) Are M3-1/M3-2 near-duplicate tissue? cells per clade, ratio consistency:")
ratio = t["n_M3-1"] / t["n_M3-2"]
print(f"   n_M3-1/n_M3-2 ratio: median {ratio.median():.2f}, IQR "
      f"[{ratio.quantile(.25):.2f}, {ratio.quantile(.75):.2f}]  (tight IQR => same tissue block)")
ratio4 = t["n_M3-1"] / t["n_M3-4"]
print(f"   n_M3-1/n_M3-4 ratio: median {ratio4.median():.2f}, IQR "
      f"[{ratio4.quantile(.25):.2f}, {ratio4.quantile(.75):.2f}]")
