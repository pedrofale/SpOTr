import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = pd.read_csv(HERE / "branch_null_M3-1.csv")
obs, nul = R[R.real].iloc[0], R[~R.real]
fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.1))

for a, col, ttl, xl in [
        (ax[0], "gain_full", "Two-branch gain, full grid", "best − best uniform"),
        (ax[1], "gain_held", "Two-branch gain, held out", "best − best uniform"),
        (ax[2], "best", "Absolute accuracy of the best point", "soft accuracy")]:
    v = nul[col].values; o = float(obs[col])
    a.hist(v, bins=13, color="0.72", edgecolor="white", label="40 random splits\n(same 240/1205 sizes)")
    a.axvline(o, color="#d62728", lw=2.6, label=f"real branch split  {o:+.4f}"
              if col != "best" else f"real branch split  {o:.4f}")
    a.axvline(v.mean(), color="0.35", ls="--", lw=1.6, label=f"null mean  {v.mean():.4f}")
    z = (o - v.mean()) / max(v.std(), 1e-9)
    p = (1 + int((v >= o).sum())) / (1 + len(v))
    a.set_title(f"{ttl}\nz = {z:+.2f}" + (f",  p = {p:.2f}" if col != "best" else ""), fontsize=10)
    a.set_xlabel(xl); a.set_ylabel("random splits")
    a.legend(frameon=False, fontsize=7.6, loc="upper left")
fig.suptitle("M3-1, c = 0.30: does the TREE's branch split beat an arbitrary split "
             "with the same sizes and the same two free parameters?", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(HERE / "branch_null.png", dpi=155)
print("wrote branch_null.png")
