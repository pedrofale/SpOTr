import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = pd.read_csv(HERE / "branching_sweep.csv"); S["sat"] = S.sat.fillna("None").astype(str)
P = pd.read_csv(HERE.parent / "petracer" / "exact_fgw_mean-norm_M3-4.csv")

fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.6))
CS, CE = "#2a78d6", "#eb6834"

g = S[(S.sat == "0.5") & (S.sigma_expr == 10.0)].groupby("alpha").agg(
    soft=("soft", "mean"), err=("err", "mean"))
for a, (x, soft, err, ttl, sub) in enumerate([
        (g.index.values, g.soft.values, g.err.values,
         "Branching simulator (calibrated to PEtracer)", "expression noise 10"),
        (P.alpha.values, P.soft.values, P.err.values,
         "PEtracer M3-4, exact solver", "the real data")]):
    A = ax[a]
    s = soft / soft.max()
    A.plot(x, s, "o-", color=CS, lw=2.4, ms=6, label="soft (clade composition)")
    B = A.twinx()
    B.plot(x, err, "s--", color=CE, lw=2.2, ms=6, label="median placement error")
    B.set_ylabel("median placement error", color=CE)
    B.tick_params(axis="y", colors=CE)
    bs, be = x[np.argmax(soft)], x[np.argmin(err)]
    A.axvline(bs, color=CS, ls=":", lw=2)
    A.axvline(be, color=CE, ls=":", lw=2)
    A.annotate(f"soft picks\nα={bs:g}", (bs, 0.35), color=CS, fontsize=9,
               ha="center", fontweight="bold")
    A.annotate(f"placement picks\nα={be:g}", (be, 0.15), color="#a83c10", fontsize=9,
               ha="center", fontweight="bold")
    A.set_xlabel("α"); A.set_ylabel("soft accuracy (scaled)", color=CS)
    A.tick_params(axis="y", colors=CS)
    A.set_title(f"{ttl}\n{sub}", fontsize=10.5)
    A.grid(alpha=.22); A.set_axisbelow(True)
    if a == 0:
        A.legend(frameon=False, fontsize=8.5, loc="lower center")
        B.legend(frameon=False, fontsize=8.5, loc="center right")
fig.suptitle("Soft clade-composition accuracy is biased toward high α — it rewards landing among "
             "clade-mates,\nand the tree term encodes clade identity directly. "
             "The simulator shows the same bias where the truth is known.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.savefig(HERE / "metric_bias.png", dpi=155)
print("wrote metric_bias.png")
print("\nPEtracer M3-4:")
for _, r in P.iterrows():
    print(f"  alpha {r.alpha:.2f}   soft {r.soft:.4f}   err {r.err:6.1f}")
