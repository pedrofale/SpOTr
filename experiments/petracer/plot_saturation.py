import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
INK, GRID = "#0b0b0b", "#d8d8d4"
fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.0), sharey=True)
for k, SEC in enumerate(["M3-4", "M3-1", "M3-2"]):
    CT = np.asarray(np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)["C_tree"], float)
    v = CT[np.triu_indices(len(CT), 1)]
    cap = v.max(); frac = float(np.mean(v >= cap - 1e-9))
    a = ax[k]
    a.hist(v[v < cap - 1e-9], bins=70, color="#2a78d6", label="resolved pairs")
    a.bar([cap], [np.sum(v >= cap - 1e-9)], width=1.4, color="#a83c10",
          label=f"at the ceiling ({frac:.0%})")
    a.set_yscale("log")
    a.set_xlabel("patristic (tree) distance between two cells")
    a.set_title(f"{SEC}\n{frac:.0%} of all pairs sit at exactly {cap:.0f}", fontsize=11)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a.grid(alpha=.25); a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("cell pairs (log scale)")
fig.suptitle("Saturation: the recorder runs out of resolution, and every deeper relationship "
             "collapses to one number.\nFor those pairs the tree says only “not close” — it "
             "cannot say how far.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.84])
fig.savefig(HERE / "saturation.png", dpi=150, facecolor="white")
for SEC in ["M3-4", "M3-1", "M3-2"]:
    CT = np.asarray(np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)["C_tree"], float)
    v = CT[np.triu_indices(len(CT), 1)]
    cap = v.max(); r = v[v < cap - 1e-9]
    print(f"{SEC}: ceiling {cap:.1f}   at ceiling {np.mean(v>=cap-1e-9):.1%}   "
          f"resolved pairs span {r.min():.1f}–{r.max():.1f}")
