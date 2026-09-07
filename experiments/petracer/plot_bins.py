"""What 'bins' means: groups of clade-pairs that share one weight in the tree term."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
CT = np.asarray(d["C_tree"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
cl, clade = np.unique(cc, return_inverse=True)
K = len(cl)
Dkk = np.array([[CT[np.ix_(clade == a, clade == b)].mean() for b in range(K)] for a in range(K)])

def bin_map(Dkk, nbin):
    K = len(Dkk); lab = np.zeros((K, K), int); off = ~np.eye(K, dtype=bool)
    v = Dkk[off]
    if nbin > 1:
        edges = np.quantile(v, np.linspace(0, 1, nbin)[1:-1]) if nbin > 2 else []
        lab[off] = np.digitize(v, edges) + 1
    return lab

FITTED = {1: [3.0], 2: [1.7, 1.0], 3: [1.7, 1.0, 1.0],
          5: [1.7, 1.0, 0.3, 1.0, 1.0], 8: [1.7, 1.0, 0.0, .15, .15, 1.0, 1.7, 1.0]}
fig, ax = plt.subplots(1, 5, figsize=(17.5, 4.3))
a = ax[0]
im = a.imshow(Dkk, cmap="Blues")
a.set_title("the raw material:\nmean tree distance between clades", fontsize=10.5)
fig.colorbar(im, ax=a, fraction=.046).ax.tick_params(labelsize=7)
for k, nb in enumerate([1, 2, 3, 8]):
    a = ax[k + 1]
    lab = bin_map(Dkk, nb)
    cmap = ListedColormap(plt.get_cmap("tab10")(np.arange(max(nb, 1))))
    a.imshow(lab, cmap=cmap, vmin=0, vmax=max(nb - 1, 1))
    w = FITTED[nb]
    a.set_title(f"{nb} bin{'s' if nb>1 else ''}  ({nb} weight{'s' if nb>1 else ''})\n"
                + "  ".join(f"w{j}={w[j]:g}" for j in range(min(nb, 4)))
                + ("  ..." if nb > 4 else ""), fontsize=10)
    if nb == 1:
        a.set_xlabel("every pair weighted the same\n= ordinary FGW", fontsize=9)
    elif nb == 2:
        a.set_xlabel("diagonal = same clade\noff-diagonal = different clades", fontsize=9)
    else:
        a.set_xlabel("cross-clade pairs split further\nby tree distance", fontsize=9)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
    a.set_ylabel("clade" if a is ax[0] else "")
fig.suptitle("Each cell of the 18x18 matrix is one clade-pair. A 'bin' is a group of clade-pairs "
             "that share a weight.\nMore bins gave EXACTLY the same mapping as 2 bins: "
             "349.7 um in-sample, 336.9 held out.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(HERE / "what_are_bins.png", dpi=150, facecolor="white")
off = ~np.eye(K, dtype=bool)
print(f"clade-pair tree distances: diagonal (same clade) {np.diag(Dkk).mean():.1f}, "
      f"off-diagonal {Dkk[off].mean():.1f}  (range {Dkk[off].min():.1f}-{Dkk[off].max():.1f})")
print(f"so the off-diagonal spans only {Dkk[off].max()-Dkk[off].min():.1f} of 70 -- "
      f"there is very little for extra bins to separate")
