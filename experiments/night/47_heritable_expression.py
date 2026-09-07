"""IS THERE HERITABLE GENE EXPRESSION THAT COULD SERVE AS AN ENDOGENOUS LINEAGE SIGNAL?

Pedro's proposal: find genes whose expression tracks the phylogeny but is unrelated to phenotype,
and use them for lineage tracing directly -- no recorder needed.

The measurement.  For every gene g and every pair of cells (i,k):
    d_g(i,k) = |x_gi - x_gk|          expression dissimilarity
    t(i,k)   = patristic distance      how long ago they diverged
    s(i,k)   = spatial distance
Phylogenetic signal for gene g is corr(d_g, t).  A positive value means relatives are more alike.

THE CONFOUND THAT DECIDES IT.  Clones occupy territory, so relatives are also NEIGHBOURS, and any
gene with spatial structure -- a niche gene, an oxygen gradient, a field-of-view batch effect --
will show phylogenetic signal for reasons that have nothing to do with heritability.  So the
number that matters is the PARTIAL correlation of (d_g, t) controlling for s.  Null: permute which
cell carries which row of the tree, which destroys heritability while preserving everything else.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
NPAIR = 300_000


def partial(a, b, c):
    """corr(a,b | c), all standardised columns/vectors"""
    def z(v):
        v = v - v.mean(0)
        return v / np.maximum(v.std(0), 1e-12)
    a, b, c = z(a), z(b.reshape(-1, 1)), z(c.reshape(-1, 1))
    ra = a - c * (c.T @ a) / len(c)
    rb = b - c * (c.T @ b) / len(c)
    ra /= np.maximum(ra.std(0), 1e-12); rb /= np.maximum(rb.std(0), 1e-12)
    return (ra * rb).mean(0)


out = {}
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    X = np.asarray(d["X_sc"], np.float32)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    genes = np.asarray(d["genes"]).astype(str)
    n = len(X)
    i = rng.integers(0, n, NPAIR); k = rng.integers(0, n, NPAIR)
    m = i != k; i, k = i[m], k[m]
    t = CT[i, k]
    s = np.linalg.norm(xy[i] - xy[k], axis=1)
    D = np.abs(X[i] - X[k])                       # npair x ngene

    raw = partial(D, t, np.zeros_like(t))         # plain correlation
    ctl = partial(D, t, s)                        # controlling for spatial distance
    # null: shuffle which cell carries which tree row
    perm = rng.permutation(n)
    tn = CT[perm[i], perm[k]]
    raw_n = partial(D, tn, np.zeros_like(tn))
    ctl_n = partial(D, tn, s)

    R = pd.DataFrame(dict(gene=genes, raw=raw, controlled=ctl,
                          null_raw=raw_n, null_controlled=ctl_n))
    out[SEC] = R
    thr = np.quantile(np.abs(R.null_controlled), 0.99)
    print(f"\n=== {SEC} ===  {n} cells, {len(genes)} genes, {len(i):,} pairs")
    print(f"  phylogenetic signal, RAW           : mean {R.raw.mean():+.4f}   "
          f"max {R.raw.max():+.4f}   null mean {R.null_raw.mean():+.4f}")
    print(f"  CONTROLLING for spatial distance   : mean {R.controlled.mean():+.4f}   "
          f"max {R.controlled.max():+.4f}   null mean {R.null_controlled.mean():+.4f}")
    print(f"  genes above the 99th pct of |null| ({thr:.4f}): "
          f"{int((R.controlled > thr).sum())} / {len(genes)}")
    print("  top 8 by controlled signal:")
    for _, r in R.sort_values("controlled", ascending=False).head(8).iterrows():
        print(f"     {r.gene:<12} raw {r.raw:+.4f}   controlled {r.controlled:+.4f}")

A = pd.concat([v.assign(section=k) for k, v in out.items()])
A.to_csv(HERE / "heritable_expression.csv", index=False)
P = A.pivot_table(index="gene", columns="section", values="controlled")
print("\n\nDOES THE PER-GENE SIGNAL REPRODUCE ACROSS SECTIONS?")
from scipy.stats import pearsonr, spearmanr
for a, b in [("M3-4", "M3-1"), ("M3-4", "M3-2"), ("M3-1", "M3-2")]:
    print(f"   {a} vs {b}: pearson {pearsonr(P[a], P[b]).statistic:+.3f}   "
          f"spearman {spearmanr(P[a], P[b]).statistic:+.3f}")
print("\ngenes in the top 20 of ALL THREE sections:")
tops = [set(out[s].sort_values("controlled", ascending=False).head(20).gene) for s in out]
print("  ", sorted(tops[0] & tops[1] & tops[2]))
