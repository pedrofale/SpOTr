"""MOTILITY FROM A SINGLE SLICE.

Given r(a), a clade's cells should sit at r(a) for their own divergence ages.  The ratio

    m_tau  =  RMS observed separation within clade tau  /  RMS separation r(a) predicts

is a dimensionless motility index: 1.0 = exactly as dispersed as its lineage says, > 1 = the cells
have moved further than their divergence times allow.  It needs one slice and the tree.

Three questions, in order of how much they matter:
  1. does it vary between clades more than a size-matched random group of cells would?
  2. does it reproduce between two halves of the same section?          (is it estimable at all)
  3. does it reproduce between SECTIONS?                                (is it a clade property)
Question 3 is the one that decides whether this is biology or an artefact of how a section cuts a
clade -- the same test that killed per-clade alpha.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.clonespread import fit_spread, clade_dispersion

SECS = ["M3-4", "M3-1", "M3-2"]
rng = np.random.default_rng(0)
D, disp, nulls = {}, {}, {}
for SEC in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    iu = np.triu_indices(len(xy), 1)
    sp = fit_spread(CT[iu], pdist(xy))
    D[SEC] = (CT, xy, cc, sp)
    disp[SEC] = clade_dispersion(CT, xy, cc, sp)
    # size-matched random groupings: how much would this vary by chance?
    sizes = {k: int((cc == k).sum()) for k in disp[SEC]}
    nl = []
    for k, nsz in sizes.items():
        vals = []
        for _ in range(40):
            j = rng.choice(len(cc), nsz, replace=False)
            fake = np.array(["x"] * len(cc)); fake[j] = "g"
            v = clade_dispersion(CT, xy, fake, sp).get("g")
            if v is not None:
                vals.append(v)
        nl.append(np.std(vals))
    nulls[SEC] = float(np.mean(nl))

print("1.  DOES IT VARY BETWEEN CLADES?  (dimensionless motility index per clade)")
T = pd.DataFrame(disp)
print(T.round(2).to_string())
for SEC in SECS:
    v = np.array(list(disp[SEC].values()))
    print(f"   {SEC}: spread {v.min():.2f}-{v.max():.2f}   sd {v.std():.3f}   "
          f"size-matched random sd {nulls[SEC]:.3f}   ratio {v.std()/nulls[SEC]:.1f}x")

print("\n2.  DOES IT REPRODUCE BETWEEN HALVES OF ONE SECTION?")
pairs = []
for SEC in SECS:
    CT, xy, cc, sp = D[SEC]
    for rep in range(3):
        h = np.random.default_rng(100 + rep).permutation(len(cc))
        A, B = h[:len(cc)//2], h[len(cc)//2:]
        da = clade_dispersion(CT[np.ix_(A,A)], xy[A], cc[A], sp)
        db = clade_dispersion(CT[np.ix_(B,B)], xy[B], cc[B], sp)
        for k in set(da) & set(db):
            pairs.append((da[k], db[k]))
pa = np.array(pairs)
print(f"   n={len(pa)} clade-halves   pearson {pearsonr(pa[:,0],pa[:,1]).statistic:+.3f}"
      f"   spearman {spearmanr(pa[:,0],pa[:,1]).statistic:+.3f}")

print("\n3.  DOES IT REPRODUCE BETWEEN SECTIONS?  (the test that decides it)")
P = T.dropna()
print(f"   {len(P)} clades present in all three sections")
for i in range(3):
    for j in range(i+1, 3):
        a, b = SECS[i], SECS[j]
        print(f"   {a} vs {b}:  pearson {pearsonr(P[a],P[b]).statistic:+.3f}"
              f"   spearman {spearmanr(P[a],P[b]).statistic:+.3f}")
T.to_csv(HERE / "motility.csv")
