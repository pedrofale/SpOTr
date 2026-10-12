"""Summary of 55_: tree weight chosen leave-one-section-out; paired over the 18 splits."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "decomposition_check.csv")
key = ["sec", "seed", "half"]

def loso(df, arm):
    g = df[df.arm == arm]; out = []
    for sec in g.sec.unique():
        wsel = g[g.sec != sec].groupby("w").err.mean().idxmin()
        out.append(g[(g.sec == sec) & (g.w == wsel)].assign(w_sel=wsel))
    return pd.concat(out).set_index(key)

arms = ["FGW", "decomposed, ground truth", "decomposed, law", "law, continuous"]
expr = d[d.arm == "expression only"].set_index(key).err
sel = {a: loso(d, a) for a in arms}

# reference: generalised FGW (learned rank 2) and FGW from the 30-split head-to-head, same 18 splits
h = pd.read_csv(HERE / "headtohead.csv").rename(columns={"section": "sec"})
h = h[h.seed.isin([0, 1, 2])]
for a, nm in [("learned rank 2", "generalised FGW, rank 2 [36_]"), ("FGW", "FGW via POT/genfgw [36_]")]:
    g = h[(h.arm == a) & (h.w > 0)].rename(columns={"arm": "arm_"}).assign(arm=a)
    sel[nm] = loso(g, a)

print("median placement error (um), mean over splits; tree weight chosen on the OTHER two sections\n")
tab = pd.DataFrame({"expression only": expr.groupby("sec").mean()})
for a, s in sel.items():
    tab[a] = s.err.groupby("sec").mean()
tab.loc["mean"] = tab.mean()
print(tab.T.round(1).to_string())
print("\nselected w per held-out section:")
for a in arms:
    print(f"   {a:<28} " + "  ".join(f"{s}:{w:.3g}" for s, w in sel[a].groupby('sec').w_sel.first().items()))

print("\npaired comparisons over 18 splits (positive = row arm better, um):")
def pair(a, b, A, B):
    x = (B - A.reindex(B.index)).dropna()
    w, l = int((x > 0).sum()), int((x < 0).sum())
    print(f"   {a:<28} vs {b:<30} {w:2d}/{len(x)} wins  mean {x.mean():+6.1f}  "
          f"sign p = {binomtest(w, w + l).pvalue if w + l else 1:.3g}")
for a in ["decomposed, ground truth", "decomposed, law", "law, continuous"]:
    pair(a, "FGW (same solver)", sel[a].err, sel["FGW"].err)
pair("decomposed, law", "decomposed, ground truth", sel["decomposed, law"].err, sel["decomposed, ground truth"].err)
pair("decomposed, ground truth", "generalised FGW [36_]", sel["decomposed, ground truth"].err, sel["generalised FGW, rank 2 [36_]"].err)
pair("decomposed, law", "generalised FGW [36_]", sel["decomposed, law"].err, sel["generalised FGW, rank 2 [36_]"].err)
pair("FGW (same solver)", "FGW via POT/genfgw [36_]", sel["FGW"].err, sel["FGW via POT/genfgw [36_]"].err)

print("\nuntuned: median over the weight grid (w > 0), mean over splits")
print(d[d.w > 0].groupby(["arm", "sec"] + ["seed", "half"]).err.median().groupby(["arm", "sec"]).mean().unstack().round(1).to_string())

m = pd.read_csv(HERE / "decomposition_check_classes.csv")
print("\nclass statistics, mean over splits (um): ground truth from reference pairs vs growth law")
for b, nm in enumerate(["recent <20 d", "20-30 d", "30-35 d", "founding"]):
    print(f"   {nm:<13} pairs {100*m[f'frac_{b}'].mean():5.1f}%   mean sep: truth {m[f'gt_mu_{b}'].mean():6.0f}"
          f"  law {m[f'law_mu_{b}'].mean():6.0f}   sd: truth {m[f'gt_sd_{b}'].mean():5.0f}  law {m[f'law_sd_{b}'].mean():5.0f}")
