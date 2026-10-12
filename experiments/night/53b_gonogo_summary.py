"""Summary of 53_ under the pre-declared rule: lam chosen leave-one-section-out, wins counted per split."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "gonogo_petracer_pairwise.csv")
key = ["sec", "dens", "seed"]
PAIR = ["POTTS pairwise", "TREE pairwise (divergence time)"]

def loso(arm, metric, better):
    """per split: lam chosen on the other two sections (same density) by the mean of `metric`"""
    h = d[d.arm.isin([arm, "mixture (unary only)"])].copy()
    h = h[(h.arm == arm) | (h.lam == 0)].drop_duplicates(subset=key + ["lam"])
    out = []
    for (sec, dens), g in h.groupby(["sec", "dens"]):
        other = h[(h.sec != sec) & (h.dens == dens)].groupby("lam")[metric].mean()
        lam = other.idxmax() if better == "max" else other.idxmin()
        out.append(g[g.lam == lam].assign(arm=arm + " [lam LOSO]", lam_sel=lam))
    return pd.concat(out)

for metric, better in [("macro", "max"), ("lin_err", "min")]:
    print(f"\n==================== {metric} ({'higher' if better == 'max' else 'lower'} is better); "
          f"lam chosen leave-one-section-out on {metric}")
    base = d[d.arm.isin(["kNN vote (k LOO)", "kNN vote, tree-aware decision", "15-NN vote"])]
    mix = d[d.arm == "mixture (unary only)"].drop_duplicates(subset=key)
    sel = pd.concat([loso(a, metric, better) for a in PAIR])
    allr = pd.concat([base, mix, sel])
    tab = allr.groupby(["dens", "arm"])[metric].mean().unstack(0)
    print(tab.round(3).to_string())
    print("\n lam selected (by density, per held-out section):")
    print(sel.groupby(["arm", "dens"]).lam_sel.agg(lambda s: sorted(set(s))).to_string())
    print("\n paired wins of TREE [lam LOSO] over each comparator (out of 9 splits), sign-test p:")
    tr = sel[sel.arm.str.startswith("TREE")].set_index(key)[metric]
    for comp in ["kNN vote (k LOO)", "kNN vote, tree-aware decision", "mixture (unary only)",
                 "POTTS pairwise [lam LOSO]"]:
        c = allr[allr.arm == comp].set_index(key)[metric]
        for dens in sorted(d.dens.unique(), reverse=True):
            a = tr.xs(dens, level="dens"); b = c.xs(dens, level="dens").reindex(a.index)
            diff = (a - b) if better == "max" else (b - a)
            w = int((diff > 1e-12).sum()); l = int((diff < -1e-12).sum())
            p = binomtest(w, w + l).pvalue if w + l else 1.0
            print(f"   vs {comp:<32} dens {dens:.2f}: {w}/9 wins, {l} losses, "
                  f"mean {diff.mean():+.4f}, p = {p:.3f}")

# oracle ceiling: best lam per split, the most favourable reading
print("\n==================== oracle: best lam per split (upper bound), macro")
o = d[d.arm.isin(PAIR)].groupby(key + ["arm"]).macro.max().unstack("arm")
k = d[d.arm == "kNN vote (k LOO)"].set_index(key).macro
o["kNN vote (k LOO)"] = k
print(o.groupby("dens").mean().round(3).to_string())
