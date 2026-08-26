"""Q25: a spot-side anchor diagnostic that works when cell_pred is circular.

cell_pred needs clone labels and the anchor to come from DIFFERENT measurements. If the tree
was built from CNVs (Numbat) and the anchor is also CNV, "can CNV predict clone?" is 1.0 by
construction and the diagnostic is vacuous. spot_fit (mean best correlation to a clone
signature) is non-circular but flat at the decision boundary: poor 0.661 vs very poor 0.660,
p = 0.91.

Max correlation saturates. What should discriminate is the MARGIN -- a good anchor makes each
spot resemble one clone signature much more than the others; a noisy one makes all
correlations similar. Three margin-style statistics, all label-free and computable from the
spot anchor and the clone signatures alone:

  margin    mean (top1 - top2) correlation across spots
  softmax   mean max softmax probability over clone correlations
  entropy   mean negative entropy of that softmax (higher = sharper)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
QUAL = [("clean", 0.0, 150), ("good", 1.0, 50), ("poor", 3.0, 20), ("very poor", 6.0, 8)]
prev = pd.read_csv(HERE / "q25_prev.csv") if (HERE / "q25_prev.csv").exists() else None
base = pd.read_csv(HERE / "q24_anchor_regime.csv")
rows = []
for seed in sorted(base.seed.unique()):
    d = np.load(HERE / f"sim_ctrl_s{seed}.npz", allow_pickle=True)
    rg = np.random.default_rng(11000 + seed)          # same stream as q24
    cidx = d["cell_clade"]; K = len(d["clades"]); n = len(cidx)
    cnv_c, cnv_s = np.asarray(d["cnv_clone"], float), np.asarray(d["cnv_spot"], float)
    nb_all = cnv_s.shape[1]
    for name, nz, nb in QUAL:
        keep = rg.choice(nb_all, size=nb, replace=False)
        _ = cnv_c[cidx][:, keep] + rg.normal(0, 0.25 + 0.15 * nz, size=(n, nb))  # keep stream aligned
        spot_anchor = cnv_s[:, keep] + rg.normal(0, 0.15 * nz, size=(cnv_s.shape[0], nb))
        Sg = cnv_c[:, keep]
        Sz = (Sg - Sg.mean(1, keepdims=True)) / np.maximum(Sg.std(1, keepdims=True), 1e-9)
        Yz = (spot_anchor - spot_anchor.mean(1, keepdims=True)) / \
            np.maximum(spot_anchor.std(1, keepdims=True), 1e-9)
        Cmat = Yz @ Sz.T / nb                            # spots x clones
        srt = np.sort(Cmat, axis=1)
        margin = float((srt[:, -1] - srt[:, -2]).mean())
        # softmax sharpness, temperature-free by standardising each spot's correlations
        Z = (Cmat - Cmat.mean(1, keepdims=True)) / np.maximum(Cmat.std(1, keepdims=True), 1e-9)
        P = np.exp(Z); P = P / P.sum(1, keepdims=True)
        rows.append(dict(seed=seed, quality=name, margin=margin,
                         softmax=float(P.max(1).mean()),
                         neg_entropy=float(-(-(P * np.log(np.maximum(P, 1e-30))).sum(1)).mean())))
r = pd.DataFrame(rows).merge(base[["seed", "quality", "feat", "spotr", "cell_pred", "spot_fit"]],
                             on=["seed", "quality"])
r["feat_better"] = r.feat > r.spotr
print(f"{'quality':>11}{'margin':>9}{'softmax':>9}{'neg_ent':>9}{'spot_fit':>10}{'cell_pred':>11}{'feat better':>13}")
print("-" * 72)
for q, s in r.groupby("quality", sort=False):
    print(f"{q:>11}{s.margin.mean():>9.3f}{s.softmax.mean():>9.3f}{s.neg_entropy.mean():>9.3f}"
          f"{s.spot_fit.mean():>10.3f}{s.cell_pred.mean():>11.3f}"
          f"{str(int(s.feat_better.sum())) + '/' + str(len(s)):>13}")
print("-" * 72)
print(f"\n{'diagnostic':>12}{'AUC (picks right strategy)':>28}{'best acc':>10}{'poor vs very poor p':>21}")
for nm in ["margin", "softmax", "neg_entropy", "spot_fit", "cell_pred"]:
    a = roc_auc_score(r.feat_better.astype(int), r[nm])
    a = max(a, 1 - a)
    best = max(((r[nm] > t) == r.feat_better).mean()
               for t in np.linspace(r[nm].min(), r[nm].max(), 80))
    p = stats.mannwhitneyu(r[r.quality == "poor"][nm], r[r.quality == "very poor"][nm])[1]
    flag = "  <-- usable" if a > 0.9 and p < 0.01 else ""
    print(f"{nm:>12}{a:>28.3f}{best*100:>9.0f}%{p:>21.4f}{flag}")
r.to_csv(HERE / "q25_spot_diag.csv", index=False)
