"""Summary of an anchor_fast run: placement error against FGW at its leave-one-section-out weight, paired by
split; and, for the per-clade arm, learned L_k against L_k measured from TRUE positions (evaluation only),
with a permutation null over clades.  Usage: python 61b_summary.py anchor_fast2.csv"""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / (sys.argv[1] if len(sys.argv) > 1 else "anchor_fast2.csv"))
h = pd.read_csv(HERE / "decomposition_check.csv")
fgw = h[(h.arm == "FGW") & (h.half == 0) & np.isclose(h.w, 0.1968, atol=1e-3)].set_index(["sec", "seed"]).err
print(f"FGW at its leave-one-section-out weight, these splits: {fgw[fgw.index.get_level_values(1).isin(d.seed.unique())].mean():.1f} um")
for cov in sorted(d["cov"].unique(), reverse=True):
    g = d[d["cov"] == cov].set_index(["sec", "seed", "arm"])
    print(f"\ncoverage {cov:.0%}")
    for col in [c for c in ("err_exact_heldout", "err_entropic_heldout", "err_exact", "err_entropic") if c in g.columns]:
        t = g[col].unstack("arm"); base = fgw.reindex(t.index)
        print(f"  {col}:  " + "   ".join(f"{a} {t[a].mean():.0f} ({int(((base - t[a]) > 0).sum())}/{len(t)} beat FGW)" for a in t.columns))

spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)
rows = []
for _, r in d[d.arm.isin(["pc-law", "pc-free", "pc-sup"])].iterrows():
    z = np.load(HERE / "anchor_fast" / f"{r.tag}.npz"); P = m60.prep(r.sec, int(r.seed))
    Lk, c = z["Lk"], float(z["c"]); lm, ll = [], []
    for k in range(P["K"]):
        idx = np.where(P["cl"] == k)[0]
        if len(idx) < 30:
            continue
        iu = np.triu_indices(len(idx), 1)
        d2 = ((P["xy_true"][idx][:, None] - P["xy_true"][idx][None]) ** 2).sum(-1)[iu]
        pk = P["NC"][np.ix_(idx, idx)][iu] / (2 * np.pi * P["rho"])
        lm.append(np.sqrt(max(np.mean(d2 / 2 - c * c * pk), 0.0))); ll.append(Lk[k])
    rho = spearmanr(ll, lm)[0]
    null = [spearmanr(np.random.default_rng(i).permutation(ll), lm)[0] for i in range(1000)]
    p_perm = float(np.mean(np.array(null) >= rho))
    rows.append(dict(arm=r.arm, sec=r.sec, seed=r.seed, cov=r["cov"], n_clades=len(ll), rho=rho, p_perm=p_perm,
                     spread_learned=float(np.std(np.log(ll))),
                     spread_measured=float(np.std(np.log(np.maximum(lm, 1.0))))))
if rows:
    t = pd.DataFrame(rows)
    print("\nper-clade: learned L_k vs L_k measured from true positions")
    print(t.round(3).to_string(index=False))
    for a, g in t.groupby("arm"):
        ok = g.rho.notna()
        print(f"{a}: mean rho {g.rho[ok].mean():+.3f}; p_perm < 0.05 in {((g.p_perm < 0.05) & ok).sum()}/{ok.sum()} splits; "
              f"learned log-spread {g.spread_learned.mean():.3f} vs measured {g.spread_measured.mean():.3f}")
