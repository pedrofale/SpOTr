"""Is the per-clade oracle headroom REAL, or does coordinate ascent overfit the spots?

run_transfer_matrix.py fits alpha_k by coordinate ascent on TRUE accuracy over all spots of
a section, then scores the same spots. With 18 free parameters and 204-429 spots that is
overfit-capable, so the diagonal is an upper bound, not an achievable number.

Here alpha is fitted on a random HALF of the spots and scored on the held-out half.
Everything (best global, uniform-at-mean, shuffled) is selected on the fit half and scored
on the held-out half, so the comparison is clean.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_transfer_matrix import Sec, GRID_G, GRID_K, SECS   # reuses solver + scorer

rng = np.random.default_rng(1)
NSPLIT = 3
rows = []
for name in SECS:
    S_ = Sec(name)
    for rep in range(NSPLIT):
        h = rng.random(S_.m) < 0.5
        fit, hel = np.where(h)[0], np.where(~h)[0]
        if len(fit) < 20 or len(hel) < 20: continue
        # global alpha selected on the fit half
        gs = {v: S_.solve(np.full(S_.K, v)) for v in GRID_G}
        bg = max(gs, key=lambda v: S_.score(gs[v], fit))
        bg_held = S_.score(gs[bg], hel)
        a0_held = S_.score(gs[0.0], hel)
        # oracle per-clade, fitted on the fit half ONLY
        ak = np.full(S_.K, float(bg), np.float32)
        cur = S_.acc(ak, fit)
        for _ in range(2):
            for k in range(S_.K):
                best, bv = cur, ak[k]
                for v in GRID_K:
                    if v == ak[k]: continue
                    t = ak.copy(); t[k] = v
                    a_ = S_.acc(t, fit)
                    if a_ > best: best, bv = a_, v
                ak[k], cur = bv, best
        g_ak = S_.solve(ak)
        pc_held = S_.score(g_ak, hel)
        un_held = S_.score(S_.solve(np.full(S_.K, float(ak.mean()))), hel)
        sh = [S_.score(S_.solve(rng.permutation(ak)), hel) for _ in range(8)]
        rows.append(dict(section=name, rep=rep, n_fit=len(fit), n_held=len(hel),
                         best_global=bg, alpha0_held=a0_held, global_held=bg_held,
                         perclade_held=pc_held, uniform_held=un_held,
                         shuffled_held=float(np.nanmean(sh)), shuffled_sd=float(np.nanstd(sh)),
                         perclade_fit=cur, alpha_mean=float(ak.mean()), alpha_sd=float(ak.std())))
        print(f"  {name} rep{rep}: fit {len(fit)}/held {len(hel)} | a=0 {a0_held:.4f} | "
              f"global(a={bg}) {bg_held:.4f} | per-clade {pc_held:.4f} | uniform@mean {un_held:.4f} | "
              f"shuffled {np.nanmean(sh):.4f}+-{np.nanstd(sh):.4f} || "
              f"pc-global {pc_held-bg_held:+.4f}", flush=True)

r = pd.DataFrame(rows)
r.to_csv(HERE / "petracer_split_half_oracle.csv", index=False)
print("\n" + "=" * 78)
print("HELD-OUT per-clade gain (alpha fitted on the other half of the spots)")
print(f"{'section':<10}{'pc - global':>14}{'pc - uniform':>15}{'pc - shuffled':>15}{'z vs shuffled':>15}")
print("-" * 69)
for s, d in r.groupby("section"):
    z = ((d.perclade_held - d.shuffled_held) / d.shuffled_sd.clip(lower=1e-9)).mean()
    print(f"{s:<10}{(d.perclade_held-d.global_held).mean():>+14.4f}"
          f"{(d.perclade_held-d.uniform_held).mean():>+15.4f}"
          f"{(d.perclade_held-d.shuffled_held).mean():>+15.4f}{z:>+15.1f}")
print("-" * 69)
print(f"{'ALL':<10}{(r.perclade_held-r.global_held).mean():>+14.4f}"
      f"{(r.perclade_held-r.uniform_held).mean():>+15.4f}"
      f"{(r.perclade_held-r.shuffled_held).mean():>+15.4f}")
print(f"\noverfitting gap (fit-half acc minus held-out acc): "
      f"{(r.perclade_fit-r.perclade_held).mean():+.4f}")
print("wrote petracer_split_half_oracle.csv")
