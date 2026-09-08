"""STOP CONDITION 1: does the head-to-head survive a statistic that is not each arm's best w?

36_headtohead.py swept a common w grid and NOTES.md quoted, per split, each arm's OWN MINIMUM over
that grid.  That minimum is chosen using the test error, so it is an oracle: it asks "how good can
this cost be made to look" rather than "how good is this cost".  Three statistics are reported here
side by side on the SAME 30 splits:

  best-w             per split, min over w > 0            ORACLE      (what NOTES.md quoted)
  median-over-grid   per split, median over w > 0         no selection, robustness to w
  matched-w          both arms at the SAME single w        no selection, one shared operating point

and, because "matched at which w?" is itself a choice, two honest selection rules:

  fixed-w (pooled)   the single w minimising mean error over ALL splits, per arm  (still in-sample)
  LOSO-w             w chosen on the other two sections, evaluated on the held-out one  (clean)

The CSV lives on branch spotr-evaluation; this checkout may be on main, so fall back to git show.
"""
import io, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
CSV = HERE / "headtohead.csv"

if CSV.exists():
    d = pd.read_csv(CSV)
else:
    blob = subprocess.run(["git", "-C", str(HERE.parent.parent), "show",
                           "spotr-evaluation:experiments/night/headtohead.csv"],
                          capture_output=True, text=True, check=True).stdout
    d = pd.read_csv(io.StringIO(blob))
    print(f"[headtohead.csv not in this checkout -- read from branch spotr-evaluation]\n")

SPLIT = ["section", "seed", "half"]
ARMS = ["FGW", "clone spread", "learned rank 2"]
pos = d[d.w > 0].copy()                     # w == 0 is the expression-only arm, identical for all
grid = np.sort(pos.w.unique())
n_split = d.groupby(SPLIT).ngroups
print(f"{len(d)} rows | {n_split} splits | {len(grid)} positive w values "
      f"[{grid.min():.3g} .. {grid.max():.3g}] | arms {sorted(d.arm.unique())}")

# w == 0 is solved with ot.emd for every arm, so it must be identical across arms -- check.
z = d[d.w == 0].pivot_table(index=SPLIT, columns="arm", values="err")
print(f"expression-only (w=0) agreement across arms: max spread {np.nanmax(z.max(1) - z.min(1)):.2e} um")
print(f"expression-only mean error: {z.mean(1).mean():.1f} um\n")


def paired(tab, a, b, label):
    """tab: index = split, columns = arm, values = error in um.  Positive delta = a beats b."""
    x, y = tab[a].values, tab[b].values
    dl = y - x
    wins = int((dl > 0).sum()); n = int(np.isfinite(dl).sum())
    p = binomtest(wins, n, 0.5, alternative="two-sided").pvalue
    pw = wilcoxon(x, y).pvalue
    print(f"  {label:<34} {wins:2d}/{n}  {dl.mean():+6.1f} um  "
          f"(median {np.median(dl):+6.1f})  sign p = {p:.3g}  wilcoxon p = {pw:.3g}")
    return dict(stat=label, a=a, b=b, wins=wins, n=n, delta=dl.mean(),
                median_delta=float(np.median(dl)), sign_p=p, wilcoxon_p=pw)


rows_summary, rows_paired = [], []

# ---------------------------------------------------------------- 1. best-w (oracle) and median
stats = {"best-w (ORACLE)": pos.groupby(SPLIT + ["arm"]).err.min().unstack("arm"),
         "median-over-grid": pos.groupby(SPLIT + ["arm"]).err.median().unstack("arm")}

# ---------------------------------------------------------------- 2. matched-w, pooled selection
mean_by_w = pos.groupby(["arm", "w"]).err.mean().unstack("arm")
w_star = {a: float(mean_by_w[a].idxmin()) for a in ARMS}
w_common = float(mean_by_w[ARMS].mean(1).idxmin())
print("single w minimising mean error over all 30 splits:")
for a in ARMS:
    print(f"  {a:<16} w* = {w_star[a]:<8.4g}  mean {mean_by_w[a].min():6.1f} um")
print(f"  {'COMMON w':<16} w* = {w_common:<8.4g}  "
      f"(minimises the across-arm mean)\n")

at = pos.set_index(SPLIT + ["arm", "w"]).err
stats["fixed-w (per-arm w*)"] = pd.DataFrame(
    {a: at.xs((a, w_star[a]), level=("arm", "w")) for a in ARMS})
stats[f"matched-w (common w={w_common:.4g})"] = pd.DataFrame(
    {a: at.xs((a, w_common), level=("arm", "w")) for a in ARMS})

# ---------------------------------------------------------------- 3. LOSO-w (clean selection)
loso = {}
for a in ARMS:
    sub = pos[pos.arm == a]
    out = []
    for sec in sub.section.unique():
        w_sel = float(sub[sub.section != sec].groupby("w").err.mean().idxmin())
        held = sub[(sub.section == sec) & (sub.w == w_sel)].set_index(SPLIT).err
        out.append(held)
        print(f"  LOSO {a:<16} hold out {sec}: w selected on the other two = {w_sel:.4g}")
    loso[a] = pd.concat(out)
stats["LOSO-w (clean)"] = pd.DataFrame(loso)
print()

# ---------------------------------------------------------------- the side-by-side table
order = list(stats)
print("=" * 92)
print("MEAN ERROR (um), lower is better -- same 30 splits under every statistic")
print("=" * 92)
print(f"{'statistic':<32} " + "".join(f"{a:>17}" for a in ARMS))
for s in order:
    t = stats[s]
    print(f"{s:<32} " + "".join(f"{t[a].mean():17.1f}" for a in ARMS))
    for a in ARMS:
        rows_summary.append(dict(statistic=s, arm=a, mean_err=t[a].mean(),
                                 sd=t[a].std(), n=len(t)))
print()

print("PER SECTION (mean um)")
for s in order:
    g = stats[s].groupby(level="section")[ARMS].mean()
    print(f"\n  {s}")
    print("    " + f"{'section':<10}" + "".join(f"{a:>17}" for a in ARMS))
    for sec, r in g.iterrows():
        print("    " + f"{sec:<10}" + "".join(f"{r[a]:17.1f}" for a in ARMS))
        for a in ARMS:
            rows_summary.append(dict(statistic=s, arm=a, section=sec, mean_err=r[a], sd=np.nan, n=0))
print("\n")

print("=" * 92)
print("PAIRED COMPARISONS across the 30 splits")
print("=" * 92)
for s in order:
    print(f"\n{s}")
    for a, b in [("clone spread", "FGW"), ("learned rank 2", "FGW"),
                 ("clone spread", "learned rank 2")]:
        r = paired(stats[s], a, b, f"{a} beats {b}")
        r["statistic"] = s
        rows_paired.append(r)

# ---------------------------------------------------------------- 4. matched-w at EVERY w
print("\n" + "=" * 92)
print("MATCHED-w AT EVERY GRID POINT: clone spread vs FGW, both arms at the same w")
print("=" * 92)
print(f"{'w':>10} {'alpha':>7} {'FGW':>9} {'cspread':>9} {'delta':>8} {'wins/30':>9} {'sign p':>9}")
rows_perw = []
for w in grid:
    t = pd.DataFrame({a: at.xs((a, w), level=("arm", "w")) for a in ARMS})
    dl = t["FGW"] - t["clone spread"]
    wins = int((dl > 0).sum())
    p = binomtest(wins, len(dl), 0.5).pvalue
    flag = " *" if p < 0.05 else ""
    print(f"{w:10.4g} {w/(1+w):7.3f} {t['FGW'].mean():9.1f} {t['clone spread'].mean():9.1f} "
          f"{dl.mean():+8.1f} {wins:6d}/30 {p:9.3g}{flag}")
    rows_perw.append(dict(w=w, alpha=w/(1+w), fgw=t["FGW"].mean(),
                          clonespread=t["clone spread"].mean(),
                          learned=t["learned rank 2"].mean(),
                          delta=dl.mean(), wins=wins, sign_p=p))
pw = pd.DataFrame(rows_perw)
print(f"\n  clone spread has the lower mean at {int((pw.delta > 0).sum())}/{len(pw)} grid points; "
      f"majority of splits at {int((pw.wins > 15).sum())}/{len(pw)}; "
      f"sign p < 0.05 at {int((pw.sign_p < 0.05).sum())}/{len(pw)}.")

# ---------------------------------------------------------------- 5. how peaked is each arm in w?
print("\n" + "=" * 92)
print("SENSITIVITY TO w -- why the oracle statistic flatters one arm more than the other")
print("=" * 92)
for a in ARMS:
    per = pos[pos.arm == a].groupby(SPLIT).err
    gap = (per.median() - per.min())
    bw = pos[pos.arm == a].loc[pos[pos.arm == a].groupby(SPLIT).err.idxmin(), SPLIT + ["w"]]
    print(f"  {a:<16} median-minus-best {gap.mean():6.1f} um   "
          f"worst-minus-best {(per.max() - per.min()).mean():7.1f} um   "
          f"argmin w: median {bw.w.median():.4g}, "
          f"{bw.w.nunique()} distinct values over 30 splits")
print("\n  A LARGE median-minus-best means the arm's reported number depends heavily on getting w"
      "\n  right, and the oracle statistic hands it that for free.")

# ---------------------------------------------------------------- 6. cost per solve
print("\n" + "=" * 92)
print("SECONDS PER SOLVE (w > 0)")
print("=" * 92)
tt = pos.groupby(["section", "arm"]).secs.mean().unstack("arm")
print(tt.round(2).to_string())
print(f"\n  overall  " + "  ".join(f"{a} {pos[pos.arm==a].secs.mean():.2f}s" for a in ARMS))
sp = pos[pos.arm == "FGW"].secs.mean() / pos[pos.arm == "clone spread"].secs.mean()
print(f"  clone spread is {sp:.2f}x faster than FGW on this grid.")

out = HERE / "h2h_reanalysis"
out.mkdir(exist_ok=True)
pd.DataFrame(rows_summary).to_csv(out / "summary.csv", index=False)
pd.DataFrame(rows_paired).to_csv(out / "paired.csv", index=False)
pw.to_csv(out / "per_w.csv", index=False)
for s in order:
    stats[s].to_csv(out / ("splits_" + s.split(" (")[0].replace(" ", "_") + ".csv"))
print(f"\nwrote {out}")
