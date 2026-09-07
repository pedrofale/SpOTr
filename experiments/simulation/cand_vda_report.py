"""Report for cand_vda-two-channel-variance-decomposition.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
r = pd.read_csv(HERE / "cand_vda-two-channel-variance-decomposition.csv")
pd.set_option("display.width", 250)
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
mono = lambda v: bool((np.diff(v) > 0).all() or (np.diff(v) < 0).all())


def summarise(sub, scol, label):
    per = []
    for s_, g in sub.groupby("seed"):
        g = g.set_index("alpha").sort_index()
        v = g[scol].values
        per.append(dict(seed=s_, sel=float(g[scol].idxmin()), opt=float(g.acc.idxmax()),
                        mono=mono(v),
                        mono_g=mono(g.Sg_icc.values), mono_t=mono(g.St_icc.values),
                        acc_sel=float(g.acc[g[scol].idxmin()]), acc_a0=float(g.acc[0.0]),
                        acc_a02=float(g.acc[0.20]), acc_opt=float(g.acc.max())))
    p = pd.DataFrame(per)
    print(f"\n### {label}   [S = {scol}]   n={len(p)} seeds")
    print(f"  selected alpha per seed : {', '.join(f'{v:.2f}' for v in p.sel)}")
    print(f"  optimal  alpha per seed : {', '.join(f'{v:.2f}' for v in p.opt)}")
    print(f"  agreement (argmin S == argmax acc) : {int((np.abs(p.sel-p.opt)<1e-9).sum())}/{len(p)}")
    print(f"  mean |selected - optimal|          : {np.abs(p.sel-p.opt).mean():.3f}")
    print(f"  S monotone in alpha                : {int(p.mono.sum())}/{len(p)} seeds")
    print(f"  V_gene(ICC) monotone / V_tree(ICC) monotone : "
          f"{int(p.mono_g.sum())}/{len(p)}  {int(p.mono_t.sum())}/{len(p)}")
    print(f"  ACCURACY  selected {p.acc_sel.mean():.4f} | alpha=0 {p.acc_a0.mean():.4f} | "
          f"fixed 0.2 {p.acc_a02.mean():.4f} | oracle {p.acc_opt.mean():.4f}")
    print(f"  beats alpha=0 : {p.acc_sel.mean()>p.acc_a0.mean()}   "
          f"beats fixed 0.2 : {p.acc_sel.mean()>p.acc_a02.mean()}")
    return p


main = r[(r.tag == "main")]
print("=" * 100)
print("PER-SEED TABLE (main run, R=3 splits, scale held fixed)")
print("=" * 100)
print(f"{'seed':>5} " + "".join(f"{'a='+str(v):>22}" for v in GRID))
print(f"{'':>5} " + "".join(f"{'acc / S_icc / S_raw':>22}" for _ in GRID))
for s_, g in main.groupby("seed"):
    g = g.set_index("alpha")
    print(f"{s_:>5} " + "".join(f"{g.acc[v]:>7.3f}/{g.S_icc[v]:.3f}/{g.S_raw[v]:.3f}" for v in GRID))

print("\n" + "=" * 100)
print("COMPONENTS  (mean over seeds): V_gene must fall, V_tree must rise -- the claimed mechanism")
print("=" * 100)
m = main.groupby("alpha")[["V_gene", "V_tree", "Sg_icc", "St_icc", "S_raw", "S_icc",
                           "medC0", "ent", "Sp", "acc"]].mean()
print(m.to_string())

pm_icc = summarise(main, "S_icc", "MAIN, repaired criterion (ICC / scale-free)")
pm_raw = summarise(main, "S_raw", "MAIN, criterion AS SPECIFIED (raw um^2 / 2trCov)")

for tag, lab in [("shuffled", "SHUFFLED-TREE CONTROL (leaf-permuted patristic)"),
                 ("frac75", "SPLIT-FRACTION CHECK (75/25 gene split)"),
                 ("noscalefix", "NO SCALE FIX (cost magnitude free to drift, as shipped)")]:
    sub = r[r.tag == tag]
    if len(sub):
        print("\n" + "=" * 100)
        p = summarise(sub, "S_icc", lab)
        summarise(sub, "S_raw", lab + "  [raw]")

w = r[r.tag == "wide"]
if len(w):
    print("\n" + "=" * 100)
    print("WIDE GRID (0 .. 0.95) -- does S have ANY interior minimum over the full range?")
    print("=" * 100)
    mw = w.groupby("alpha")[["V_gene", "V_tree", "S_raw", "S_icc", "acc"]].mean()
    print(mw.to_string())
    per = []
    for s_, g in w.groupby("seed"):
        g = g.set_index("alpha").sort_index()
        per.append((s_, float(g.S_icc.idxmin()), float(g.S_raw.idxmin()), float(g.acc.idxmax())))
    print("  seed / argmin S_icc / argmin S_raw / argmax acc:")
    for t in per:
        print(f"   {t[0]:>3}  {t[1]:.2f}  {t[2]:.2f}  {t[3]:.2f}")

# real vs shuffled: does the criterion behave differently?
sh = r[r.tag == "shuffled"]
if len(sh):
    a = main.groupby("seed").apply(lambda g: g.set_index("alpha").S_icc.idxmin())
    b = sh.groupby("seed").apply(lambda g: g.set_index("alpha").S_icc.idxmin())
    j = pd.concat([a.rename("real"), b.rename("shuf")], axis=1)
    print("\nSHUFFLED vs REAL selected alpha (S_icc):")
    print(j.to_string())
    print(f"  identical selection in {int((j.real==j.shuf).sum())}/{len(j)} seeds; "
          f"mean real {j.real.mean():.3f} vs shuffled {j.shuf.mean():.3f}")
    ar = main.groupby("seed").apply(lambda g: g.set_index("alpha").S_raw.idxmin())
    br = sh.groupby("seed").apply(lambda g: g.set_index("alpha").S_raw.idxmin())
    print(f"  [S_raw] mean real {ar.mean():.3f} vs shuffled {br.mean():.3f}; "
          f"identical in {int((ar.values==br.values).sum())}/{len(ar)}")
