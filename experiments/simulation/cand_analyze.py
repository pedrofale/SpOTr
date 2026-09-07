import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
csv = sys.argv[1] if len(sys.argv) > 1 else "cand_two-sided-split-half-stability.csv"
r = pd.read_csv(HERE / csv)
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
mono = lambda x: bool((np.diff(x) > 0).all() or (np.diff(x) < 0).all())

def block(variant, tag, mode, grid=GRID, knob="1x"):
    sub = r[r["mode"] == mode]
    ae, at = f"Aexpr_{variant}", f"Atree_{variant}_{knob}"
    if at not in sub.columns: return None
    per = []
    print(f"\n{'='*112}\n### {tag}   [mode={mode}, agreement={variant}, tree-perturbation={knob}]")
    hdr = f"{'seed':>5} " + "".join(f"{'a='+format(v,'.2f'):>26}" for v in grid)
    print(hdr); print(f"{'':>5} " + "".join(f"{'S5 / Aexpr / Atree / acc':>26}" for _ in grid))
    print("-"*len(hdr))
    for s, g in sub.groupby("seed"):
        g = g.set_index("alpha").sort_index()
        g = g.loc[[v for v in grid]]
        if g[at].isna().any(): continue
        S5 = np.sqrt(g[ae].values * g[at].values)
        line = f"{s:>5} " + "".join(
            f"{S5[i]:>7.4f}/{g[ae].values[i]:.3f}/{g[at].values[i]:.3f}/{g.acc.values[i]:.3f}"
            for i in range(len(grid)))
        print(line)
        sel = grid[int(np.argmax(S5))]
        opt = float(g.acc.idxmax())
        per.append(dict(seed=s, sel=sel, opt=opt, S5mono=mono(S5),
                        Ae_inc=bool((np.diff(g[ae].values) > 0).all()),
                        At_dec=bool((np.diff(g[at].values) < 0).all()),
                        interior=bool(0 < int(np.argmax(S5)) < len(grid)-1),
                        acc_sel=float(g.acc[sel]), acc_0=float(g.acc[0.0]),
                        acc_02=float(g.acc[0.20]), acc_opt=float(g.acc.max())))
    p = pd.DataFrame(per); n = len(p)
    print("-"*len(hdr))
    print(f"selected alpha per seed : {', '.join(f'{v:.2f}' for v in p.sel)}")
    print(f"accuracy-optimal alpha  : {', '.join(f'{v:.2f}' for v in p.opt)}")
    print(f"agreement_n/total       : {int((np.abs(p.sel-p.opt)<1e-9).sum())}/{n}")
    print(f"within one grid step    : {int((np.abs(p.sel-p.opt)<=0.16).sum())}/{n}")
    print(f"mean |selected - optimal|: {np.abs(p.sel-p.opt).mean():.4f}")
    print(f"monotone_seeds (S5)     : {int(p.S5mono.sum())}/{n}    interior argmax: {int(p.interior.sum())}/{n}")
    print(f"A_expr increasing 6/6   : {int(p.Ae_inc.sum())}/{n}    A_tree decreasing 6/6: {int(p.At_dec.sum())}/{n}")
    print(f"ACCURACY  selected {p.acc_sel.mean():.4f} | alpha=0 {p.acc_0.mean():.4f} | "
          f"fixed 0.2 {p.acc_02.mean():.4f} | oracle {p.acc_opt.mean():.4f}")
    print(f"beats alpha=0: {p.acc_sel.mean()>p.acc_0.mean()} ({int((p.acc_sel>p.acc_0).sum())}/{n} seeds) | "
          f"beats fixed 0.2: {p.acc_sel.mean()>p.acc_02.mean()} ({int((p.acc_sel>p.acc_02).sum())}/{n} seeds)")
    return p

print("\n########## ZERO-SOLVE PRE-TEST (margin / entropy vs alpha, real tree) ##########")
sub = r[r["mode"] == "real"]
pm = []
for s, g in sub.groupby("seed"):
    g = g.set_index("alpha").sort_index().loc[GRID]
    pm.append(dict(seed=s, argmax_margin=float(g.margin.idxmax()), mono=mono(g.margin.values),
                   H_drift=float(g.H_can.max()-g.H_can.min()), H0=float(g.H_can[0.0]),
                   Htun_drift=float(g.H_tun.max()-g.H_tun.min())))
    print(f"  seed {s} margin: " + " ".join(f"{v:.3f}" for v in g.margin.values) +
          f"   | H(eps fixed): " + " ".join(f"{v:.3f}" for v in g.H_can.values) +
          f"   | H(eps tuned): " + " ".join(f"{v:.3f}" for v in g.H_tun.values))
pm = pd.DataFrame(pm)
print(f"  argmax(margin) per seed: {', '.join(f'{v:.2f}' for v in pm.argmax_margin)}")
print(f"  margin monotone in alpha: {int(pm.mono.sum())}/{len(pm)}; "
      f"argmax at a grid endpoint: {int(((pm.argmax_margin==0.0)|(pm.argmax_margin==0.70)).sum())}/{len(pm)}; "
      f"in [0.10,0.35]: {int(((pm.argmax_margin>=0.10)&(pm.argmax_margin<=0.35)).sum())}/{len(pm)}")
print(f"  mean row-entropy drift across alpha: fixed eps {pm.H_drift.mean():.4f}, "
      f"tuned eps {pm.Htun_drift.mean():.4f} (target held at H(alpha=0))")

print("\n########## MECHANISM CHECK at alpha=1.0 (A_tree(1)=c1, A_expr(1) pin) ##########")
for s, g in sub.groupby("seed"):
    g = g.set_index("alpha")
    if 1.0 in g.index:
        print(f"  seed {s}: A_expr(1)={g.Aexpr_deb[1.0]:.4f} (pin=1)  A_tree(1)={g.Atree_deb_1x[1.0]:.4f}  "
              f"| A_tree(0)={g.Atree_deb_1x[0.0]:.4f} (pin=1)  A_expr(0)={g.Aexpr_deb[0.0]:.4f}  acc(1)={g.acc[1.0]:.3f}")

pa = block("deb", "PRIMARY: repaired (blur-corrected agreement)", "real")
pb = block("raw", "as-specified cosine agreement", "real")
pc = block("hard", "hard-argmax spatial agreement", "real")
ps = block("deb", "SHUFFLED-TREE CONTROL", "shuffled")
print("\n########## KNOB TEST (tree perturbation at 0.5x / 2x calibrated magnitude) ##########")
for kn in ("0.5x", "1x", "2x"):
    pk = block("deb", f"knob {kn}", "real", knob=kn)

print("\n\n##################### SAME CONTROLS FOR THE AS-SPECIFIED (raw cosine) VARIANT #####################")
block("raw", "SHUFFLED-TREE CONTROL (raw cosine)", "shuffled")
for kn in ("0.5x", "1x", "2x"):
    block("raw", f"knob {kn} (raw cosine)", "real", knob=kn)
print("\n\n##################### MIN INSTEAD OF GEOMETRIC MEAN (raw cosine, real) #####################")
sub = r[r["mode"] == "real"]
per = []
for s, g in sub.groupby("seed"):
    g = g.set_index("alpha").sort_index().loc[GRID]
    Smin = np.minimum(g["Aexpr_raw"].values, g["Atree_raw_1x"].values)
    sel = GRID[int(np.argmax(Smin))]
    per.append(dict(seed=s, sel=sel, opt=float(g.acc.idxmax()), acc_sel=float(g.acc[sel]),
                    acc_0=float(g.acc[0.0]), acc_02=float(g.acc[0.20]), acc_opt=float(g.acc.max())))
p = pd.DataFrame(per)
print(f"selected: {', '.join(f'{v:.2f}' for v in p.sel)}   optimal: {', '.join(f'{v:.2f}' for v in p.opt)}")
print(f"agreement {int((np.abs(p.sel-p.opt)<1e-9).sum())}/{len(p)}  mean|sel-opt| {np.abs(p.sel-p.opt).mean():.3f}  "
      f"acc sel {p.acc_sel.mean():.4f} | a=0 {p.acc_0.mean():.4f} | 0.2 {p.acc_02.mean():.4f} | oracle {p.acc_opt.mean():.4f}")
