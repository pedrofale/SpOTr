"""v2 of cand_permuted-tree-excess-agreement.py -- ONE implementation fix.

In v1 the q29 scale fix used a SEPARATE reference magnitude for each gene half
(ref_h = median|C_h(alpha=0)|). At alpha=1 the cost is tree-only and therefore identical for
the two halves BEFORE the rescale, but the two refs differ, so the two halves were solved at
different effective temperatures and A_T(1) < 1. That broke the candidate's structural
guarantee S(1)=0 for a purely implementational reason (|S(1)| up to 0.51).

v2 uses ONE shared reference (geometric mean of the two halves') for both halves in every arm,
which makes the alpha=1 problems literally identical and restores S(1)=0 exactly.

Reduced vs v1 to stay inside the compute cap: R=4 permutation nulls (v1 used 6); accuracy and
the clade-swap / restart controls are taken from the v1 CSV (unaffected by this fix, since the
accuracy sweep is single-view and the controls are relative comparisons under the same bug).
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID6 = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
GRID = GRID6 + [0.85, 1.0]
R_CELL = 4

import importlib.util
spec = importlib.util.spec_from_file_location(
    "candv1", HERE / "cand_permuted-tree-excess-agreement.py")
v1 = importlib.util.module_from_spec(spec)
sys.modules["candv1"] = v1
spec.loader.exec_module(v1)
posterior_mean, concordance, solve = v1.posterior_mean, v1.concordance, v1.solve

old = pd.read_csv(HERE / "cand_permuted-tree-excess-agreement.csv")
ACC = {(int(r.seed), float(r.alpha)): float(r.acc)
       for r in old[(old.variant == "real") & (old.arm == "T") & (old.init == "g0")].itertuples()}

curves, seeds = [], []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    seeds.append(seed); t0 = time.time()
    cidx = np.asarray(d["cell_clade"]); K, n, G = len(d["clades"]), len(cidx), d["X_sc"].shape[1]
    Amat = np.zeros((n, K), np.float32); Amat[np.arange(n), cidx] = 1.0
    omega = jnp.asarray(Amat); xy = np.asarray(d["xy_st"], np.float64); w = np.full(n, 1.0 / n)
    Ctree_real = np.asarray(d["C_tree"], np.float32)
    rng = np.random.default_rng(1000 + seed); pg = rng.permutation(G)
    halves = [pg[:G // 2], pg[G // 2:]]

    _, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                            Ctree_real, d["C_space"])
    g0 = a[:, None] * b[None, :]
    half_in = [S.prepare_ot_inputs(ad.AnnData(np.ascontiguousarray(d["X_sc"][:, h])),
                                   ad.AnnData(np.ascontiguousarray(d["X_st"][:, h])),
                                   Ctree_real, d["C_space"])[0] for h in halves]

    def cell_perm_tree(r):
        p = np.random.default_rng(7000 + 100 * seed + r).permutation(n)
        return Ctree_real[np.ix_(p, p)]

    def arm_A(Ctree):
        CT = jnp.asarray(Ctree); scales, refs = [], []
        for Cfh in half_in:
            sFh, sLh = S.make_cladexfgw_scales(Cfh, CT, C_S, a, b, g0, eps=EPS)
            Ch0 = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cfh, CT, C_S,
                                               a, b, g0, omega, sFh, sLh, equalize="measured")
            scales.append((sFh, sLh)); refs.append(float(jnp.median(jnp.abs(Ch0))))
        ref = float(np.exp(np.mean(np.log(refs))))          # <-- THE FIX: shared reference
        out = {}
        for av in GRID:
            mus = [posterior_mean(solve(av, Cfh, CT, C_S, a, b, omega, sFh, sLh, K, g0, ref), xy)
                   for Cfh, (sFh, sLh) in zip(half_in, scales)]
            out[av] = concordance(mus[0], mus[1], w)
        return out

    blocks = {("real", "T"): arm_A(Ctree_real), ("shuf", "T"): arm_A(cell_perm_tree(99))}
    for r in range(R_CELL):
        blocks[("real", f"P{r}")] = arm_A(cell_perm_tree(r))
        blocks[("shuf", f"P{r}")] = arm_A(cell_perm_tree(50 + r))
    for (variant, arm), c in blocks.items():
        for av in GRID:
            curves.append(dict(seed=seed, variant=variant, arm=arm, alpha=av, A=c[av],
                               acc=ACC[(seed, av)]))
    print(f"  seed {seed} done in {time.time()-t0:.1f}s", flush=True)

cu = pd.DataFrame(curves)
cu.to_csv(HERE / "cand_permuted-tree-excess-agreement_v2.csv", index=False)


def Sc_of(seed, variant):
    sub = cu[(cu.seed == seed) & (cu.variant == variant)]
    AT = sub[sub.arm == "T"].set_index("alpha").A.reindex(GRID)
    AP = pd.concat([g.set_index("alpha").A.reindex(GRID)
                    for _, g in sub[sub.arm != "T"].groupby("arm")], axis=1)
    return AT, AP, AT - AP.mean(1)


print("\n" + "=" * 110)
print("v2 (shared scale reference):  S(alpha) = A_T - mean_r A_P  [REAL tree]  and TRUE accuracy")
print("=" * 110)
print(f"{'seed':>5}" + "".join(f"{'a=' + str(v):>17}" for v in GRID6) + "     |  a=0.85          a=1.00")
per = []
for s_ in seeds:
    AT, AP, Sc = Sc_of(s_, "real")
    acc = pd.Series({v: ACC[(s_, v)] for v in GRID})
    print(f"{s_:>5}" + "".join(f"{Sc[v]:>+9.4f}/{acc[v]:.3f}" for v in GRID6) +
          f"     |{Sc[0.85]:+9.4f}/{acc[0.85]:.3f} {Sc[1.0]:+9.4f}/{acc[1.0]:.3f}")
    dd = np.diff(Sc.values)
    per.append(dict(seed=s_, sel=float(Sc.idxmax()), sel6=float(Sc.reindex(GRID6).idxmax()),
                    opt=float(acc.idxmax()), opt6=float(acc.reindex(GRID6).idxmax()),
                    acc_sel=float(acc[Sc.idxmax()]), acc_a0=float(acc[0.0]),
                    acc_a02=float(acc[0.20]), acc_opt=float(acc.max()),
                    S0=float(Sc[0.0]), S1=float(Sc[1.0]), Smax=float(Sc.max()),
                    Ssd=float(AP.std(1)[Sc.idxmax()]),
                    mono=bool((dd > 0).all() or (dd < 0).all())))
p = pd.DataFrame(per)
print("\n--- construction checks ---")
print(f"  |S(0)| < 1e-10 in {int((p.S0.abs()<1e-10).sum())}/{len(p)}   "
      f"|S(1)| < 1e-10 in {int((p.S1.abs()<1e-10).sum())}/{len(p)}  (max |S(1)| = {p.S1.abs().max():.2e})")
print(f"  argmax S strictly interior : {int(((p.sel>0)&(p.sel<1)).sum())}/{len(p)}")
print(f"  S MONOTONE in alpha        : {int(p.mono.sum())}/{len(p)}")
print("\n--- SELECTION ---")
print(f"  selected : {', '.join(f'{v:.2f}' for v in p.sel)}")
print(f"  optimal  : {', '.join(f'{v:.2f}' for v in p.opt)}")
ag = int((np.abs(p.sel - p.opt) < 1e-9).sum())
print(f"  AGREEMENT {ag}/{len(p)}   mean|sel-opt| {np.abs(p.sel-p.opt).mean():.3f}   "
      f"sel in [0.10,0.35] {int(((p.sel>=0.10)&(p.sel<=0.35)).sum())}/{len(p)}   sd(sel) {p.sel.std():.4f}")
print(f"  MEAN ACCURACY  selected {p.acc_sel.mean():.4f} | a=0 {p.acc_a0.mean():.4f} | "
      f"a=0.2 {p.acc_a02.mean():.4f} | oracle {p.acc_opt.mean():.4f}")
print(f"  beats a=0 {p.acc_sel.mean()>p.acc_a0.mean()} ({int((p.acc_sel>p.acc_a0).sum())}/{len(p)})   "
      f"beats a=0.2 {p.acc_sel.mean()>p.acc_a02.mean()} ({int((p.acc_sel>p.acc_a02).sum())}/{len(p)})")
print(f"  [6-pt grid] selected {', '.join(f'{v:.2f}' for v in p.sel6)} "
      f"agreement {int((np.abs(p.sel6-p.opt6)<1e-9).sum())}/{len(p)}")
print(f"  max_a S > 3sd(perm) in {int((p.Smax>3*p.Ssd).sum())}/{len(p)}")
rho = spearmanr(p.sel, p.opt)
print(f"  Spearman rho(sel, opt) = {rho.statistic:+.3f} (p={rho.pvalue:.3f})")

print("\n--- SHUFFLED-TREE CONTROL ---")
q = pd.DataFrame([dict(seed=s_, sel=float(Sc_of(s_, "shuf")[2].idxmax()),
                       Smax=float(Sc_of(s_, "shuf")[2].max())) for s_ in seeds])
print(f"  selected (shuffled) : {', '.join(f'{v:.2f}' for v in q.sel)}")
print(f"  selected (real)     : {', '.join(f'{v:.2f}' for v in p.sel)}")
print(f"  SAME in {int((np.abs(q.sel.values-p.sel.values)<1e-9).sum())}/{len(p)}   "
      f"mean|diff| {np.abs(q.sel.values-p.sel.values).mean():.3f}")
print(f"  max_a S real {p.Smax.mean():+.4f} vs shuffled {q.Smax.mean():+.4f} "
      f"(ratio {p.Smax.mean()/max(abs(q.Smax.mean()),1e-9):.2f}x)")
p.to_csv(HERE / "cand_permuted-tree-excess-agreement_v2_summary.csv", index=False)
