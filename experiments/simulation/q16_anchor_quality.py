"""Q16: how good does the anchor have to be?

If Q15 shows a clean CNV channel fixes the objective, the practical question is how much
degradation it tolerates. Real CNV calls on Visium are noisy: Numbat/CalicoST infer copy
number from pooled spot expression, over a limited number of segments, with substantial
uncertainty. An anchor that only works when perfect is not usable.

Two axes are degraded independently:
  noise    Gaussian noise added to the spot CNV profile (as a multiple of the simulated level)
  bins     number of genomic segments retained (fewer segments = coarser calls)

For each setting: does argmin(CNV loss) still land on alpha > 0, and how far is it from the
accuracy optimum?
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
NOISE = [0.0, 0.5, 1.0, 2.0, 4.0]
BINS = [150, 50, 20, 8]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(5000 + seed)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"])
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    cnv_s, cnv_c = np.asarray(d["cnv_spot"], float), np.asarray(d["cnv_clone"], float)
    nb_all = cnv_s.shape[1]

    couplings = {}
    for av in GRID:
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        couplings[av] = g
    accs = {av: (spot_ci[np.asarray(g).argmax(1)] == cidx).mean() for av, g in couplings.items()}
    a_acc = max(accs, key=lambda k: accs[k])

    for nz in NOISE:
        for nb in BINS:
            keep = rg.choice(nb_all, size=nb, replace=False)
            Y = cnv_s[:, keep] + rg.normal(0, 0.15 * nz, size=(cnv_s.shape[0], nb))
            sig = jnp.asarray((cnv_c[:, keep] * 2.0).astype(np.float32))
            Yj = jnp.asarray(Y.astype(np.float32))
            ls = {av: float(S.deconvolution_loss_ls(Yj, g, jnp.asarray(A), sig))
                  for av, g in couplings.items()}
            a_cnv = min(ls, key=lambda k: ls[k])
            rows.append(dict(seed=seed, noise=nz, bins=nb, a_cnv=a_cnv, a_acc=a_acc,
                             hit=int(a_cnv == a_acc), pos=int(a_cnv > 0),
                             err=abs(a_cnv - a_acc)))

r = pd.DataFrame(rows)
n = r.seed.nunique()
print(f"{n} seeds. Each cell: % of seeds where argmin(CNV loss) > 0  /  mean |selected - optimal|\n")
print(f"{'noise x':>9} " + "".join(f"{str(bb) + ' bins':>16}" for bb in BINS))
print("-" * (9 + 16 * len(BINS)))
for nz in NOISE:
    line = f"{nz:>9.1f} "
    for bb in BINS:
        s = r[(r.noise == nz) & (r.bins == bb)]
        line += f"{100 * s.pos.mean():>8.0f}% /{s.err.mean():>6.3f}"
    print(line)
print(f"\nreference: the EXPRESSION loss selects alpha > 0 in 0% of seeds at any setting")
best = r[(r.noise == 0.0) & (r.bins == 150)]
print(f"clean anchor (no extra noise, all bins): alpha > 0 in {100*best.pos.mean():.0f}% of seeds, "
      f"mean |error| {best.err.mean():.3f}")
worst_ok = r[r.groupby(["noise", "bins"]).pos.transform("mean") >= 0.8]
if len(worst_ok):
    w = worst_ok.groupby(["noise", "bins"]).size().reset_index()
    print(f"settings where alpha > 0 is selected in >=80% of seeds: "
          f"{', '.join(f'noise x{a:.1f}/{b} bins' for a, b, _ in w.values)}")
r.to_csv(HERE / "q16_anchor_quality.csv", index=False)
