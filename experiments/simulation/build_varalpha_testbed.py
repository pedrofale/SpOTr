"""Build a testbed where the OPTIMAL alpha actually varies.

Round 2 established that the controlled simulator cannot discriminate alpha selectors:
argmax(acc) = 0.20 in 6/10 seeds, total headroom above the constant alpha=0.2 is +0.0056, and
the zero-information rule "always 0.2" beats every criterion tested on agreement and MAE. So
every "alpha is unlearnable" result from it is UNDERPOWERED.

Fix the instrument before testing anything else. Two knobs should move alpha*:
  target_r        tree-vs-layout concordance (default 0.85) -- a better tree should want MORE alpha
  target_confuse  measured within-pair expression discrimination (default 0.58) -- more
                  confusable expression (LOWER value) should want MORE alpha

Deliverable: an oracle alpha sweep over the grid, and a verdict on whether alpha* spans the
grid. That is the result even if no criterion is ever run on it.
"""
import subprocess, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
R_VALS = [0.55, 0.70, 0.85]
C_VALS = [0.45, 0.58, 0.72]
SEEDS = [0, 1, 2, 3]
OUT = HERE / "varalpha"
OUT.mkdir(exist_ok=True)

# ---------- 1. generate ----------
jobs = []
for r_, c_, sd in itertools.product(R_VALS, C_VALS, SEEDS):
    f = OUT / f"va_r{r_}_c{c_}_s{sd}.npz"
    jobs.append((r_, c_, sd, f))
todo = [j for j in jobs if not j[3].exists()]
print(f"[gen] {len(jobs)} datasets, {len(todo)} to generate", flush=True)
for r_, c_, sd, f in todo:
    cmd = [sys.executable, str(HERE / "simulate_controlled.py"), "--seed", str(sd),
           "--target-r", str(r_), "--target-confuse", str(c_), "--out", str(f)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not f.exists():
        print(f"  FAILED r={r_} c={c_} s={sd}: {p.stderr[-300:]}", flush=True)
    else:
        print(f"  ok r={r_} c={c_} s={sd}", flush=True)

# ---------- 2. oracle alpha sweep ----------
rows = []
for r_, c_, sd, f in jobs:
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"])
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    def cost(av, g):
        return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T,
                                            C_S, a, b, g, omega, sF, sL, equalize="measured")
    REF = float(jnp.median(jnp.abs(cost(0.0, g0))))
    accs = {}
    for av in GRID:
        g = g0
        for _ in range(J):
            C = cost(av, g)
            C = C * (REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))   # scale fixed
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        accs[av] = float((spot_ci[np.asarray(g).argmax(1)] == cidx).mean())
    real = np.asarray(d["realized"], float)
    rows.append(dict(target_r=r_, target_confuse=c_, seed=sd, in_band=bool(d["in_band"][0]),
                     got_r=real[0], got_sep=real[1], got_confuse=real[2],
                     alpha_star=max(accs, key=lambda k: accs[k]),
                     **{f"acc_{v}": accs[v] for v in GRID}))
    print(f"  r={r_} c={c_} s={sd} in_band={rows[-1]['in_band']}  alpha*={rows[-1]['alpha_star']}"
          f"  acc " + " ".join(f"{accs[v]:.3f}" for v in GRID), flush=True)

t = pd.DataFrame(rows); t.to_csv(HERE / "varalpha_oracle.csv", index=False)
ib = t[t.in_band]
print("\n" + "=" * 90)
print(f"generated {len(t)} datasets, {len(ib)} in band")
print(f"\nalpha* distribution (in-band): {dict(ib.alpha_star.value_counts().sort_index())}")
print(f"alpha* spans {ib.alpha_star.nunique()} of {len(GRID)} grid values")
print("\nmean alpha* by setting:")
piv = ib.pivot_table(index="target_r", columns="target_confuse", values="alpha_star", aggfunc="mean")
print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
print("\nheadroom above the best single hindsight constant:")
best_c = max(GRID, key=lambda v: ib[f"acc_{v}"].mean())
oracle = ib.apply(lambda x: x[f"acc_{x.alpha_star}"], axis=1).mean()
print(f"  best hindsight constant alpha={best_c} -> {ib[f'acc_{best_c}'].mean():.4f}")
print(f"  per-dataset oracle                -> {oracle:.4f}")
print(f"  HEADROOM                          -> {oracle - ib[f'acc_{best_c}'].mean():+.4f}"
      f"   (old testbed: +0.0056)")
print(f"  constant-selector agreement       -> "
      f"{int((ib.alpha_star == best_c).sum())}/{len(ib)}   (old testbed: 6/10)")
