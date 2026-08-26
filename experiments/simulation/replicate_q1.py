"""Does the Q1 result -- the tree improves the mapping -- replicate across simulation seeds?

One seed is thin evidence for the headline claim, and every earlier positive on this project
that was reported from a single run later inverted. Regenerates the coarse-to-fine testbed
under several seeds and re-runs the global alpha/phi sweep on each.
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from scipy.spatial.distance import cdist

EPS, T, J = 0.02, 50, 3
AG = [0.0, 0.20, 0.35, 0.50, 0.70]
PG = [0.0, 0.50, 1.0]
rows = []
for seed in [0, 1, 3, 4, 5, 6, 7]:
    f = HERE / ("sim_cf.npz" if seed == 0 else f"sim_cf_s{seed}.npz")
    if not f.exists():
        # BirthDeathFitnessSimulator kills all lineages for some seeds; that is a simulator
        # failure, not a result -- skip rather than let it look like a data point
        p = subprocess.run([sys.executable, str(HERE / "simulate_coarse_fine.py"),
                            "--seed", str(seed), "--out", f.name],
                           cwd=HERE, capture_output=True)
        if p.returncode != 0 or not f.exists():
            print(f"seed {seed}: simulation failed (lineage extinction) -- skipped")
            continue
    d = np.load(f, allow_pickle=True)
    cell_clade, cell_sub, spot_sub = d["cell_clade"], d["cell_sub"], d["spot_sub"]
    clades, is_tf = d["clades"], d["is_tree_fine"]
    K = len(clades)
    tf = np.isin(cell_clade, clades[is_tf])
    _D = cdist(d["xy_sc"], d["xy_st"]) / np.linalg.norm(d["xy_st"].max(0) - d["xy_st"].min(0))
    A = np.zeros((len(cell_clade), K), dtype=np.float32)
    for i, c in enumerate(cell_clade):
        A[i, list(clades).index(c)] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega, Omega = jnp.asarray(A), jnp.asarray(A @ A.T)
    g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    def run(av, pv):
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             use_decomp=True, Omega=Omega,
                                             phi_k=jnp.full((K,), pv, dtype=jnp.float32),
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        g = np.asarray(g)
        if not np.isfinite(g).all():
            return np.nan, np.nan, np.nan
        j = g.argmax(1); ok = spot_sub[j] == cell_sub
        w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
        return ok.mean(), ok[tf].mean(), (w * _D).sum(1).mean()

    base = run(0.0, 0.0)
    best, bcfg = (-np.inf,), None
    for av in AG:
        for pv in PG:
            if av == 0.0 and pv > 0:
                continue
            v = run(av, pv)
            if v[0] > best[0]:
                best, bcfg = v, (av, pv)
    rows.append(dict(seed=seed, acc0=base[0], acc0_tf=base[1], ed0=base[2],
                     acc=best[0], acc_tf=best[1], ed=best[2],
                     alpha=bcfg[0], phi=bcfg[1]))
    print(f"seed {seed}: alpha=0 -> acc_sub {base[0]:.4f} (tf {base[1]:.4f}, E-dist {base[2]:.4f})"
          f" | best (a={bcfg[0]}, p={bcfg[1]}) -> {best[0]:.4f} (tf {best[1]:.4f}, "
          f"E-dist {best[2]:.4f})  gain {best[0]-base[0]:+.4f}")

r = pd.DataFrame(rows)
r["gain"] = r.acc - r.acc0
r["gain_tf"] = r.acc_tf - r.acc0_tf
r["ed_gain"] = r.ed0 - r.ed
print("\n" + "=" * 74)
print(r[["seed", "alpha", "phi", "acc0", "acc", "gain", "gain_tf", "ed_gain"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nsubclade-accuracy gain from the tree: mean {r.gain.mean():+.4f}  "
      f"(per seed: {', '.join(f'{v:+.4f}' for v in r.gain)})")
print(f"  on tree-fine clades only:           mean {r.gain_tf.mean():+.4f}  "
      f"(per seed: {', '.join(f'{v:+.4f}' for v in r.gain_tf)})")
print(f"  E-dist reduction:                   mean {r.ed_gain.mean():+.4f}")
print("-> " + ("the tree helps in EVERY seed" if (r.gain > 0).all() else "NOT consistent across seeds"))
r.to_csv(HERE / "replicate_q1.csv", index=False)
