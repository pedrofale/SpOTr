"""Does the tree's TOPOLOGY help, or only its clade membership?

The cell x cell patristic metric is saturated: 49% of M3-4 pairs sit at exactly the maximum,
between-clade distances have CV 0.04, and the same/different-clade binary alone explains
R2 = 0.56 of its variance. So the GW term may be doing nothing more than "same-clade cells
belong near each other".

Four tree metrics, identical everywhere else:
  real       the measured patristic distances
  flat       within-clade distances kept; ALL between-clade distances set to their mean
             -> keeps clade membership + within-clade structure, destroys between-clade topology
  binary     within-clade 0, between-clade 1 -> pure clade indicator, no topology at all
  shuffled   real matrix with leaf labels permuted -> same distance spectrum, no cell identity
If real ~ flat ~ binary, claim (a) is about clade labels, not about the phylogeny.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
GRID = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
rng = np.random.default_rng(0)
rows = []

for SEC in ["M3-1", "M3-2", "M3-4"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    K, n = len(clades), len(cc)
    cidx = np.array([clades.index(c) for c in cc])
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    COMP = np.load(HERE / f"comp_{SEC}.npy").astype(float)
    CT0 = np.asarray(d["C_tree"], float)
    same = cidx[:, None] == cidx[None, :]

    flat = CT0.copy(); flat[~same] = CT0[~same].mean()
    binary = np.where(same, 0.0, 1.0); np.fill_diagonal(binary, 0.0)
    perm = rng.permutation(n); shuf = CT0[np.ix_(perm, perm)]

    for tname, CT in [("real", CT0), ("flat", flat), ("binary", binary), ("shuffled", shuf)]:
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                                 CT.astype(np.float32),
                                                 np.asarray(d["C_space"], np.float32))
        omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
        sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
        for av in GRID:
            def build(g):
                return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf,
                                                    C_T, C_S, a, b, g, omega, sF, sL,
                                                    equalize="measured")
            g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
            gn = np.asarray(g)
            acc = float(COMP[gn.argmax(1), cidx].mean()) if np.isfinite(gn).all() else np.nan
            rows.append(dict(section=SEC, tree=tname, alpha=av, soft=acc))
        print(f"  {SEC} {tname:9s} " + " ".join(f"{r['soft']:.3f}" for r in rows[-len(GRID):]),
              flush=True)

r = pd.DataFrame(rows); r.to_csv(HERE / "flat_tree_control.csv", index=False)
print("\n" + "=" * 78)
print(f"{'section':<9}{'tree':<10}{'best alpha':>11}{'best acc':>10}{'acc at a=0':>12}{'gain':>9}")
print("-" * 61)
for (s_, tn), g_ in r.groupby(["section", "tree"]):
    g_ = g_.set_index("alpha")
    print(f"{s_:<9}{tn:<10}{g_.soft.idxmax():>11.2f}{g_.soft.max():>10.4f}"
          f"{g_.soft[0.0]:>12.4f}{g_.soft.max()-g_.soft[0.0]:>+9.4f}")
print("\nmean best accuracy across the three sections:")
for tn, g_ in r.groupby("tree"):
    print(f"  {tn:<10}{g_.groupby('section').soft.max().mean():.4f}")
