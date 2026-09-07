"""Does the alpha a cell 'wants' cluster BY CLADE? That is what per-clade alpha requires.

Per-cell placement error (distance from the cell's true position to the spot it is assigned)
on an alpha grid. Then: how much of the variance in each cell's best alpha is explained by
clade membership, against a label-permutation null.
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

for SEC in ["M3-1", "M3-4"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    n, K = len(cc), len(clades)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                             np.asarray(d["C_tree"], np.float32),
                                             np.asarray(d["C_space"], np.float32))
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    E = np.zeros((n, len(GRID)))
    for c_, av in enumerate(GRID):
        def build(g):
            return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T,
                                                C_S, a, b, g, omega, sF, sL, equalize="measured")
        g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
        E[:, c_] = np.linalg.norm(xy_st[np.asarray(g).argmax(1)] - xy_sc, axis=1)
    best = np.array(GRID)[E.argmin(1)]

    def eta2(v, lab):
        gm = v.mean()
        ssb = sum((v[lab == k].mean() - gm) ** 2 * (lab == k).sum() for k in np.unique(lab))
        return ssb / max(((v - gm) ** 2).sum(), 1e-12)

    e_obs = eta2(best, cidx)
    null = np.array([eta2(best, rng.permutation(cidx)) for _ in range(999)])
    pv = (np.sum(null >= e_obs) + 1) / 1000

    print(f"\n=== {SEC} === {n} cells")
    print(f"  median error by alpha: " + " ".join(f"{v:.0f}" for v in np.median(E, axis=0)))
    print(f"  distribution of each cell's best alpha: "
          + " ".join(f"{v:g}:{int((best==v).sum())}" for v in GRID))
    print(f"  variance in best alpha explained by CLADE: eta2 = {e_obs:.4f}"
          f"   permutation null {null.mean():.4f} +- {null.std():.4f}   p = {pv:.3f}")
    # how flat is a cell's own error-vs-alpha curve? if flat, 'best alpha' is noise
    span = (E.max(1) - E.min(1))
    print(f"  per-cell error span across the alpha grid: median {np.median(span):.0f} um "
          f"(vs median error {np.median(E):.0f} um)")
    # per-clade mean-error curves: do clades actually peak at different alpha?
    rows = []
    for k, c_ in enumerate(clades):
        m = cidx == k
        if m.sum() < 30: continue
        mu = E[m].mean(0)
        rows.append(dict(clade=c_, n=int(m.sum()), best_alpha=GRID[int(mu.argmin())],
                         err_at_best=mu.min(), err_at_globalbest=mu[int(np.median(E,0).argmin())]))
    t = pd.DataFrame(rows)
    print(f"  clades with >=30 cells: {len(t)}; their best alpha: "
          + " ".join(f"{v:g}:{int((t.best_alpha==v).sum())}" for v in GRID if (t.best_alpha==v).any()))
    t.to_csv(HERE / f"percell_alpha_{SEC}.csv", index=False)
    np.save(HERE / f"percell_err_{SEC}.npy", E)
