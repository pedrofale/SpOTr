"""Reliability at the level of the PARAMETER: is a CLADE's optimal alpha reproducible?

A cell's argmin is noisy, but alpha_k averages over ~150 cells. Split the spots in half, solve
each independently, and compare each clade's optimal alpha (from its mean error curve) between
halves. Compare against the same thing for RANDOM groupings of the same sizes -- if random
groups are equally reproducible, the reproducibility is an artefact of averaging, not of clade.
"""
import sys
from pathlib import Path
import numpy as np, anndata as ad, jax.numpy as jnp
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

EPS, T, J, TAU = 0.02, 50, 3, 0.01
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90])
NREP = 3
rng = np.random.default_rng(0)

for SEC in ["M3-4", "M3-1"]:
    d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
    xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    clades = list(np.asarray(d["clades"]).astype(str))
    cidx = np.array([clades.index(c) for c in cc])
    CT = np.asarray(d["C_tree"], np.float32); CSp = np.asarray(d["C_space"], np.float32)
    n, m, K = len(cc), X_st.shape[0], len(clades)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    keep = [k for k in range(K) if (cidx == k).sum() >= 30]

    def errors(cols):
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(
            ad.AnnData(X_sc), ad.AnnData(X_st[cols]), CT, CSp[np.ix_(cols, cols)])
        om = jnp.asarray(A); g0 = a[:, None] * b[None, :]
        sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
        E = np.zeros((n, len(GRID)))
        for c_, av in enumerate(GRID):
            def build(g):
                return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf,
                                                    C_T, C_S, a, b, g, om, sF, sL,
                                                    equalize="measured")
            g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
            E[:, c_] = np.linalg.norm(xy_st[cols][np.asarray(g).argmax(1)] - xy_sc, axis=1)
        return E

    def group_opt(E, lab, groups):
        """optimal alpha per group, from the group's mean error curve"""
        return np.array([GRID[E[lab == k].mean(0).argmin()] for k in groups])

    print(f"\n=== {SEC} ===  {len(keep)} clades with >=30 cells", flush=True)
    rc, rr_, rnd = [], [], []
    for rep in range(NREP):
        h = rng.random(m) < 0.5
        EA, EB = errors(np.where(h)[0]), errors(np.where(~h)[0])
        oA, oB = group_opt(EA, cidx, keep), group_opt(EB, cidx, keep)
        r = pearsonr(oA, oB)[0]; rho = spearmanr(oA, oB)[0]
        rc.append(r); rr_.append(rho)
        # random groupings of the SAME sizes -- controls for the averaging effect
        sizes = [int((cidx == k).sum()) for k in keep]
        perm = rng.permutation(n); rlab = np.full(n, -1); p0 = 0
        for gi, sz in enumerate(sizes):
            rlab[perm[p0:p0+sz]] = gi; p0 += sz
        gs = list(range(len(sizes)))
        rA, rB = group_opt(EA, rlab, gs), group_opt(EB, rlab, gs)
        rnd.append(pearsonr(rA, rB)[0])
        print(f"  rep{rep}: clade optimal alpha, half A vs half B   r = {r:+.3f}  rho = {rho:+.3f}"
              f"  | exact agreement {np.mean(oA==oB):.2f}"
              f"  || size-matched RANDOM groups r = {rnd[-1]:+.3f}", flush=True)
        print(f"          A: {' '.join(f'{v:g}' for v in oA)}")
        print(f"          B: {' '.join(f'{v:g}' for v in oB)}")
    print(f"  MEAN clade-level reliability r = {np.mean(rc):+.3f} (rho {np.mean(rr_):+.3f})")
    print(f"  MEAN random-grouping control  r = {np.mean(rnd):+.3f}")
    print(f"  --> clade beats size-matched random by {np.mean(rc)-np.mean(rnd):+.3f}")
