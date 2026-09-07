"""Is a cell's preferred alpha a reproducible property, or noise?

Split the SPOTS into two disjoint halves and solve each mapping problem independently. Each
cell then gets two independent estimates of its best alpha. If they agree, the within-clade
variation is real and worth partitioning. If not, there is nothing there to partition and
clade may already capture most of the reliable signal.

Also reports the ceiling this puts on ANY partition: reliability bounds how much of the
variance a perfect grouping could explain.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
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

    rs, rho_s, eta_A, eta_B = [], [], [], []
    def eta2(v, lab):
        gm = v.mean()
        ssb = sum((v[lab == k].mean() - gm) ** 2 * (lab == k).sum() for k in np.unique(lab))
        return ssb / max(((v - gm) ** 2).sum(), 1e-12)

    print(f"\n=== {SEC} === {n} cells, {m} spots -> two halves of ~{m//2}", flush=True)
    for rep in range(NREP):
        h = rng.random(m) < 0.5
        cA, cB = np.where(h)[0], np.where(~h)[0]
        EA, EB = errors(cA), errors(cB)
        bA, bB = GRID[EA.argmin(1)], GRID[EB.argmin(1)]
        r = pearsonr(bA, bB)[0]; rr = spearmanr(bA, bB)[0]
        rs.append(r); rho_s.append(rr)
        eta_A.append(eta2(bA, cidx)); eta_B.append(eta2(bB, cidx))
        print(f"  rep{rep}: test-retest of each cell's best alpha  r = {r:+.3f}  "
              f"rho = {rr:+.3f}  | agree exactly {np.mean(bA==bB):.3f} "
              f"| eta2(clade) {eta_A[-1]:.4f} / {eta_B[-1]:.4f}", flush=True)

    r_mean = float(np.mean(rs))
    print(f"  MEAN test-retest reliability r = {r_mean:+.3f}")
    print(f"  mean eta2 explained by clade  = {np.mean(eta_A+eta_B):.4f}")
    if r_mean > 0:
        print(f"  --> reliable variance is {r_mean:.3f} of the total; clade explains "
              f"{np.mean(eta_A+eta_B):.4f}, i.e. {np.mean(eta_A+eta_B)/r_mean:.1%} of what is "
              f"reliably there. A perfect partition could reach {r_mean:.3f}.")
    else:
        print("  --> no reliable per-cell signal: the within-clade spread is noise, and no "
              "finer partition can capture it.")
