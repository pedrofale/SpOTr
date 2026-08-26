"""Q20: can a Brownian-motion-in-space likelihood on the tree select alpha?

The expression loss is monotone DECREASING in alpha because it measures how well the coupling
reconstructs expression, which the linear OT term optimises directly. A BM likelihood measures
how well the mapped positions agree with the TREE, which the GW term optimises directly, so it
should be monotone INCREASING -- the same trap at the other end.

If that holds, neither alone can select alpha, but their SUM should have an interior optimum.
And the sum is not an arbitrary blend: it is a joint likelihood, with the mapping asked to
explain the expression AND to be consistent with spatial evolution along the tree.

BM on a tree gives leaf positions covariance V = sigma^2 * S with S[i,j] the shared root-to-MRCA
path. For an ultrametric of height H that is S = H - d(i,j)/2. Profiling out sigma^2 and the
root state mu, and dropping the log|V| term (constant across alpha):

    sigma_hat^2 = tr(r' V^-1 r) / (n*k),  r = x - mu_GLS
    logL_BM     = -0.5 * n*k * (log(2*pi*sigma_hat^2) + 1) + const

so maximising logL_BM means minimising the GLS residual under the phylogenetic covariance.
The expression term is the same construction on the deconvolution MSE. Their natural relative
weight is the ratio of observation counts, but spot-gene values are strongly correlated so
that over-counts the expression evidence; the weight is therefore also swept, to ask whether
ANY fixed weight selects the right alpha consistently.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    cidx, spot_ci, xy_st = d["cell_clade"], d["spot_clade"], d["xy_st"]
    K = len(d["clades"]); n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    Y = jnp.asarray(np.asarray(d["X_st"], np.float32))
    sig = jnp.asarray(np.stack([np.asarray(d["X_sc"])[cidx == k].mean(0)
                                for k in range(K)]).astype(np.float32) * 2.0)

    # Phylogenetic covariance. The BM form S = H - d/2 is only a valid covariance for an
    # ULTRAMETRIC tree; C_tree here is additive (neighbor-joining) with within-clade blocks
    # substituted, so it is indefinite. Eigenvalue clipping projects to the nearest PSD matrix
    # and keeps the BM interpretation. The exp(-d/s) variant is an OU-like alternative that is
    # PSD by construction for any tree metric (tree metrics are of negative type), and is run
    # as a robustness check in case the clipping distorts the answer.
    Ct = np.asarray(d["C_tree"], float)
    H = Ct.max() / 2.0
    # Solve via eigendecomposition rather than Cholesky: after clipping, the matrices are
    # near-singular and Cholesky fails, but a floored spectral inverse is stable.
    COVS = {"bm": H - Ct / 2.0, "ou": np.exp(-Ct / max(np.median(Ct), 1e-9))}
    facs = {}
    for nm, Vm in COVS.items():
        ev, U = np.linalg.eigh((Vm + Vm.T) / 2.0)
        ev = np.maximum(ev, 1e-4 * max(ev.max(), 1e-12))
        inv = lambda M, _U=U, _e=ev: _U @ ((_U.T @ M) / _e[:, None])
        one = np.ones((n, 1))
        facs[nm] = (inv, float((one.T @ inv(one)).ravel()[0]))

    G = np.asarray(d["X_st"]).shape[1]
    m = len(spot_ci)
    for av in GRID:
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        gn = np.asarray(g)
        if not np.isfinite(gn).all():
            continue
        w = gn / np.maximum(gn.sum(1, keepdims=True), 1e-30)
        x = w @ xy_st                                        # mapped positions, n x 2
        lls = {}
        for nm, (inv, den) in facs.items():
            mu = (np.ones((1, n)) @ inv(x)) / den              # GLS root state
            rr = x - mu
            s2 = float(np.trace(rr.T @ inv(rr))) / (n * 2)
            lls[nm] = -0.5 * n * 2 * (np.log(2 * np.pi * max(s2, 1e-300)) + 1)
        mse = float(S.deconvolution_loss_ls(Y, g, jnp.asarray(A), sig))
        ll_ex = -0.5 * m * G * (np.log(2 * np.pi * mse) + 1)
        rows.append(dict(seed=seed, alpha=av, ll_bm=lls["bm"], ll_ou=lls["ou"],
                         ll_expr=ll_ex, mse=mse,
                         acc=(spot_ci[gn.argmax(1)] == cidx).mean(), n=n, m=m, G=G))

r = pd.DataFrame(rows)
per = []
for s, sub in r.groupby("seed"):
    q = sub.set_index("alpha")
    per.append(dict(seed=s, a_bm=float(q.ll_bm.idxmax()), a_ex=float(q.ll_expr.idxmax()),
                    a_acc=float(q.acc.idxmax()), a_ou=float(q.ll_ou.idxmax()),
                    mono_bm=int((q.sort_index().ll_bm.diff().dropna() > 0).all())))
p = pd.DataFrame(per); N = len(p)
print(f"{'argmax EXPRESSION logL':<26}: {', '.join(f'{v:.2f}' for v in p.a_ex)}")
print(f"{'argmax BM logL':<26}: {', '.join(f'{v:.2f}' for v in p.a_bm)}")
print(f"{'argmax ACCURACY':<26}: {', '.join(f'{v:.2f}' for v in p.a_acc)}")
print(f"\nBM logL monotone INCREASING in alpha in {int(p.mono_bm.sum())}/{N} seeds "
      f"(the predicted mirror failure)")
print(f"BM alone picks the accuracy optimum in {int((p.a_bm == p.a_acc).sum())}/{N}")
print(f"{'argmax OU logL (PSD check)':<26}: {', '.join(f'{v:.2f}' for v in p.a_ou)}")

print("\n" + "=" * 74 + "\njoint logL = logL_expr + w * logL_BM")
nat = (r.m.iloc[0] * r.G.iloc[0]) / (r.n.iloc[0] * 2)
print(f"{'weight w':>12}{'argmax alpha per seed':>34}{'hits optimum':>15}")
print("-" * 74)
best_w = None
for w in [0.0, 1.0, nat, 10 * nat, 50 * nat, 100 * nat, 300 * nat, 1000 * nat, 1e9]:
    picks = []
    for s, sub in r.groupby("seed"):
        q = sub.set_index("alpha")
        picks.append(float((q.ll_expr + w * q.ll_bm).idxmax()))
    hits = sum(1 for pk, tr in zip(picks, p.a_acc) if pk == tr)
    lab = "expr only" if w == 0 else ("BM only" if w == 1e9 else f"{w:.3g}")
    print(f"{lab:>12}{', '.join(f'{v:.2f}' for v in picks):>34}{hits:>10}/{N}")
    if best_w is None or hits > best_w[1]:
        best_w = (w, hits)
print("-" * 74)
print(f"best weight hits the accuracy optimum in {best_w[1]}/{N} seeds")
print(f"(naive count-ratio weight w = {nat:.1f}; spot-gene values are correlated so this "
      f"over-counts expression)")
r.to_csv(HERE / "q20_bm_likelihood.csv", index=False)
