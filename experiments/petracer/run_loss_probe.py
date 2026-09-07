"""Can any loss computable WITHOUT ground truth tell a good alpha from a bad one?

Every iterative scheme -- the paper's eq:data_objective included -- needs this. If the loss it
optimises does not track mapping quality, alternating between the OT solve and an alpha update
just optimises noise. So: sweep alpha, and for each solve compute both the truth (placement
error, unavailable in practice) and several self-supervised losses that ARE available.

  deconvolution   Poisson NLL of the observed spot expression under the mapped cell mixture
                  (this is the paper's objective)
  heldout genes   fit the map on half the genes, score how well the mapped cells predict the
                  other half's spatial pattern
  spot coherence  do cells sent to the same spot agree in expression?
  fgw objective   the transport cost itself (expected to be useless -- monotone in alpha)
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np, pandas as pd, ot
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent

GRID = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9]
d = np.load(HERE / "petracer_M3-4.npz", allow_pickle=True)
X_sc = np.asarray(d["X_sc"], float); X_st = np.asarray(d["X_st"], float)
CT = np.asarray(d["C_tree"], float); CS = np.asarray(d["C_space"], float)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
n, m = X_sc.shape[0], X_st.shape[0]
G = X_sc.shape[1]
rng = np.random.default_rng(0)
half = rng.permutation(G); gA, gB = half[:G//2], half[G//2:]
C1, C2 = CT/CT.mean(), CS/CS.mean()
p, q = np.ones(n)/n, np.ones(m)/m
M_all = ot.dist(X_sc, X_st); M_all = M_all / M_all.max()
M_A = ot.dist(X_sc[:, gA], X_st[:, gA]); M_A = M_A / M_A.max()

Yobs = np.clip(X_st - X_st.min(), 1e-6, None)          # non-negative stand-in for counts
rows = []
for a in GRID:
    T = np.asarray(ot.gromov.fused_gromov_wasserstein(
        M_all, C1, C2, p, q, loss_fun="square_loss", alpha=a, max_iter=200,
        tol_rel=1e-9, tol_abs=1e-9))
    am = T.argmax(1)
    err = float(np.median(np.linalg.norm(xy_st[am] - xy_sc, axis=1)))
    # deconvolution: predicted spot profile = mass-weighted mean of the cells sent there
    W = T / np.maximum(T.sum(0, keepdims=True), 1e-300)
    pred = W.T @ X_sc
    lam = np.clip(pred - pred.min() + 1e-6, 1e-6, None)
    lam = lam * (Yobs.sum() / lam.sum())
    deconv = float(np.mean(lam - Yobs * np.log(lam)))
    # held-out genes: map with half the genes, predict the other half
    T_A = np.asarray(ot.gromov.fused_gromov_wasserstein(
        M_A, C1, C2, p, q, loss_fun="square_loss", alpha=a, max_iter=200,
        tol_rel=1e-9, tol_abs=1e-9))
    W_A = T_A / np.maximum(T_A.sum(0, keepdims=True), 1e-300)
    predB = W_A.T @ X_sc[:, gB]
    heldout = float(np.mean((predB - X_st[:, gB]) ** 2))
    # spot coherence: expression agreement among cells sent to the same spot
    coh = float(np.mean([X_sc[am == j].std(0).mean() for j in range(m) if (am == j).sum() > 2]))
    rows.append(dict(alpha=a, err=err, deconv=deconv, heldout=heldout, coherence=coh))
    print(f"  alpha {a:.2f}  err {err:6.1f}  deconv {deconv:9.4f}  heldout {heldout:9.4f}"
          f"  coherence {coh:.4f}", flush=True)

R = pd.DataFrame(rows); R.to_csv(HERE / "loss_probe.csv", index=False)
print(f"\nDoes each loss track the truth?  (rho vs placement error; +1 = perfect guide)")
for c in ["deconv", "heldout", "coherence"]:
    r = spearmanr(R[c], R.err)
    pick = R.alpha[R[c].idxmin()]
    print(f"  {c:<12} rho {r.statistic:+.3f}  p {r.pvalue:.3f}   "
          f"picks alpha={pick:g}   (truth picks {R.alpha[R.err.idxmin()]:g})")
