"""Q28: does the ALPHA READOUT survive sparse anchor coverage?

Q19 measured mapping accuracy against anchor coverage and found the loss route wins below
~10%. But for a dataset where the point is to LEARN alpha rather than map better -- the embryo
case -- the quantity that matters is whether alpha_k still recovers which clades are spatially
coherent when the anchor covers only a few percent of spots.

Q24 degraded anchor QUALITY (readout survived to AUC 0.79). This degrades COVERAGE, which is
the axis a sparse in-situ barcode readout actually sits on. The loss is masked to the labelled
spots; the learned alpha applies everywhere.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")
_ORIG = S.deconvolution_loss_ls

EPS, T, J = 0.02, 50, 3
COV = [1.0, 0.33, 0.22, 0.10, 0.05, 0.028, 0.01]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(41000 + seed)
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"]); n = len(cidx); m = len(spot_ci)
    lab = (~is_disp).astype(int)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    cnv_c, cnv_s = np.asarray(d["cnv_clone"], float), np.asarray(d["cnv_spot"], float)
    keep = rg.choice(cnv_s.shape[1], size=50, replace=False)
    Y = cnv_s[:, keep] + rg.normal(0, 0.15, size=(m, 50))
    sig = jnp.asarray((cnv_c[:, keep] * 2.0).astype(np.float32))
    Yj = jnp.asarray(Y.astype(np.float32))

    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A)
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)

    for cov in COV:
        k = max(2, int(round(cov * m)))
        obs = np.zeros(m, bool); obs[rg.choice(m, size=k, replace=False)] = True
        mask = jnp.asarray(obs.astype(np.float32))[:, None]

        def masked(Yv, gam, ca, cc, _m=mask):
            p = jnp.einsum('ji,ik->jk', gam.T, ca)
            p = p / jnp.maximum(jnp.sum(p, axis=1, keepdims=True), 1e-12)
            sm = jnp.einsum('jk,kg->jg', p, cc / 2.0)
            return jnp.sum(((Yv - sm) ** 2) * _m) / jnp.maximum(jnp.sum(_m) * Yv.shape[1], 1.0)

        S.deconvolution_loss_ls = masked
        try:
            res = S.learn_alpha_gamma_perclade_alpha_data(
                Cf, Yj, C_T, C_S, a, b, jnp.asarray(A), sig, omega, eps=EPS, T_sinkhorn=50,
                J_alt=3, K_outer=300, lr=0.05, alpha_init=jnp.full((K,), 0.10),
                gamma0=gi, use_loss_ls=True)
            al = np.asarray(res["alpha"])
        finally:
            S.deconvolution_loss_ls = _ORIG
        ok = np.isfinite(al).all()
        rows.append(dict(seed=seed, cov=cov, n_spots=k,
                         auc=roc_auc_score(lab, al) if ok else np.nan,
                         ratio=(al[lab == 1].mean() / max(al[lab == 0].mean(), 1e-9)) if ok else np.nan))
    print(f"  seed {seed}: " + "  ".join(f"{int(x['cov']*100)}%:{x['auc']:.2f}" for x in rows[-len(COV):]))

r = pd.DataFrame(rows)
print("\n" + "=" * 64)
print(f"{'coverage':>10}{'labelled spots':>16}{'alpha AUC':>12}{'sd':>8}{'compact/disp':>14}")
print("-" * 64)
for c in COV:
    s = r[r["cov"] == c]
    print(f"{c*100:>9.1f}%{s.n_spots.mean():>16.0f}{s.auc.mean():>12.3f}{s.auc.std():>8.3f}"
          f"{s.ratio.mean():>13.1f}x")
print("-" * 64)
good = [c for c in COV if r[r["cov"] == c].auc.mean() > 0.75]
print(f"\nreadout holds (AUC > 0.75) down to {min(good)*100:.1f}% coverage"
      if good else "\nreadout does not hold at any tested coverage")
print(f"eTRACER stages for reference: baseline 2.8%, late escape ~22%, early escape ~33%")
r.to_csv(HERE / "q28_coverage_readout.csv", index=False)
