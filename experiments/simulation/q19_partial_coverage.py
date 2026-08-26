"""Q19: does the loss route beat the feature route when the anchor covers only some spots?

Q18 showed that whenever the anchor is usable, putting it in the FEATURE cost beats using it
as a loss, roughly two to one. That kills the "CNV anchor" proposal in its general form and
generalises the barcode objection: an anchor good enough to fix the objective is good enough
to map with directly.

But real anchors are PARTIAL. eTRACER late-escape carries a clone barcode on ~7,140 of 32,023
bins (~22%); Numbat/CalicoST calls are confident on some spots and not others. A feature has
to be present for every spot to enter the cost, so unlabelled spots contribute an
uninformative row. A loss does not: it can be evaluated on the labelled subset alone while the
learned alpha applies everywhere.

Accuracy is scored over ALL spots, including unlabelled ones, since that is what the mapping
has to cover.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")
_orig_loss = S.deconvolution_loss_ls

EPS, T, J = 0.02, 50, 3
COV = [1.0, 0.5, 0.22, 0.10, 0.03]
NOISE, NBINS = 1.0, 50            # "good" quality from Q18, where the feature route won
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(8000 + seed)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"]); n = len(cidx); m = len(spot_ci)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc, X_st = np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)
    cnv_c, cnv_s = np.asarray(d["cnv_clone"], float), np.asarray(d["cnv_spot"], float)
    keep = rg.choice(cnv_s.shape[1], size=NBINS, replace=False)
    cell_anchor = cnv_c[cidx][:, keep] + rg.normal(0, 0.25 + 0.15 * NOISE, size=(n, NBINS))
    spot_anchor_full = cnv_s[:, keep] + rg.normal(0, 0.15 * NOISE, size=(m, NBINS))
    sig = jnp.asarray((cnv_c[:, keep] * 2.0).astype(np.float32))
    z = lambda M: (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)

    for cov in COV:
        obs = np.zeros(m, dtype=bool)
        obs[rg.choice(m, size=max(2, int(round(cov * m))), replace=False)] = True

        # --- feature route: unlabelled spots get the mean anchor profile (uninformative) ---
        sp_feat = spot_anchor_full.copy()
        sp_feat[~obs] = spot_anchor_full[obs].mean(0)
        Xsc_f = np.hstack([z(X_sc), z(cell_anchor)])
        Xst_f = np.hstack([z(X_st), z(sp_feat)])
        Cf2, C_T2, C_S2, a2, b2 = S.prepare_ot_inputs(
            ad.AnnData(Xsc_f.astype(np.float32)), ad.AnnData(Xst_f.astype(np.float32)),
            d["C_tree"], d["C_space"])
        g = a2[:, None] * b2[None, :]
        sF2, sL2 = S.make_cladexfgw_scales(Cf2, C_T2, C_S2, a2, b2, g, eps=EPS)
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cf2, C_T2, C_S2,
                                             a2, b2, g, jnp.asarray(A), sF2, sL2,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a2, b2, EPS, T)
        gf = np.asarray(g)
        acc_feat = (spot_ci[gf.argmax(1)] == cidx).mean() if np.isfinite(gf).all() else np.nan

        # --- loss route: alpha learned from the LABELLED spots only ---
        mask = jnp.asarray(obs.astype(np.float32))[:, None]
        def masked_loss(Y, gam, ca, cc, _m=mask):
            p = jnp.einsum('ji,ik->jk', gam.T, ca)
            p = p / jnp.maximum(jnp.sum(p, axis=1, keepdims=True), 1e-12)
            sm = jnp.einsum('jk,kg->jg', p, cc / 2.0)
            return jnp.sum(((Y - sm) ** 2) * _m) / jnp.maximum(jnp.sum(_m) * Y.shape[1], 1.0)
        S.deconvolution_loss_ls = masked_loss
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc.astype(np.float32)),
                                                 ad.AnnData(X_st.astype(np.float32)),
                                                 d["C_tree"], d["C_space"])
        gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
        res = S.learn_alpha_gamma_perclade_alpha_data(
            Cf, jnp.asarray(spot_anchor_full.astype(np.float32)), C_T, C_S, a, b,
            jnp.asarray(A), sig, jnp.asarray(A), eps=EPS, T_sinkhorn=50, J_alt=3,
            K_outer=300, lr=0.05, alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
        S.deconvolution_loss_ls = _orig_loss
        gl = np.asarray(res["coupling"])
        acc_loss = (spot_ci[gl.argmax(1)] == cidx).mean() if np.isfinite(gl).all() else np.nan
        rows.append(dict(seed=seed, cov=cov, feature=acc_feat, loss=acc_loss))
        print(f"  seed {seed:>2} cov {cov:>5.2f}: feature route {acc_feat:.4f}   loss route {acc_loss:.4f}")

r = pd.DataFrame(rows)
print("\n" + "=" * 66)
print(f"{'anchor coverage':>16}{'feature route':>16}{'loss route (SpOTr)':>21}{'winner':>12}")
print("-" * 66)
for cov in COV:
    s = r[r["cov"] == cov]
    w = "SpOTr" if s.loss.mean() > s.feature.mean() else "feature"
    print(f"{cov * 100:>14.0f}%{s.feature.mean():>16.4f}{s.loss.mean():>21.4f}{w:>12}")
print("-" * 66)
for cov in COV:
    s = r[r["cov"] == cov]
    print(f"{cov*100:>14.0f}%: SpOTr wins in {int((s.loss > s.feature).sum())}/{len(s)} seeds")
r.to_csv(HERE / "q19_partial_coverage.csv", index=False)
