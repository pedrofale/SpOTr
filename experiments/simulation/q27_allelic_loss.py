"""Q27: what loss for an allelic anchor (Numbat on cells, CalicoST on spots)?

deconvolution_loss_ls is a LINEAR mixture, spot = sum_k w_k * clone_k. That is right for total
copy number, which is additive across the cells in a spot. It is WRONG for B-allele frequency,
which is a ratio:

    BAF_spot = sum_k w_k * B_k  /  sum_k w_k * (A_k + B_k)

a ratio of weighted sums, not a weighted average of per-clone BAFs. And the observation is
allele COUNTS (b of n reads), which are binomial rather than Gaussian -- at the read depths a
Visium segment actually gets, that matters.

Three candidates, all fed identical data:
  ls_linear   least squares, predicting a weighted average of per-clone BAFs   (wrong link)
  ls_ratio    least squares, predicting the ratio of weighted sums            (right link)
  binomial    binomial negative log-likelihood on the counts, ratio link      (right link+noise)

Inputs a real pipeline supplies: per-clone allele-specific copy number (major A_k, minor B_k
per segment) from Numbat; per-spot phased B-allele and total counts from CalicoST's inputs.
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
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
DEPTHS = [20, 200]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]) or "spot_comp" not in d.files:
        continue
    rg = np.random.default_rng(31000 + seed)
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    comp = np.asarray(d["spot_comp"], float)
    K = len(d["clades"]); n = len(cidx); m = comp.shape[0]
    lab = (~is_disp).astype(int)
    A1 = np.zeros((n, K), np.float32); A1[np.arange(n), cidx] = 1.0

    # ---- allele-specific copy number per clone, imbalance correlated with the tree ----
    Tot = np.asarray(d["cnv_clone"], float)                      # total CN, tree-correlated
    z = (Tot - Tot.mean()) / max(Tot.std(), 1e-9)
    frac = np.clip(0.5 - 0.4 * np.abs(z), 0.03, 0.5)             # minor-allele fraction
    Bk = Tot * frac
    Ak = Tot - Bk
    G = Tot.shape[1]
    p_true = (comp @ Bk) / np.maximum(comp @ (Ak + Bk), 1e-9)

    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A1); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    couplings = {}
    for av in GRID:
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T,
                                             C_S, a, b, g, omega, sF, sL, equalize="measured")
            g, _u = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        couplings[av] = g
    accs = {av: (spot_ci[np.asarray(g).argmax(1)] == cidx).mean() for av, g in couplings.items()}
    a_acc = max(accs, key=lambda k: accs[k])

    for depth in DEPTHS:
        nn = np.full((m, G), depth, dtype=float)
        bb = rg.binomial(depth, np.clip(p_true, 1e-6, 1 - 1e-6)).astype(float)
        baf = bb / nn
        Akj, Bkj = jnp.asarray(Ak), jnp.asarray(Bk)
        bafk = jnp.asarray(Bk / np.maximum(Ak + Bk, 1e-9))
        bafj, bbj, nnj = jnp.asarray(baf), jnp.asarray(bb), jnp.asarray(nn)

        def props(gam, ca):
            p = jnp.einsum('ji,ik->jk', gam.T, ca)
            return p / jnp.maximum(jnp.sum(p, axis=1, keepdims=True), 1e-12)

        def ls_linear(Y, gam, ca, cc):
            return jnp.mean((bafj - props(gam, ca) @ bafk) ** 2)

        def ratio_p(gam, ca):
            w = props(gam, ca)
            return (w @ Bkj) / jnp.maximum(w @ (Akj + Bkj), 1e-9)

        def ls_ratio(Y, gam, ca, cc):
            return jnp.mean((bafj - ratio_p(gam, ca)) ** 2)

        def binom(Y, gam, ca, cc):
            p = jnp.clip(ratio_p(gam, ca), 1e-6, 1 - 1e-6)
            return -jnp.mean(bbj * jnp.log(p) + (nnj - bbj) * jnp.log1p(-p))

        for name, fn in [("ls_linear", ls_linear), ("ls_ratio", ls_ratio), ("binomial", binom)]:
            ls = {av: float(fn(None, g, jnp.asarray(A1), None)) for av, g in couplings.items()}
            a_loss = min(ls, key=lambda k: ls[k])
            mono = bool(np.all(np.diff([ls[v] for v in GRID]) > 0))
            S.deconvolution_loss_ls = fn
            try:
                res = S.learn_alpha_gamma_perclade_alpha_data(
                    Cf, jnp.asarray(baf.astype(np.float32)), C_T, C_S, a, b, jnp.asarray(A1),
                    jnp.asarray(bafk), omega, eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=300,
                    lr=0.05, alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
                al = np.asarray(res["alpha"])
                auc = roc_auc_score(lab, al) if np.isfinite(al).all() else np.nan
                ratio = (al[lab == 1].mean() / max(al[lab == 0].mean(), 1e-9)
                         if np.isfinite(al).all() else np.nan)
            finally:
                S.deconvolution_loss_ls = _ORIG
            rows.append(dict(seed=seed, depth=depth, loss=name, a_loss=a_loss, a_acc=a_acc,
                             monotone=int(mono), auc=auc, ratio=ratio))
    print(f"  seed {seed} done")

r = pd.DataFrame(rows)
print("\n" + "=" * 78)
for depth in DEPTHS:
    print(f"\nread depth {depth} per segment per spot")
    print(f"{'loss':>11}{'picks alpha>0':>15}{'monotone':>11}{'alpha AUC':>11}{'a_c/a_d':>10}")
    print("-" * 60)
    for nm in ["ls_linear", "ls_ratio", "binomial"]:
        s = r[(r.depth == depth) & (r.loss == nm)]
        print(f"{nm:>11}{str(int((s.a_loss > 0).sum())) + '/' + str(len(s)):>15}"
              f"{str(int(s.monotone.sum())) + '/' + str(len(s)):>11}"
              f"{s.auc.mean():>11.3f}{s.ratio.mean():>9.1f}x")
print(f"\naccuracy-optimal alpha across seeds: "
      f"{', '.join(f'{v:.2f}' for v in r[(r.loss=='binomial') & (r.depth==DEPTHS[0])].a_acc)}")
r.to_csv(HERE / "q27_allelic_loss.csv", index=False)
