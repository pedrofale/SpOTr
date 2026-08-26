"""Q26: does an EXPRESSION-DERIVED anchor still work, or does it revert to the circular failure?

In cancer the tree normally comes from single-cell CNV inference, so the realistic pipeline is
Numbat on the scRNA-seq and CalicoST on the spatial data. But inferCNV -- and Numbat's
expression module -- infer copy number by SMOOTHING GENE EXPRESSION along the genome. If the
spot-side "CNV" is a transformation of spot expression, then "reconstruct spot CNV from clone
CNV signatures" is a smoothed version of "reconstruct spot expression from clone expression"
-- exactly the loss shown to be monotone with its argmin pinned at alpha = 0.

Numbat and CalicoST also use allele-specific (B-allele frequency) information, which is
independent of expression MAGNITUDE. So the practical question is which component carries the
alpha signal.

Three anchors, matched at 20 bins:
  independent   the simulated CNV channel (stands for the allelic component)
  derived       pseudo-CNV built by binning and smoothing expression along a pseudo-genome,
                exactly as inferCNV does
  mixed         half the bins from each, standing for a Numbat-style joint call
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

EPS, T, J, NB = 0.02, 50, 3, 20
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(21000 + seed)
    cidx, spot_ci, is_disp = d["cell_clade"], d["spot_clade"], d["is_dispersed"]
    K = len(d["clades"]); n = len(cidx); comp = (~is_disp).astype(int)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc, X_st = np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)
    cnv_c, cnv_s = np.asarray(d["cnv_clone"], float), np.asarray(d["cnv_spot"], float)

    # --- derived anchor: bin genes along a pseudo-genome and average, as inferCNV does ---
    order = rg.permutation(X_sc.shape[1])
    blocks = np.array_split(order, NB)
    def binned(M):
        Z = (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)
        return np.stack([Z[:, bl].mean(1) for bl in blocks], axis=1)
    der_cell, der_spot = binned(X_sc), binned(X_st)
    der_sig = np.stack([der_cell[cidx == k].mean(0) for k in range(K)])

    keep = rg.choice(cnv_s.shape[1], size=NB, replace=False)
    ind_spot = cnv_s[:, keep] + rg.normal(0, 0.15, size=(cnv_s.shape[0], NB))
    ind_sig = cnv_c[:, keep]

    h = NB // 2
    mix_spot = np.hstack([ind_spot[:, :h], der_spot[:, :h]])
    mix_sig = np.hstack([ind_sig[:, :h], der_sig[:, :h]])

    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc.astype(np.float32)),
                                             ad.AnnData(X_st.astype(np.float32)),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
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

    for name, Y, sig in [("independent", ind_spot, ind_sig), ("derived", der_spot, der_sig),
                         ("mixed", mix_spot, mix_sig)]:
        Yj = jnp.asarray(Y.astype(np.float32)); sj = jnp.asarray((sig * 2.0).astype(np.float32))
        ls = {av: float(S.deconvolution_loss_ls(Yj, g, jnp.asarray(A), sj))
              for av, g in couplings.items()}
        a_loss = min(ls, key=lambda k: ls[k])
        mono = all(np.diff([ls[v] for v in GRID]) > 0)
        res = S.learn_alpha_gamma_perclade_alpha_data(
            Cf, Yj, C_T, C_S, a, b, jnp.asarray(A), sj, omega, eps=EPS, T_sinkhorn=50, J_alt=3,
            K_outer=300, lr=0.05, alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
        al = np.asarray(res["alpha"]); ok = np.isfinite(al).all()
        rows.append(dict(seed=seed, anchor=name, a_loss=a_loss, a_acc=a_acc, monotone=int(mono),
                         auc=roc_auc_score(comp, al) if ok else np.nan,
                         a_c=al[comp == 1].mean() if ok else np.nan,
                         a_d=al[comp == 0].mean() if ok else np.nan))
    print(f"  seed {seed:>2}: " + "  ".join(
        f"{r['anchor']} argmin a={r['a_loss']:.2f} AUC {r['auc']:.2f}" for r in rows[-3:]))

r = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(f"{'anchor':>13}{'argmin(loss) alpha':>20}{'monotone in alpha':>20}{'alpha AUC':>11}{'a_c/a_d':>12}")
print("-" * 78)
for nm in ["independent", "derived", "mixed"]:
    s = r[r.anchor == nm]
    picks = ", ".join(f"{v:.2f}" for v in s.a_loss)
    print(f"{nm:>13}{picks:>20}{int(s.monotone.sum())}/{len(s):<18}{s.auc.mean():>11.3f}"
          f"{s.a_c.mean() / max(s.a_d.mean(), 1e-9):>11.1f}x")
print("-" * 78)
print(f"accuracy-optimal alpha: {', '.join(f'{v:.2f}' for v in r[r.anchor=='independent'].a_acc)}")
for nm in ["independent", "derived", "mixed"]:
    s = r[r.anchor == nm]
    print(f"{nm:>13}: picks alpha>0 in {int((s.a_loss > 0).sum())}/{len(s)}, "
          f"AUC above 0.7 in {int((s.auc > 0.7).sum())}/{len(s)}")
r.to_csv(HERE / "q26_derived_anchor.csv", index=False)
