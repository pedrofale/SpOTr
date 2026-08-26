"""Q18: when is SpOTr needed at all, versus just using the anchor directly?

If the anchor is a barcode read in situ on both sides, clone identity is a LOOKUP: every cell
knows its clone, every spot knows its clones, and clone-level mapping is solved without any
tree. SpOTr's clone-level contribution is then redundant -- which is exactly the objection.
On eTRACER LE the answer key IS the barcode, so the reported ARI 0.311 -> 0.348 is measured on
a task the data already answers at 1.0.

The method is only needed where clone identity is NOISY EVIDENCE rather than a label -- CNVs
inferred per spot are weak, mixed across cells, and often unresolvable. So the question is
where the crossover sits. Four estimators, as anchor quality is degraded:

  expression only     OT on expression features, alpha = 0        (no anchor, no tree)
  anchor only         OT on anchor features, alpha = 0            (use the anchor directly)
  expression+anchor   OT on both, alpha = 0                       (the obvious baseline)
  SpOTr               OT on expression + tree, per-clade alpha    (anchor used only as a loss)

If "anchor only" wins at high anchor quality and SpOTr wins at low quality, that is a clean
statement of the regime the method is for.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")

EPS, T, J = 0.02, 50, 3
QUAL = [("clean", 0.0, 150), ("good", 1.0, 50), ("poor", 3.0, 20), ("very poor", 6.0, 8)]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        continue
    rg = np.random.default_rng(7000 + seed)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"]); n = len(cidx)
    A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
    X_sc, X_st = np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)
    cnv_c, cnv_s = np.asarray(d["cnv_clone"], float), np.asarray(d["cnv_spot"], float)
    nb_all = cnv_s.shape[1]

    def run(Cf_src_sc, Cf_src_st, alpha_vec, Y=None, sig=None, learn=False):
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(
            ad.AnnData(Cf_src_sc.astype(np.float32)), ad.AnnData(Cf_src_st.astype(np.float32)),
            d["C_tree"], d["C_space"])
        omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
        if learn:
            gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
            res = S.learn_alpha_gamma_perclade_alpha_data(
                Cf, Y, C_T, C_S, a, b, jnp.asarray(A), sig, omega,
                eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=300, lr=0.05,
                alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
            g = np.asarray(res["coupling"])
        else:
            sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
            g = g0
            for _ in range(J):
                C = build_perclade_alpha_cost_v2(jnp.asarray(alpha_vec.astype(np.float32)),
                                                 Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                                 equalize="measured")
                g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
            g = np.asarray(g)
        if not np.isfinite(g).all():
            return np.nan
        return (spot_ci[g.argmax(1)] == cidx).mean()

    expr_only = run(X_sc, X_st, np.zeros(K))
    for name, nz, nb in QUAL:
        keep = rg.choice(nb_all, size=nb, replace=False)
        cell_anchor = cnv_c[cidx][:, keep] + rg.normal(0, 0.25 + 0.15 * nz, size=(n, nb))
        spot_anchor = cnv_s[:, keep] + rg.normal(0, 0.15 * nz, size=(cnv_s.shape[0], nb))
        z = lambda M: (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)
        anchor_only = run(cell_anchor, spot_anchor, np.zeros(K))
        both = run(np.hstack([z(X_sc), z(cell_anchor)]), np.hstack([z(X_st), z(spot_anchor)]),
                   np.zeros(K))
        spotr = run(X_sc, X_st, None, learn=True,
                    Y=jnp.asarray(spot_anchor.astype(np.float32)),
                    sig=jnp.asarray((cnv_c[:, keep] * 2.0).astype(np.float32)))
        rows.append(dict(seed=seed, quality=name, expr=expr_only, anchor=anchor_only,
                         both=both, spotr=spotr))
        print(f"  seed {seed:>2} {name:>10}: expr {expr_only:.4f}  anchor {anchor_only:.4f}  "
              f"expr+anchor {both:.4f}  SpOTr {spotr:.4f}")

r = pd.DataFrame(rows)
print("\n" + "=" * 78)
print(f"{'anchor quality':>15}{'expression':>13}{'anchor only':>13}{'expr+anchor':>13}{'SpOTr':>10}{'best':>14}")
print("-" * 78)
for name, _, _ in QUAL:
    s = r[r.quality == name]
    vals = {"expression": s.expr.mean(), "anchor only": s.anchor.mean(),
            "expr+anchor": s.both.mean(), "SpOTr": s.spotr.mean()}
    best = max(vals, key=lambda k: vals[k])
    print(f"{name:>15}{vals['expression']:>13.4f}{vals['anchor only']:>13.4f}"
          f"{vals['expr+anchor']:>13.4f}{vals['SpOTr']:>10.4f}{best:>14}")
print("-" * 78)
for name, _, _ in QUAL:
    s = r[r.quality == name]
    print(f"{name:>15}: SpOTr beats anchor-only in {int((s.spotr > s.anchor).sum())}/{len(s)}, "
          f"beats expr+anchor in {int((s.spotr > s.both).sum())}/{len(s)}")
r.to_csv(HERE / "q18_when_needed.csv", index=False)
