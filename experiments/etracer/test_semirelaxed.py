"""Does semi-relaxed OT free the output composition from the reference composition?

The mismatch is imposed exactly through the source marginal: each cell's mass is set so the
clone shares of `a` reproduce the NATURAL scRNA-seq composition (which differs from the
tissue by TVD 0.538). Under balanced OT the output composition must then equal that wrong
composition. Semi-relaxed OT should let it move toward the truth.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.semirelaxed import sinkhorn_semirelaxed

EPS, T_SINK, J_ALT, ALPHA = 0.02, 100, 3, 0.075


def main():
    d = np.load(HERE / "etracer_inputs_LE.npz", allow_pickle=True)
    A, clones, truth, cell_clone = d["A"], d["clones"], d["bin_clone"], d["cell_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a_unif, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])

    # natural scRNA-seq clone composition for these clones
    lin = pd.read_csv(HERE / "sc_cells_LE_matched.csv")
    nat = lin[lin.lineageGrp.astype(str).isin([str(c) for c in clones])] \
            .lineageGrp.astype(str).value_counts(normalize=True)
    nat = nat.reindex([str(c) for c in clones]).fillna(0.0)
    nat = nat / nat.sum()
    true_frac = np.array([(truth == c).mean() for c in clones])

    # source marginal that reproduces the natural composition
    n_in = A.sum(0)
    per_cell = np.where(n_in > 0, nat.to_numpy() / np.maximum(n_in, 1), 0.0)
    a_nat = (A * per_cell[None, :]).sum(1)
    a_nat = jnp.asarray(a_nat / a_nat.sum())

    tvd = lambda p, q: 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()
    print(f"[setup] {A.shape[0]} cells x {len(truth)} bins, {len(clones)} clones")
    print(f"[setup] TVD(natural scRNA composition, true tissue composition) = "
          f"{tvd(nat.to_numpy(), true_frac):.3f}\n")

    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a_nat, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a_nat, b, gamma0, eps=EPS)

    def run(tau):
        g = gamma0
        uv = None
        for _ in range(J_ALT):
            C = S.build_fgw_cost(ALPHA, Cf, C_T, C_S, a_nat, b, g, sF, sL)
            if tau is None:                       # balanced reference
                g, uv = S.sinkhorn_unrolled_safe(C, a_nat, b, EPS, T_SINK, uv)
            else:
                g, uv = sinkhorn_semirelaxed(C, a_nat, b, EPS, T_SINK, tau=tau, uv0=uv)
        return np.asarray(g)

    print(f"{'solver':<22} {'TVD(out,truth)':>15} {'corr(out,ref)':>14} {'acc':>7} {'ARI':>8}")
    print("-" * 72)
    rows = []
    for label, tau in [("balanced", None), ("semi-relaxed tau=10", 10.0),
                       ("semi-relaxed tau=1", 1.0), ("semi-relaxed tau=0.1", 0.1),
                       ("semi-relaxed tau=0.01", 0.01), ("semi-relaxed tau=0.001", 0.001)]:
        g = run(tau)
        if not np.isfinite(g).all():
            print(f"{label:<22}   non-finite"); continue
        prop = g.T @ A
        out_frac = prop.sum(0) / prop.sum()
        pn = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        pred = clones[pn.argmax(1)]
        rows.append(dict(solver=label, tvd=tvd(out_frac, true_frac),
                         corr_ref=float(np.corrcoef(out_frac, nat.to_numpy())[0, 1]),
                         acc=accuracy_score(truth, pred), ari=adjusted_rand_score(truth, pred)))
        r = rows[-1]
        print(f"{label:<22} {r['tvd']:>15.3f} {r['corr_ref']:>14.4f} "
              f"{r['acc']:>7.3f} {r['ari']:>8.3f}")
    print("-" * 72)
    print(f"{'(reference composition)':<22} {tvd(nat.to_numpy(), true_frac):>15.3f}"
          f" {1.0:>14.4f}")
    print("\n  TVD(out,truth): 0 = output composition matches the tissue")
    print("  corr(out,ref):  1.000 = output composition IS the reference composition")
    pd.DataFrame(rows).to_csv(HERE / "semirelaxed_composition.csv", index=False)


if __name__ == "__main__":
    main()
