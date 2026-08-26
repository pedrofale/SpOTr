"""Claim (b), reformulated: measure per-clade concordance from INFERRED positions.

Instead of asking a fitted mixing parameter to report concordance, use the mapping to give
each dissociated cell an inferred position, then measure per-clade tree-vs-space concordance
directly -- the same statistic used on real spots.

Non-circularity: the mapping is done with alpha = 0, so the tree is NOT used to place cells.
Any tree-vs-inferred-space agreement therefore comes from the data, not from the prior.
As a control we also report alpha > 0, where the tree IS used and concordance would be
partly manufactured.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial.distance import pdist, squareform
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.semirelaxed import sinkhorn_semirelaxed

EPS, T_SINK, J_ALT, MIN_CELLS = 0.02, 100, 3, 20


def main():
    d = np.load(HERE / "sim_inputs.npz", allow_pickle=True)
    A, clones, cell_clone = d["A"], d["clones"], d["cell_clone"]
    lam, meas, xy = d["true_lambda"], d["measured_r"], d["xy"]
    C_tree = d["C_tree"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, C_tree, d["C_space"])
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    print(f"[test] {A.shape[0]} cells x {len(xy)} spots, {len(clones)} clades")

    def run(alpha, tau=None):
        g, uv = gamma0, None
        for _ in range(J_ALT):
            C = S.build_fgw_cost(alpha, Cf, C_T, C_S, a, b, g, sF, sL)
            if tau is None:
                g, uv = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK, uv)
            else:
                g, uv = sinkhorn_semirelaxed(C, a, b, EPS, T_SINK, tau=tau, uv0=uv)
        return np.asarray(g)

    def perclade_inferred(g):
        """Infer each cell's position, then measure per-clade tree-vs-space concordance."""
        w = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
        pos = w @ xy                                   # transport-weighted mean location
        out = []
        for k, c in enumerate(clones):
            ii = np.where(cell_clone == c)[0]
            if len(ii) < MIN_CELLS:
                out.append(np.nan); continue
            dt = squareform(C_tree[np.ix_(ii, ii)], checks=False)
            ds = pdist(pos[ii])
            out.append(np.nan if (dt.std() == 0 or ds.std() == 0)
                       else float(np.corrcoef(dt, ds)[0, 1]))
        return np.array(out)

    print(f"\n{'mapping':<28} {'n':>4} {'vs TRUE lambda':>16} {'p':>8} "
          f"{'vs measured r':>15} {'p':>8}")
    print("-" * 86)
    rows = []
    for label, alpha, tau in [("alpha=0 (expression only)", 0.0, None),
                              ("alpha=0, semi-relaxed", 0.0, 0.01),
                              ("alpha=0.075 (tree used)", 0.075, None),
                              ("alpha=0.30 (tree used)", 0.30, None)]:
        g = run(alpha, tau)
        if not np.isfinite(g).all():
            print(f"{label:<28}  non-finite"); continue
        inf_r = perclade_inferred(g)
        ok = np.isfinite(inf_r)
        pr, pp = stats.pearsonr(inf_r[ok], lam[ok])
        pr2, pp2 = stats.pearsonr(inf_r[ok], meas[ok])
        print(f"{label:<28} {ok.sum():>4} {pr:>+16.3f} {pp:>8.4f} {pr2:>+15.3f} {pp2:>8.4f}")
        rows.append(dict(mapping=label, n=int(ok.sum()), r_vs_lambda=pr, p_lambda=pp,
                         r_vs_measured=pr2, p_measured=pp2))
        if alpha == 0.0 and tau is None:
            best = pd.DataFrame(dict(clade=clones[ok], true_lambda=lam[ok],
                                     inferred_concordance=inf_r[ok])).sort_values("true_lambda")
            best.to_csv(HERE / "inferred_concordance.csv", index=False)
    print("-" * 86)
    print("  alpha=0 rows are the honest test: the tree is not used to place cells.")
    print("  alpha>0 rows are the circularity control -- there the tree helps place cells,")
    print("  so agreement is partly manufactured and should read HIGHER.")
    pd.DataFrame(rows).to_csv(HERE / "inferred_concordance_summary.csv", index=False)


if __name__ == "__main__":
    main()
