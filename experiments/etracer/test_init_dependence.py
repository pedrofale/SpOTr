"""Does the eTRACER tree benefit survive a better initialisation?

If the GW term's value is largely about escaping a poor cold start, then warm-starting from
the expression-only solution should shrink it. If the tree carries information expression
cannot supply, the gain should persist regardless of initialisation.

Shuffled-tree controls throughout, so "gain" always means gain over a structure-free tree of
the same distance distribution -- not merely over alpha = 0.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

EPS, T, J = 0.02, 100, 5
ALPHAS = [0.0, 0.05, 0.075, 0.15, 0.30]
rng = np.random.default_rng(0)


def main():
    d = np.load(HERE / "etracer_inputs_LE.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    print(f"[init] {A.shape[0]} cells x {len(truth)} bins, {len(clones)} clones")

    cold = a[:, None] * b[None, :]
    warm, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=T, J_alt=J, alpha=0.0)

    def solve(C_tree_mat, alpha, g0):
        g, uv = g0, None
        F_c = S.dc(Cf, a, b)
        L_c = S.dc(S.compute_Lgw(C_tree_mat, C_S, a, b, g0), a, b)
        sF, sL = S.mad_abs(F_c), S.mad_abs(L_c)
        for _ in range(J):
            C = S.build_fgw_cost(alpha, Cf, C_tree_mat, C_S, a, b, g, sF, sL)
            g, uv = S.sinkhorn_unrolled_safe(C, a, b, EPS, T, uv)
        return np.asarray(g)

    def ari(g):
        if not np.isfinite(g).all():
            return np.nan
        p = g.T @ A
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        return adjusted_rand_score(truth, clones[p.argmax(1)])

    n = A.shape[0]
    shuf = [d["C_tree"][np.ix_(pi, pi)] for pi in (rng.permutation(n) for _ in range(2))]

    print(f"\n{'init':<8} {'alpha':>7} {'real tree':>11} {'shuffled':>10} {'real-shuf':>11}")
    print("-" * 52)
    rows = []
    for iname, g0 in [("cold", cold), ("warm", warm)]:
        base = None
        for av in ALPHAS:
            r = ari(solve(d["C_tree"], av, g0))
            sv = float(np.nanmean([ari(solve(sm, av, g0)) for sm in shuf])) if av > 0 else np.nan
            if av == 0.0:
                base = r
            mark = ""
            print(f"{iname:<8} {av:>7.3f} {r:>11.4f} "
                  f"{'--' if av == 0 else format(sv, '.4f'):>10} "
                  f"{'--' if av == 0 else format(r - sv, '+.4f'):>11}{mark}")
            rows.append(dict(init=iname, alpha=av, real=r, shuffled=sv, base=base))
        sub = [x for x in rows if x["init"] == iname and x["alpha"] > 0]
        bi = max(sub, key=lambda x: x["real"])
        print(f"{'':8} -> best alpha {bi['alpha']:.3f}: real {bi['real']:.4f}, "
              f"alpha=0 {base:.4f}, gain over alpha=0 {bi['real']-base:+.4f}, "
              f"over shuffled {bi['real']-bi['shuffled']:+.4f}\n")
    pd.DataFrame(rows).to_csv(HERE / "init_dependence.csv", index=False)


if __name__ == "__main__":
    main()
