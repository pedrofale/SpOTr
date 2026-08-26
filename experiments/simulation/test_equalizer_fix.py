"""A/B the original equalizer against the v2 fix, on simulated data with known lambda.

Uses the SINGLE-CLADE perturbation design -- the one that is conceptually right (move one
clade's phi, hold the rest) but that the original equalizer made unreadable. Two things to
check:

  1. artifact: how often does a clade's best phi land exactly on the baseline? Under the
     original equalizer this was 15/23, because deviating from a uniform vector is itself
     penalised. It should largely disappear under "measured".
  2. recovery: does preferred phi correlate with the KNOWN per-clade lambda?
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE_ALPHA, BASE_PHI = 0.10, 0.5
EPS, T_SINK, J_ALT, MIN_SPOTS = 0.02, 50, 3, 20


def main():
    d = np.load(HERE / "sim_inputs.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    lam = d["true_lambda"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    ak = jnp.asarray(np.full(K, BASE_ALPHA, dtype=np.float32))
    n_spots = np.array([(truth == c).sum() for c in clones])
    print(f"[test] {A.shape[0]} cells x {len(truth)} spots, {K} clades, "
          f"{(n_spots>=MIN_SPOTS).sum()} with >= {MIN_SPOTS} spots")

    def couple(pk, mode):
        g = gamma0
        for _ in range(J_ALT):
            C = build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             use_decomp=True, Omega=Omega, phi_k=pk,
                                             equalize=mode)
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
        return np.asarray(g)

    def clade_score(g, k, normalise):
        """normalise=True is the per-spot share (couples all clades); False is clade k's
        RAW transported mass, which removes one of the two coupling paths."""
        prop = g.T @ A
        if normalise:
            prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        t = (truth == clones[k]).astype(float)
        if t.std() == 0 or prop[:, k].std() == 0:
            return np.nan
        return float(np.corrcoef(prop[:, k], t)[0, 1])

    out = {}
    for mode, norm in [("measured", True), ("measured", False)]:
        rows = []
        for k in range(K):
            curve = []
            for v in P_GRID:
                pk = jnp.asarray(np.full(K, BASE_PHI, dtype=np.float32)).at[k].set(v)
                g = couple(pk, mode)
                curve.append(clade_score(g, k, norm) if np.isfinite(g).all() else np.nan)
            best = P_GRID[int(np.nanargmax(curve))] if np.isfinite(curve).any() else np.nan
            rows.append(dict(clade=str(clones[k]), n_spots=int(n_spots[k]),
                             true_lambda=float(lam[k]), pref_phi=best))
        df = pd.DataFrame(rows)
        sub = df[(df.n_spots >= MIN_SPOTS) & df.pref_phi.notna()]
        ties = int((np.abs(sub.pref_phi - BASE_PHI) < 1e-9).sum())
        pr, pp = stats.pearsonr(sub.pref_phi, sub.true_lambda)
        sr, sp = stats.spearmanr(sub.pref_phi, sub.true_lambda)
        hi = sub[sub.true_lambda >= 0.5].pref_phi.mean()
        lo = sub[sub.true_lambda < 0.5].pref_phi.mean()
        out[(mode, norm)] = dict(n=len(sub), ties=ties, pearson=pr, p_pearson=pp,
                         spearman=sr, p_spearman=sp, hi=hi, lo=lo)
        df.to_csv(HERE / f"equalizer_{mode}_norm{int(norm)}.csv", index=False)

    print(f"\n{'equalizer':<12} {'n':>4} {'best==baseline':>15} {'Pearson':>9} {'p':>7} "
          f"{'Spearman':>10} {'p':>7} {'phi hi/lo':>14}")
    print("-" * 84)
    for key in [("measured", True), ("measured", False)]:
        r = out[key]
        tag = "v2 + share" if key[1] else "v2 + RAW mass"
        print(f"{tag:<12} {r['n']:>4} {r['ties']:>10}/{r['n']:<4} {r['pearson']:>+9.3f} "
              f"{r['p_pearson']:>7.3f} {r['spearman']:>+10.3f} {r['p_spearman']:>7.3f} "
              f"{r['hi']:>6.3f}/{r['lo']:<6.3f}")
    print("-" * 84)
    print("  'best==baseline' is the artifact: a clade scoring best at exactly the baseline")
    print("  because deviating from a uniform vector is penalised in itself.")


if __name__ == "__main__":
    main()
