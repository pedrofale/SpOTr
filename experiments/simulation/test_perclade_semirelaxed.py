"""Does semi-relaxed OT make per-clade phi readable? (claim b)

Balanced OT pins every column of gamma to b_j, so perturbing one clade necessarily moves
all of them and the unperturbed configuration wins by default. Relaxing the SOURCE marginal
lets a clade's total mass change when its own parameter changes, which should make
single-clade perturbation local -- and therefore interpretable.

Ground truth is the known per-clade lambda from the simulation.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import sinkhorn_semirelaxed

P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
BASE_ALPHA, BASE_PHI = 0.10, 0.5
EPS, T_SINK, J_ALT, MIN_SPOTS = 0.02, 50, 3, 20


def main():
    d = np.load(HERE / "sim_inputs.npz", allow_pickle=True)
    A, clones, truth, lam = d["A"], d["clones"], d["bin_clone"], d["true_lambda"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    ak = jnp.asarray(np.full(K, BASE_ALPHA, dtype=np.float32))
    n_spots = np.array([(truth == c).sum() for c in clones])
    print(f"[test] {A.shape[0]} cells x {len(truth)} spots, {K} clades, "
          f"{(n_spots >= MIN_SPOTS).sum()} with >= {MIN_SPOTS} spots")

    def couple(pk, tau):
        g, uv = gamma0, None
        for _ in range(J_ALT):
            C = build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             use_decomp=True, Omega=Omega, phi_k=pk,
                                             equalize="measured")
            if tau is None:
                g, uv = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK, uv)
            else:
                g, uv = sinkhorn_semirelaxed(C, a, b, EPS, T_SINK, tau=tau, uv0=uv)
        return np.asarray(g)

    def score(g, k):
        prop = g.T @ A
        t = (truth == clones[k]).astype(float)
        if t.std() == 0 or prop[:, k].std() == 0:
            return np.nan
        return float(np.corrcoef(prop[:, k], t)[0, 1])

    print(f"\n{'solver':<24} {'n':>4} {'best==base':>11} {'Pearson':>9} {'p':>7} "
          f"{'Spearman':>10} {'p':>7} {'phi hi/lo':>13}")
    print("-" * 92)
    for label, tau in [("balanced", None), ("semi-relaxed tau=0.1", 0.1),
                       ("semi-relaxed tau=0.01", 0.01)]:
        rows = []
        for k in range(K):
            curve = []
            for v in P_GRID:
                pk = jnp.asarray(np.full(K, BASE_PHI, dtype=np.float32)).at[k].set(v)
                g = couple(pk, tau)
                curve.append(score(g, k) if np.isfinite(g).all() else np.nan)
            best = P_GRID[int(np.nanargmax(curve))] if np.isfinite(curve).any() else np.nan
            rows.append(dict(clade=str(clones[k]), n_spots=int(n_spots[k]),
                             true_lambda=float(lam[k]), pref_phi=best))
        df = pd.DataFrame(rows)
        df.to_csv(HERE / f"perclade_sr_{str(tau).replace('.','p')}.csv", index=False)
        sub = df[(df.n_spots >= MIN_SPOTS) & df.pref_phi.notna()]
        ties = int((np.abs(sub.pref_phi - BASE_PHI) < 1e-9).sum())
        pr, pp = stats.pearsonr(sub.pref_phi, sub.true_lambda)
        sr, sp = stats.spearmanr(sub.pref_phi, sub.true_lambda)
        hi = sub[sub.true_lambda >= 0.5].pref_phi.mean()
        lo = sub[sub.true_lambda < 0.5].pref_phi.mean()
        print(f"{label:<24} {len(sub):>4} {ties:>7}/{len(sub):<3} {pr:>+9.3f} {pp:>7.3f} "
              f"{sr:>+10.3f} {sp:>7.3f} {hi:>5.3f}/{lo:<5.3f}")
    print("-" * 92)


if __name__ == "__main__":
    main()
