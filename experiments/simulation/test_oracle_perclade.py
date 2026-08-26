"""Is there any HEADROOM in per-clade weighting? (the prior question to interpretability)

Every earlier test asked whether a fitted per-clade weight correlates with concordance.
That is the bonus question. The primary claim is that clade-specific weights should MAP
BETTER than one global weight, because some clades carry more spatial structure than others.

Test it with an oracle: set the per-clade weights from the KNOWN lambda and compare mapping
quality against the best global weight. Controls: random per-clade weights (same
distribution, wrong assignment) and inverted weights.

  oracle >> global  -> the idea is sound and only the LEARNING is broken
  oracle ~= global  -> per-clade weighting has no headroom on this data
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T_SINK, J_ALT = 0.02, 50, 3
rng = np.random.default_rng(0)


def main():
    d = np.load(HERE / "sim_inputs.npz", allow_pickle=True)
    A, clones, truth, lam = d["A"], d["clones"], d["bin_clone"], d["true_lambda"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    print(f"[oracle] {A.shape[0]} cells x {len(truth)} spots, {K} clades")
    print(f"[oracle] lambda: {dict(zip(*np.unique(lam, return_counts=True)))}")

    def solve(ak, pk=None):
        g = gamma0
        for _ in range(J_ALT):
            C = build_perclade_alpha_cost_v2(
                jnp.asarray(ak.astype(np.float32)), Cf, C_T, C_S, a, b, g, omega, sF, sL,
                use_decomp=pk is not None, Omega=Omega if pk is not None else None,
                phi_k=None if pk is None else jnp.asarray(pk.astype(np.float32)),
                equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
        return np.asarray(g)

    def score(g):
        if not np.isfinite(g).all():
            return np.nan, np.nan
        prop = g.T @ A
        prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        pred = clones[prop.argmax(1)]
        return accuracy_score(truth, pred), adjusted_rand_score(truth, pred)

    # ---- best GLOBAL alpha ----
    print(f"\n{'setting':<44} {'acc':>7} {'ARI':>8}")
    print("-" * 62)
    best_g = (-1, None)
    for av in [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]:
        acc, ari = score(solve(np.full(K, av)))
        print(f"{'global alpha=' + format(av, '.2f'):<44} {acc:>7.3f} {ari:>8.4f}")
        if ari > best_g[0]:
            best_g = (ari, av)
    print(f"   -> best global alpha = {best_g[1]} (ARI {best_g[0]:.4f})")

    # ---- ORACLE per-clade alpha, scaled from true lambda ----
    print()
    lam_n = (lam - lam.min()) / max(lam.max() - lam.min(), 1e-9)
    best_o = (-1, None)
    for lo, hi in [(0.0, 0.20), (0.0, 0.35), (0.05, 0.35), (0.0, 0.50), (0.05, 0.20)]:
        ak = lo + (hi - lo) * lam_n
        acc, ari = score(solve(ak))
        print(f"{'ORACLE alpha in [' + f'{lo},{hi}' + '] from true lambda':<44} {acc:>7.3f} {ari:>8.4f}")
        if ari > best_o[0]:
            best_o = (ari, (lo, hi))

    # ---- controls: same values, wrong assignment ----
    print()
    lo, hi = best_o[1]
    ctrl = []
    for s in range(3):
        ak = lo + (hi - lo) * rng.permutation(lam_n)
        ctrl.append(score(solve(ak))[1])
    acc_i, ari_i = score(solve(lo + (hi - lo) * (1.0 - lam_n)))
    print(f"{'control: shuffled per-clade (3 seeds)':<44} {'':>7} {np.mean(ctrl):>8.4f}")
    print(f"{'control: INVERTED (1 - lambda)':<44} {acc_i:>7.3f} {ari_i:>8.4f}")

    # ---- oracle phi ----
    print()
    best_p = (-1, None)
    for av in [0.05, 0.10, 0.20]:
        acc, ari = score(solve(np.full(K, av), pk=lam_n))
        print(f"{'ORACLE phi = true lambda, alpha=' + format(av,'.2f'):<44} {acc:>7.3f} {ari:>8.4f}")
        if ari > best_p[0]:
            best_p = (ari, av)
        acc, ari = score(solve(np.full(K, av), pk=np.full(K, 0.5)))
        print(f"{'  (matched control: phi = 0.5 uniform)':<44} {acc:>7.3f} {ari:>8.4f}")

    print("-" * 62)
    print(f"  best global alpha        ARI {best_g[0]:.4f}")
    print(f"  best ORACLE per-clade    ARI {best_o[0]:.4f}   gain {best_o[0]-best_g[0]:+.4f}")
    print(f"  shuffled control         ARI {np.mean(ctrl):.4f}")
    print(f"  best ORACLE phi          ARI {best_p[0]:.4f}")
    print("\n  If oracle ~= global ~= shuffled, per-clade weighting has no headroom here.")


if __name__ == "__main__":
    main()
