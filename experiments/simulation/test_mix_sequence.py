"""Three questions in order, in the regime where the tree actually carries information.

  Q1  Does the tree help at all? (best global alpha > 0)   -- precondition
  Q2  Is there headroom in per-clade weights? (oracle vs best global)
  Q3  Does a fitted/preferred per-clade alpha recover the truth?

Interpretation being tested (the user's): alpha_k is how much clade k relies on STRUCTURE
rather than EXPRESSION. In "mix" mode expression is mostly external and roughly
clade-independent, so per-clade expression informativeness is near-constant and the known
lambda is a valid proxy for that ratio.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score, accuracy_score
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T_SINK, J_ALT = 0.02, 50, 3
rng = np.random.default_rng(0)


def main(npz="sim_mix.npz"):
    d = np.load(HERE / npz, allow_pickle=True)
    A, clones, truth, lam = d["A"], d["clones"], d["bin_clone"], d["true_lambda"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    n_sp = np.array([(truth == c).sum() for c in clones])
    print(f"[mix] {A.shape[0]} cells x {len(truth)} spots, {K} clades")

    def solve(ak):
        g = gamma0
        for _ in range(J_ALT):
            C = build_perclade_alpha_cost_v2(jnp.asarray(ak.astype(np.float32)),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
        return np.asarray(g)

    def score(g):
        if not np.isfinite(g).all():
            return np.nan, np.nan
        p = g.T @ A
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        pred = clones[p.argmax(1)]
        return accuracy_score(truth, pred), adjusted_rand_score(truth, pred)

    print("\nQ1  global alpha sweep -- does the tree help?")
    print(f"    {'alpha':>7} {'acc':>7} {'ARI':>8}")
    grid = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
    curve = {}
    for av in grid:
        acc, ari = score(solve(np.full(K, av)))
        curve[av] = ari
        print(f"    {av:>7.2f} {acc:>7.3f} {ari:>8.4f}")
    best_av = max(curve, key=lambda k: curve[k])
    print(f"    -> best global alpha = {best_av} (ARI {curve[best_av]:.4f}); "
          f"alpha=0 gives {curve[0.0]:.4f}  ==> tree "
          f"{'HELPS' if curve[best_av] > curve[0.0] + 1e-4 else 'does NOT help'}")
    if curve[best_av] <= curve[0.0] + 1e-4:
        print("\n    Precondition fails -- per-clade weighting cannot be tested here.")
        return

    print("\nQ2  oracle per-clade alpha from true lambda -- is there headroom?")
    lam_n = (lam - lam.min()) / max(lam.max() - lam.min(), 1e-9)
    best_o, best_rng = -1, None
    for lo, hi in [(0.0, best_av), (0.0, 2 * best_av), (best_av / 2, 1.5 * best_av),
                   (0.0, 1.5 * best_av), (best_av / 2, 2 * best_av)]:
        acc, ari = score(solve(lo + (hi - lo) * lam_n))
        print(f"    ORACLE alpha in [{lo:.2f},{hi:.2f}]      {acc:>7.3f} {ari:>8.4f}")
        if ari > best_o:
            best_o, best_rng = ari, (lo, hi)
    lo, hi = best_rng
    sh = [score(solve(lo + (hi - lo) * rng.permutation(lam_n)))[1] for _ in range(3)]
    inv = score(solve(lo + (hi - lo) * (1 - lam_n)))[1]
    print(f"    shuffled control (3 seeds)         {np.mean(sh):>16.4f}")
    print(f"    inverted control                   {inv:>16.4f}")
    print(f"    -> oracle {best_o:.4f} vs best global {curve[best_av]:.4f} "
          f"(gain {best_o - curve[best_av]:+.4f}), vs shuffled {np.mean(sh):.4f}")

    print("\nQ3  preferred per-clade alpha vs true lambda")
    a_grid = [0.0, best_av / 2, best_av, 1.5 * best_av, 2 * best_av]
    curves = {}
    for av in a_grid:
        g = solve(np.full(K, av))
        p = g.T @ A
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        curves[av] = np.array([
            np.nan if (truth == clones[k]).std() == 0 or p[:, k].std() == 0
            else np.corrcoef(p[:, k], (truth == clones[k]).astype(float))[0, 1]
            for k in range(K)])
    pref = np.array([a_grid[int(np.nanargmax([curves[v][k] for v in a_grid]))]
                     if np.isfinite([curves[v][k] for v in a_grid]).any() else np.nan
                     for k in range(K)])
    ok = np.isfinite(pref) & (n_sp >= 20)
    pr, pp = stats.pearsonr(pref[ok], lam[ok]); sr, sp = stats.spearmanr(pref[ok], lam[ok])
    print(f"    n={ok.sum()}   Pearson {pr:+.3f} (p={pp:.4f})   Spearman {sr:+.3f} (p={sp:.4f})")
    print(f"    mean preferred alpha: lambda>=0.5 -> {pref[ok & (lam>=0.5)].mean():.3f}   "
          f"lambda<0.5 -> {pref[ok & (lam<0.5)].mean():.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
