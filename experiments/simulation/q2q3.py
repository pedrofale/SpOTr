"""Q2: does per-clade weighting beat the best GLOBAL weight? Q3: is it recoverable?

Q2 uses an oracle -- per-clade alpha set from the KNOWN lambda -- so it tests headroom
without depending on any estimator. Controls: the same values shuffled across clades, and
inverted. If oracle ~= shuffled ~= global, clade-specific weighting buys nothing even with
perfect knowledge.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.metrics import adjusted_rand_score
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)


def run(npz):
    d = np.load(HERE / npz, allow_pickle=True)
    A, clones, truth, lam = d["A"], d["clones"], d["bin_clone"], d["true_lambda"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A)
    K = len(clones)
    # Cold start from the product measure, matching Q1. Warm-starting from the alpha=0
    # solution gives a much better mapping (ARI ~0.78 vs ~0.14) but collapses the tree's own
    # contribution to ~+0.002 -- so per-clade weighting would again be weighting a term that
    # does not matter in that path.
    g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
    n_sp = np.array([(truth == c).sum() for c in clones])

    def solve(ak):
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.asarray(ak.astype(np.float32)),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        return np.asarray(g)

    def ari(g):
        if not np.isfinite(g).all():
            return np.nan
        p = g.T @ A
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        return adjusted_rand_score(truth, clones[p.argmax(1)])

    grid = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
    gl = {v: ari(solve(np.full(K, v))) for v in grid}
    bv = max(gl, key=lambda k: gl[k])
    lam_n = (lam - lam.min()) / max(lam.max() - lam.min(), 1e-9)

    best_o, best_r = -np.inf, None
    for lo, hi in [(0.0, bv), (0.0, min(1.0, 2 * bv)), (bv / 2, min(1.0, 1.5 * bv)),
                   (0.0, min(1.0, 1.5 * bv)), (bv / 2, min(1.0, 2 * bv))]:
        v = ari(solve(lo + (hi - lo) * lam_n))
        if v > best_o:
            best_o, best_r = v, (lo, hi)
    lo, hi = best_r
    sh = float(np.nanmean([ari(solve(lo + (hi - lo) * rng.permutation(lam_n))) for _ in range(3)]))
    inv = ari(solve(lo + (hi - lo) * (1 - lam_n)))

    # Q3: preferred per-clade alpha from a global sweep, vs true lambda
    curves = {}
    for v in grid:
        g = solve(np.full(K, v))
        p = g.T @ A
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        curves[v] = np.array([np.nan if (truth == clones[k]).std() == 0 or p[:, k].std() == 0
                              else np.corrcoef(p[:, k], (truth == clones[k]).astype(float))[0, 1]
                              for k in range(K)])
    pref = np.array([grid[int(np.nanargmax([curves[v][k] for v in grid]))]
                     if np.isfinite([curves[v][k] for v in grid]).any() else np.nan
                     for k in range(K)])
    ok = np.isfinite(pref) & (n_sp >= 15)
    pr, pp = stats.pearsonr(pref[ok], lam[ok]) if ok.sum() >= 5 else (np.nan, np.nan)
    sr, sp = stats.spearmanr(pref[ok], lam[ok]) if ok.sum() >= 5 else (np.nan, np.nan)
    return dict(npz=npz, best_global_alpha=bv, ari_global=gl[bv], ari_alpha0=gl[0.0],
                ari_oracle=best_o, ari_shuffled=sh, ari_inverted=inv,
                n_q3=int(ok.sum()), q3_pearson=pr, q3_p=pp, q3_spearman=sr, q3_sp=sp)


rows = [run(f"sim_bd_a{A}.npz") for A in ["0.5", "0.0"]]
print(f"\n{'dataset':<18} {'best a':>7} {'ARI(a=0)':>9} {'ARI global':>11} "
      f"{'ORACLE':>8} {'shuffled':>9} {'inverted':>9} {'gain':>8}")
print("-" * 92)
for r in rows:
    print(f"{r['npz']:<18} {r['best_global_alpha']:>7.2f} {r['ari_alpha0']:>9.4f} "
          f"{r['ari_global']:>11.4f} {r['ari_oracle']:>8.4f} {r['ari_shuffled']:>9.4f} "
          f"{r['ari_inverted']:>9.4f} {r['ari_oracle']-r['ari_global']:>+8.4f}")
print("-" * 92)
print(f"\n{'dataset':<18} {'n':>4} {'Q3 Pearson':>11} {'p':>8} {'Q3 Spearman':>12} {'p':>8}")
for r in rows:
    print(f"{r['npz']:<18} {r['n_q3']:>4} {r['q3_pearson']:>+11.3f} {r['q3_p']:>8.4f} "
          f"{r['q3_spearman']:>+12.3f} {r['q3_sp']:>8.4f}")
pd.DataFrame(rows).to_csv(HERE / "q2q3_results.csv", index=False)
