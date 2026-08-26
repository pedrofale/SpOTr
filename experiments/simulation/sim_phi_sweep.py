"""Positive control: does preferred per-clade phi recover the TRUE per-clade lambda?

Same sweep design used on eTRACER, but here the ground truth is known exactly: each clade
was simulated with a chosen lambda controlling how much its within-clade spatial layout
follows its own tree. If phi cannot recover a known lambda here, the real-data null is an
algorithm limitation rather than a data limitation.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

P_GRID = np.round(np.linspace(0.0, 1.0, 11), 3)
BASE_ALPHA, EPS, T_SINK, J_ALT = 0.10, 0.02, 50, 3


def refine(vals, scores):
    s = np.asarray(scores, float)
    if not np.isfinite(s).any():
        return np.nan
    i = int(np.nanargmax(s))
    if i in (0, len(s) - 1) or not np.isfinite(s[i-1:i+2]).all():
        return float(vals[i])
    y0, y1, y2 = s[i-1], s[i], s[i+1]
    den = y0 - 2*y1 + y2
    if abs(den) < 1e-12:
        return float(vals[i])
    return float(np.clip(vals[i] + 0.5*(y0-y2)/den*(vals[1]-vals[0]), vals[0], vals[-1]))


def main(npz="sim_inputs.npz", min_spots=10):
    f = HERE / npz
    import time
    age = time.time() - f.stat().st_mtime
    if age > 3600:
        raise SystemExit(f"REFUSING: {npz} is {age/3600:.1f}h old -- likely stale. "
                         "Re-run simulate_perclade.py first.")
    d = np.load(f, allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    lam, meas = d["true_lambda"], d["measured_r"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    print(f"[sweep] {A.shape[0]} cells x {len(truth)} spots, {K} clades")

    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    ak = jnp.asarray(np.full(K, BASE_ALPHA, dtype=np.float32))

    curves = {}
    for v in P_GRID:
        pk = jnp.asarray(np.full(K, v, dtype=np.float32))
        g = gamma0
        for _ in range(J_ALT):
            C = S.build_perclade_alpha_cost(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            use_decomp=True, Omega=Omega, phi_k=pk)
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
        prop = np.asarray(g).T @ A
        prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        sc = []
        for k in range(K):
            t = (truth == clones[k]).astype(float)
            sc.append(np.nan if (t.std() == 0 or prop[:, k].std() == 0)
                      else float(np.corrcoef(prop[:, k], t)[0, 1]))
        curves[v] = np.array(sc)

    n_sp = np.array([(truth == c).sum() for c in clones])
    df = pd.DataFrame(dict(
        clade=[str(c) for c in clones], n_spots=n_sp, true_lambda=lam, measured_r=meas,
        pref_phi=[refine(P_GRID, [curves[v][k] for v in P_GRID]) for k in range(K)]))
    df.to_csv(HERE / "sim_phi_sweep.csv", index=False)
    sub = df[(df.n_spots >= min_spots) & df.pref_phi.notna()]
    print(sub.sort_values("true_lambda", ascending=False).to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))

    for col, nm in [("true_lambda", "TRUE lambda"), ("measured_r", "measured concordance")]:
        pr, pp = stats.pearsonr(sub.pref_phi, sub[col]); sr, sp = stats.spearmanr(sub.pref_phi, sub[col])
        print(f"\n  preferred phi vs {nm}   (n = {len(sub)})")
        print(f"    Pearson  {pr:+.3f} (p = {pp:.4f})    Spearman {sr:+.3f} (p = {sp:.4f})")
    hi = sub[sub.true_lambda >= 0.5].pref_phi.mean(); lo = sub[sub.true_lambda < 0.5].pref_phi.mean()
    print(f"\n    mean preferred phi: lambda>=0.5 -> {hi:.3f}   lambda<0.5 -> {lo:.3f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
