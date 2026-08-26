"""Finer per-clade phi sweep with a continuous optimum.

The 5-point grid was too coarse: many clades tied on the middle value, which flattens the
correlation. Here the grid is 11 points and the optimum is refined by fitting a parabola
through the peak and its neighbours, giving a continuous preferred phi per clade.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

P_GRID = np.round(np.linspace(0.0, 1.0, 11), 3)
BASE_ALPHA, EPS, T_SINK, J_ALT, MIN_BINS = 0.10, 0.02, 50, 3, 40


def refine(vals, scores):
    """Continuous argmax by parabolic fit through the peak and its two neighbours."""
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).any():
        return np.nan
    i = int(np.nanargmax(s))
    if i == 0 or i == len(s) - 1 or not np.isfinite(s[i-1:i+2]).all():
        return float(vals[i])
    y0, y1, y2 = s[i-1], s[i], s[i+1]
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-12:
        return float(vals[i])
    step = vals[1] - vals[0]
    off = 0.5 * (y0 - y2) / den
    return float(np.clip(vals[i] + off * step, vals[0], vals[-1]))


def run_stage(stage):
    d = np.load(HERE / f"etracer_inputs_{stage}.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); Omega = jnp.asarray((A @ A.T).astype(np.float32))
    K = len(clones)
    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    ak = jnp.asarray(np.full(K, BASE_ALPHA, dtype=np.float32))
    print(f"[{stage}] {A.shape[0]} cells x {len(truth)} bins, {K} clades, {len(P_GRID)} phi values")

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

    prem = pd.read_csv(HERE / "etracer_premise.csv")
    prem = prem[prem.stage == stage].copy()
    prem["clone_id"] = prem["clone"].str.split("-").str[0]
    prem = prem.groupby("clone_id").agg(r=("r", "mean")).reset_index()

    rows = []
    for k in range(K):
        sc = [curves[v][k] for v in P_GRID]
        rows.append(dict(stage=stage, clone_id=str(clones[k]),
                         n_bins=int((truth == clones[k]).sum()),
                         pref_phi=refine(P_GRID, sc),
                         peak_score=float(np.nanmax(sc)) if np.isfinite(sc).any() else np.nan))
    return pd.DataFrame(rows).merge(prem, on="clone_id", how="inner")


def main():
    df = pd.concat([run_stage(s) for s in ["LE", "EE"]], ignore_index=True)
    df.to_csv(HERE / "phi_fine.csv", index=False)
    sub = df[(df.n_bins >= MIN_BINS) & df.pref_phi.notna()]
    print(f"\npooled: {len(sub)} clades with >= {MIN_BINS} bins\n")
    print(sub.sort_values("r", ascending=False)[["stage","clone_id","n_bins","r","pref_phi"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pr, pp = stats.pearsonr(sub.pref_phi, sub.r); sr, sp = stats.spearmanr(sub.pref_phi, sub.r)
    hi = sub[sub.r >= sub.r.median()].pref_phi.mean(); lo = sub[sub.r < sub.r.median()].pref_phi.mean()
    print(f"\n  preferred phi vs within-clone concordance   (n = {len(sub)})")
    print(f"    Pearson  {pr:+.3f}  (p = {pp:.4f})")
    print(f"    Spearman {sr:+.3f}  (p = {sp:.4f})")
    print(f"    mean preferred phi: concordant {hi:.3f} vs discordant {lo:.3f}")
    print(f"    ties at grid midpoint: {(np.abs(sub.pref_phi-0.5)<1e-6).sum()}/{len(sub)}"
          f"  (was 15/23 on the coarse grid)")


if __name__ == "__main__":
    main()
