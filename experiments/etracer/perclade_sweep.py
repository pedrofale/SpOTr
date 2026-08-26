"""Does each clade's BEST alpha track that clade's measured spatial concordance?

This tests the interpretability claim independently of any training objective. Instead of
fitting alpha, we sweep each clade's alpha one at a time (all others held at the global
optimum) and record which value maximises that clade's own recovery. If the concept is
sound, spatially concordant clades should prefer a higher alpha.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

BASE_ALPHA = 0.075
GRID = [0.0, 0.05, 0.10, 0.20, 0.40]
EPS, T_SINK, J_ALT = 0.02, 50, 3


def couple(alpha_k, Cf, C_T, C_S, a, b, omega, sF, sL, gamma0):
    g = gamma0
    for _ in range(J_ALT):
        C = S.build_perclade_alpha_cost(alpha_k, Cf, C_T, C_S, a, b, g, omega, sF, sL)
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
    return np.asarray(g)


def main():
    d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
    A, clones, truth, cell_clone = d["A"], d["clones"], d["bin_clone"], d["cell_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A)
    K = len(clones)

    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)

    def clade_score(g, k):
        prop = g.T @ A
        prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
        tgt = (truth == clones[k]).astype(float)
        if tgt.std() == 0 or prop[:, k].std() == 0:
            return np.nan
        return float(np.corrcoef(prop[:, k], tgt)[0, 1])

    rows = []
    for k in range(K):
        best_v, best_s, curve = np.nan, -np.inf, []
        for v in GRID:
            ak = jnp.asarray(np.full(K, BASE_ALPHA, dtype=np.float32).at_set if False
                             else np.full(K, BASE_ALPHA, dtype=np.float32))
            ak = jnp.asarray(np.array(ak).copy())
            ak = ak.at[k].set(v)
            g = couple(ak, Cf, C_T, C_S, a, b, omega, sF, sL, gamma0)
            if not np.isfinite(g).all():
                curve.append(np.nan); continue
            s = clade_score(g, k)
            curve.append(s)
            if np.isfinite(s) and s > best_s:
                best_s, best_v = s, v
        rows.append(dict(clone_id=str(clones[k]), best_alpha=best_v, best_score=best_s,
                         curve=";".join(f"{c:.4f}" if np.isfinite(c) else "nan" for c in curve)))
        print(f"  {clones[k]:>5}  best alpha {best_v:<5}  score {best_s:+.4f}   curve {rows[-1]['curve']}")

    df = pd.DataFrame(rows)
    prem = pd.read_csv(HERE / "etracer_premise.csv")
    prem = prem[prem.stage == "LE"].copy()
    prem["clone_id"] = prem["clone"].str.split("-").str[0]
    prem = prem.groupby("clone_id").agg(r=("r", "mean"), p=("p", "min")).reset_index()
    df = df.merge(prem, on="clone_id", how="inner").dropna(subset=["best_alpha", "r"])
    df.to_csv(HERE / "perclade_sweep.csv", index=False)

    print(f"\n{len(df)} clades with both a swept optimum and a measured concordance\n")
    print(df[["clone_id", "best_alpha", "r", "p"]].sort_values("best_alpha", ascending=False)
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(df) >= 5:
        pr, pp = stats.pearsonr(df.best_alpha, df.r)
        sr, sp = stats.spearmanr(df.best_alpha, df.r)
        print(f"\n  Pearson  best-alpha vs concordance = {pr:+.3f} (p = {pp:.4f})")
        print(f"  Spearman                            = {sr:+.3f} (p = {sp:.4f})")
        hi = df[df.r >= df.r.median()].best_alpha.mean()
        lo = df[df.r < df.r.median()].best_alpha.mean()
        print(f"  mean best alpha: concordant clades {hi:.3f} vs discordant {lo:.3f}")


if __name__ == "__main__":
    main()
