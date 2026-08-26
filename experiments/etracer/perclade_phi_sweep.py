"""Does each clade's best PHI track its within-clone spatial concordance?

alpha_k controls structure-vs-expression. phi_k controls, within the structural term, how
much WITHIN-clade geometry is used versus cross-clade geometry:

    L_hat = (1 - phi_row) * L_cross + phi_row * L_within

The per-clone concordance measured by etracer_premise.py is a WITHIN-clone quantity (tree
distance among that clone's bins vs their spatial distance), so phi_k -- not alpha_k -- is
the parameter it should correspond to. Same objective-free sweep design as perclade_sweep.py.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

BASE_ALPHA, BASE_PHI = 0.075, 0.5
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS, T_SINK, J_ALT = 0.02, 50, 3


def couple(alpha_k, phi_k, Cf, C_T, C_S, a, b, omega, Omega, sF, sL, gamma0):
    g = gamma0
    for _ in range(J_ALT):
        C = S.build_perclade_alpha_cost(alpha_k, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                        use_decomp=True, Omega=Omega, phi_k=phi_k)
        g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T_SINK)
    return np.asarray(g)


def main():
    d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
    omega = jnp.asarray(A)
    Omega = jnp.asarray((A @ A.T).astype(np.float32))       # within-clade mask
    K = len(clones)

    gamma0, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, gamma0, eps=EPS)
    alpha_k = jnp.asarray(np.full(K, BASE_ALPHA, dtype=np.float32))

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
            pk = jnp.asarray(np.full(K, BASE_PHI, dtype=np.float32)).at[k].set(v)
            g = couple(alpha_k, pk, Cf, C_T, C_S, a, b, omega, Omega, sF, sL, gamma0)
            if not np.isfinite(g).all():
                curve.append(np.nan); continue
            s = clade_score(g, k); curve.append(s)
            if np.isfinite(s) and s > best_s:
                best_s, best_v = s, v
        rows.append(dict(clone_id=str(clones[k]), best_phi=best_v, best_score=best_s,
                         curve=";".join(f"{c:.4f}" if np.isfinite(c) else "nan" for c in curve)))
        print(f"  {clones[k]:>5}  best phi {best_v:<5} score {best_s:+.4f}  curve {rows[-1]['curve']}")

    df = pd.DataFrame(rows)
    prem = pd.read_csv(HERE / "etracer_premise.csv")
    prem = prem[prem.stage == "LE"].copy()
    prem["clone_id"] = prem["clone"].str.split("-").str[0]
    prem = prem.groupby("clone_id").agg(r=("r", "mean"), p=("p", "min")).reset_index()
    df = df.merge(prem, on="clone_id", how="inner").dropna(subset=["best_phi", "r"])
    df.to_csv(HERE / "perclade_phi_sweep.csv", index=False)

    print(f"\n{len(df)} clades\n")
    print(df[["clone_id", "best_phi", "r", "p"]].sort_values("best_phi", ascending=False)
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(df) >= 5:
        pr, pp = stats.pearsonr(df.best_phi, df.r)
        sr, sp = stats.spearmanr(df.best_phi, df.r)
        print(f"\n  Pearson  best-phi vs within-clone concordance = {pr:+.3f} (p = {pp:.4f})")
        print(f"  Spearman                                       = {sr:+.3f} (p = {sp:.4f})")
        hi = df[df.r >= df.r.median()].best_phi.mean()
        lo = df[df.r < df.r.median()].best_phi.mean()
        print(f"  mean best phi: concordant {hi:.3f} vs discordant {lo:.3f}")


if __name__ == "__main__":
    main()
