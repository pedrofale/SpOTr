"""SpOTr on eTRACER late-escape: does the real lineage tree beat a shuffled one?

Forward-only (no gradients): for each alpha we solve the FGW Sinkhorn problem and score
the coupling. The comparison is internally controlled -- real tree, shuffled tree (same
distance distribution, wrong cells) and no tree all see identical expression and spatial
inputs, so the composition mismatch between the two modalities affects all three equally.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score, accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
EPS = 0.02
rng = np.random.default_rng(0)


def score(prop, clones, truth):
    prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
    pred = np.asarray(clones)[prop.argmax(1)]
    maj = pd.Series(truth).value_counts(normalize=True).iloc[0]
    return dict(acc=accuracy_score(truth, pred), ari=adjusted_rand_score(truth, pred),
                majority=maj, n_pred_clones=len(set(pred)))


def main():
    import spotr.spotr as S
    import anndata as ad
    d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)
    A, clones, truth = d["A"], d["clones"], d["bin_clone"]
    C_space = d["C_space"]
    asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
    print(f"[run] {asc.shape[0]} cells x {ast.shape[0]} bins, {len(clones)} clones, eps={EPS}")

    variants = {"real tree": d["C_tree"]}
    for s in range(3):
        p = rng.permutation(d["C_tree"].shape[0])
        variants[f"shuffled tree {s+1}"] = d["C_tree"][np.ix_(p, p)]

    rows = []
    for name, T in variants.items():
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(asc, ast, T, C_space)
        for al in ALPHAS:
            if al == 0.0 and name != "real tree":
                continue                      # alpha=0 ignores the tree; report once
            g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=100, J_alt=5, alpha=al)
            g = np.asarray(g)
            if not np.isfinite(g).all():
                print(f"  {name} alpha={al}: non-finite"); continue
            r = score(g.T @ A, clones, truth)
            r.update(variant=("no tree (alpha=0)" if al == 0.0 else name), alpha=al)
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "etracer_ablation.csv", index=False)
    print(f"\n{'variant':<22} {'alpha':>6} {'acc':>7} {'major':>7} {'ARI':>8} {'#clones':>8}")
    print("-" * 62)
    for _, r in df.iterrows():
        print(f"{r['variant']:<22} {r['alpha']:>6.2f} {r['acc']:>7.3f} {r['majority']:>7.3f} "
              f"{r['ari']:>8.3f} {int(r['n_pred_clones']):>8}")
    print("-" * 62)
    real = df[df.variant == "real tree"]
    shuf = df[df.variant.str.startswith("shuffled")]
    nt = df[df.variant.str.startswith("no tree")]
    for k in ("acc", "ari"):
        rv, sv = real[k].mean(), shuf[k].mean()
        print(f"  mean {k:>4}:  real {rv:+.4f}   shuffled {sv:+.4f}   "
              f"no-tree {nt[k].mean():+.4f}   delta(real-shuf) {rv - sv:+.4f}")
        # per-alpha paired comparison
        pair = [(float(a), float(real[real.alpha == a][k].iloc[0]),
                 float(shuf[shuf.alpha == a][k].mean()))
                for a in ALPHAS if a > 0 and (real.alpha == a).any()]
        wins = sum(1 for _, r_, s_ in pair if r_ > s_)
        print(f"            real beats shuffled at {wins}/{len(pair)} alphas")


if __name__ == "__main__":
    main()
