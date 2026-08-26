"""Ablation: does the tree contribute anything to SpOTr's mapping?

Compares the real Numbat phylogeny against (a) a shuffled tree that keeps the exact
same distance distribution but destroys which cell is which, and (b) alpha=0, which
switches the structural term off entirely.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "src"))
from evaluate import score, fmt  # noqa: E402


def main(npz="bcsa3_B1_inputs.npz", balance=40, eps=0.02, alphas=(0.25, 0.5, 0.75)):
    import spotr.spotr as S
    from run_bcsa3 import build

    d, sc_ad, st_ad, C_tree, A = build(HERE / npz, balance=balance)
    clones, truth, soft = d["clones"], d["spot_clone_numbat"], d["spot_soft"]
    C_space = np.asarray(d["C_space"])
    rng = np.random.default_rng(0)

    variants = {"real tree": C_tree}
    for s in range(3):                       # shuffled: same distances, wrong cells
        p = rng.permutation(C_tree.shape[0])
        variants[f"shuffled tree {s+1}"] = C_tree[np.ix_(p, p)]

    rows = []
    for name, T in variants.items():
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(sc_ad, st_ad, T, C_space)
        for al in alphas:
            g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, eps, T_sinkhorn=100, J_alt=5, alpha=al)
            g = np.asarray(g)
            if np.isfinite(g).all():
                rows.append(score(g.T @ A, clones, truth, soft, f"{name}, alpha={al}"))
    # tree switched off entirely
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(sc_ad, st_ad, C_tree, C_space)
    g, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, eps, T_sinkhorn=100, J_alt=5, alpha=0.0)
    rows.append(score(np.asarray(g).T @ A, clones, truth, soft, "no tree (alpha=0)"))

    print(fmt(rows))
    real = [r for r in rows if r["method"].startswith("real")]
    shuf = [r for r in rows if r["method"].startswith("shuffled")]
    for key in ("argmax_acc", "ari", "mean_clone_pearson"):
        rv = np.mean([r[key] for r in real]); sv = np.mean([r[key] for r in shuf])
        print(f"\n{key:>20}:  real tree {rv:+.4f}   shuffled {sv:+.4f}   delta {rv - sv:+.4f}")


if __name__ == "__main__":
    main(npz=(sys.argv[1] if len(sys.argv) > 1 else "bcsa3_B1_inputs.npz"))
