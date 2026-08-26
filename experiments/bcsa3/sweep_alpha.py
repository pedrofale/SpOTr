"""Forward-only sweep over the feature/structure mixing weight alpha.

No gradients: for each alpha we just solve the FGW Sinkhorn problem and score the
resulting coupling. This answers whether the OT formulation can recover the
Numbat spot clones at ANY weighting, independent of whether the optimiser works.
"""
import argparse, sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from evaluate import score, fmt  # noqa: E402


def main(npz="bcsa3_B1_inputs.npz", balance=40, T_sinkhorn=100, J_alt=5, eps=0.05, seed=0):
    import jax.numpy as jnp
    import spotr.spotr as S
    from run_bcsa3 import build

    d, sc_ad, st_ad, C_tree, A = build(HERE / npz, balance=balance, seed=seed)
    clones, truth, soft = d["clones"], d["spot_clone_numbat"], d["spot_soft"]
    C_space = np.asarray(d["C_space"])

    C_feature, C_T, C_S, a, b = S.prepare_ot_inputs(sc_ad, st_ad, C_tree, C_space)
    print(f"[sweep] {sc_ad.shape[0]} cells x {st_ad.shape[0]} spots, balance={balance}, eps={eps}")

    rows = []
    for alpha in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
        gamma, _ = S.sinkhorn_fgw(C_feature, C_T, C_S, a, b, eps,
                                  T_sinkhorn=T_sinkhorn, J_alt=J_alt, alpha=alpha)
        g = np.asarray(gamma)
        if not np.isfinite(g).all():
            print(f"  alpha={alpha}: non-finite coupling, skipped")
            continue
        prop = g.T @ A
        rows.append(score(prop, clones, truth, soft, f"alpha={alpha:<4}"))
    print()
    print(fmt(rows))
    json.dump(rows, open(HERE / "sweep_alpha.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="bcsa3_B1_inputs.npz")
    ap.add_argument("--balance", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--T-sinkhorn", type=int, default=100, dest="T_sinkhorn")
    ap.add_argument("--J-alt", type=int, default=5, dest="J_alt")
    main(**vars(ap.parse_args()))
