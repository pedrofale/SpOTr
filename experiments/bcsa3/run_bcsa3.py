"""Run SpOTr on a real BCSA3 Visium section and score it against the baselines."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import anndata as ad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from evaluate import score, fmt  # noqa: E402


def balance_index(cell_clone, per_clone, seed=0):
    """Resample cells so every clone carries equal OT mass.

    prepare_ot_inputs hardcodes a uniform source marginal, so each clone's total
    mass is proportional to how many SS3 cells it has. On BCSA3 that pins the
    predicted spot composition to the SS3 composition, which does not match the
    Visium one at all. Equalising cells per clone removes that prior without
    using any information from the target.
    """
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(cell_clone):
        pool = np.where(cell_clone == c)[0]
        idx.append(rng.choice(pool, size=per_clone, replace=len(pool) < per_clone))
    return np.sort(np.concatenate(idx))


def build(npz, balance=0, seed=0):
    d = np.load(npz, allow_pickle=True)
    R_sc, C_tree, A = d["R_sc"], d["C_tree"], d["clone_assignments"]
    cell_clone = d["cell_clone"]
    if balance:
        idx = balance_index(cell_clone, balance, seed)
        R_sc, A, cell_clone = R_sc[idx], A[idx], cell_clone[idx]
        C_tree = C_tree[np.ix_(idx, idx)]
    sc_ad = ad.AnnData(np.asarray(R_sc, dtype=np.float32))
    sc_ad.obs["clade"] = [str(c) for c in cell_clone]
    sc_ad.obs["clade"] = sc_ad.obs["clade"].astype("category")
    st_ad = ad.AnnData(np.asarray(d["R_st"], dtype=np.float32))
    st_ad.layers["ratio"] = st_ad.X.copy()
    return d, sc_ad, st_ad, np.asarray(C_tree), np.asarray(A)


def main(npz="bcsa3_B1_inputs.npz", n_iters=50, lr=1e-2, alpha=0.5, eps=None,
         T_sinkhorn=50, J_alt=3, use_decomp=False, tag="spotr", out=None,
         balance=0, seed=0):
    import spotr.spotr as S

    d, sc_ad, st_ad, C_tree, A = build(HERE / npz, balance=balance, seed=seed)
    clones, truth, soft = d["clones"], d["spot_clone_numbat"], d["spot_soft"]
    Ccn = d["clone_cnvs"]
    print(f"[run] {sc_ad.shape[0]} cells x {st_ad.shape[0]} spots x {sc_ad.shape[1]} genes, "
          f"{len(clones)} clones, n_iters={n_iters} lr={lr} alpha0={alpha}")

    t0 = time.time()
    res = S.run_spotr_perclade_alpha(
        sc_ad, st_ad,
        C_tree, np.asarray(d["C_space"]),
        A, Ccn,
        spot_layer="ratio", clade_column="clade", clade_to_ignore="__none__",
        eps=eps, T_sinkhorn=T_sinkhorn, J_alt=J_alt,
        n_iters=n_iters, lr=lr, alpha=alpha, use_decomp=use_decomp,
        use_loss_ls=True,
    )
    dt = time.time() - t0

    gamma = np.asarray(res["coupling"])
    prop = gamma.T @ A                      # [spots, K]
    row = score(prop, clones, truth, soft, f"SpOTr ({tag})")
    print(f"\n[run] done in {dt:.1f}s")
    print(fmt([row]))
    print("\nalpha by clade:", {k: round(float(v), 3) for k, v in res["alpha_by_clade"].items()})
    losses = np.asarray(res.get("loss_hist", []))
    if losses.size:
        print(f"loss: {losses[0]:.5f} -> {losses[-1]:.5f}  (nan: {np.isnan(losses).any()})")

    out = Path(out or HERE / f"result_{tag}.json")
    json.dump({**row, "seconds": dt, "n_iters": n_iters, "lr": lr,
               "alpha_by_clade": {str(k): float(v) for k, v in res["alpha_by_clade"].items()}},
              open(out, "w"), indent=2)
    np.save(HERE / f"prop_{tag}.npy", prop)
    print(f"[run] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="bcsa3_B1_inputs.npz")
    ap.add_argument("--n-iters", type=int, default=50, dest="n_iters")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--T-sinkhorn", type=int, default=50, dest="T_sinkhorn")
    ap.add_argument("--J-alt", type=int, default=3, dest="J_alt")
    ap.add_argument("--use-decomp", action="store_true", dest="use_decomp")
    ap.add_argument("--tag", default="spotr")
    ap.add_argument("--balance", type=int, default=0,
                    help="resample to this many cells per clone (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
