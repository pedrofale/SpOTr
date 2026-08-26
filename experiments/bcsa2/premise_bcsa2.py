"""BCSA2 premise measurement, ALLELE-ONLY clone calls (p_y_*).

Tests, on Visium spots:
  1. clone assignment from allele posterior argmax + confidence distribution
  2. spatial coherence per clone (k=12 NN, excess over chance)
  3. permutation null for coherence (999 perms)
  4. Mantel: clone tree distance vs clone spatial centroid distance (999 perms)
"""
import sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

rng = np.random.default_rng(0)
K_NN = 12
N_PERM = 999
MIN_SPOTS = 20


def load_coords(spatial_dir):
    p = Path(spatial_dir)
    f = p / "tissue_positions.csv"
    if f.exists():
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df = df.set_index("barcode")
        return df[df["in_tissue"] == 1][["pxl_row_in_fullres", "pxl_col_in_fullres"]].astype(float)
    df = pd.read_csv(p / "tissue_positions_list.csv", header=None,
                     names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("barcode")
    return df[df["in_tissue"] == 1][["pxl_row", "pxl_col"]].astype(float)


def tree_dist(edges_csv):
    e = pd.read_csv(edges_csv)
    nodes = sorted(set(e["from_clone"]) | set(e["to_clone"]))
    idx = {c: i for i, c in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(D, 0.0)
    for _, r in e.iterrows():
        i, j = idx[r["from_clone"]], idx[r["to_clone"]]
        D[i, j] = D[j, i] = float(r["n_events"])
    for k in range(len(nodes)):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return np.array(nodes), D


def coherence(labels, nbr_idx, clones):
    """per-clone excess-over-chance neighbour concordance"""
    nbr_lab = labels[nbr_idx]                       # (n, k)
    out = {}
    for c in clones:
        m = labels == c
        share = m.mean()
        frac = (nbr_lab[m] == c).mean()
        out[c] = frac - share
    return out


def perm_null(labels, nbr_idx, clones, n_perm=N_PERM, seed=0):
    r = np.random.default_rng(seed)
    obs = coherence(labels, nbr_idx, clones)
    null = {c: np.empty(n_perm) for c in clones}
    for t in range(n_perm):
        pl = r.permutation(labels)
        nb = pl[nbr_idx]
        for c in clones:
            m = pl == c
            null[c][t] = (nb[m] == c).mean() - m.mean()
    res = {}
    for c in clones:
        p = (np.sum(null[c] >= obs[c]) + 1) / (n_perm + 1)
        res[c] = dict(excess=float(obs[c]), null_mean=float(null[c].mean()),
                      null_sd=float(null[c].std()), p=float(p))
    return res


def mantel(labels, xy, nodes, Dtree, clones, n_perm=N_PERM, seed=1):
    pos = {c: i for i, c in enumerate(nodes)}
    cl = [c for c in clones if c in pos]
    if len(cl) < 3:
        return dict(n_clones=len(cl), r=float("nan"), p=float("nan"))
    cent = np.stack([xy[labels == c].mean(0) for c in cl])
    Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
    Dt = Dtree[np.ix_([pos[c] for c in cl], [pos[c] for c in cl])]
    iu = np.triu_indices(len(cl), 1)
    r_obs = np.corrcoef(Dt[iu], Ds[iu])[0, 1]
    rr = np.random.default_rng(seed)
    null = np.array([np.corrcoef(Dt[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                     for p in (rr.permutation(len(cl)) for _ in range(n_perm))])
    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return dict(n_clones=len(cl), clones=[int(c) for c in cl], r=float(r_obs), p=float(p),
                n_distinct_perms=int(math.factorial(len(cl))))


def analyse(tag, nbdir, prefix, spatial_dir, tree_csv, out):
    nbdir = Path(nbdir)
    cp = pd.read_csv(nbdir / "clone_post_2.tsv", sep="\t")
    ycols = sorted([c for c in cp.columns if c.startswith("p_y_")], key=lambda s: int(s.split("_")[-1]))
    pcols = sorted([c for c in cp.columns if c.startswith("p_") and c.split("_")[-1].isdigit()
                    and not c.startswith("p_x_") and not c.startswith("p_y_")],
                   key=lambda s: int(s.split("_")[-1]))
    xcols = sorted([c for c in cp.columns if c.startswith("p_x_")], key=lambda s: int(s.split("_")[-1]))
    clone_ids = np.array([int(c.split("_")[-1]) for c in ycols])
    K = len(ycols)

    v = cp[cp["cell"].str.startswith(prefix + "_")].copy()
    v["bc"] = v["cell"].str.replace(f"^{prefix}_", "", regex=True)
    coords = load_coords(spatial_dir)
    n_before = len(v)
    v = v[v["bc"].isin(coords.index)].reset_index(drop=True)
    xy = coords.loc[v["bc"]].to_numpy()

    Py = v[ycols].to_numpy(); Pj = v[pcols].to_numpy(); Px = v[xcols].to_numpy()
    lab_y = clone_ids[Py.argmax(1)]
    lab_j = clone_ids[Pj.argmax(1)]
    lab_x = clone_ids[Px.argmax(1)]
    lab_opt = v["clone_opt"].to_numpy()

    maxy, maxj = Py.max(1), Pj.max(1)
    ent = lambda P: -(np.where(P > 0, P * np.log(P), 0)).sum(1) / np.log(K)
    ey, ej = ent(Py), ent(Pj)

    tree = cKDTree(xy)
    _, nbr = tree.query(xy, k=K_NN + 1)
    nbr = nbr[:, 1:]

    res = {"tag": tag, "numbat_dir": str(nbdir), "n_clones_in_post": K,
           "n_spots_in_post": int(n_before), "n_spots_with_coords": int(len(v))}

    res["posterior_confidence"] = {
        "allele_p_y": dict(mean_max=float(maxy.mean()), median_max=float(np.median(maxy)),
                           q25=float(np.percentile(maxy, 25)), q75=float(np.percentile(maxy, 75)),
                           frac_max_gt_90=float((maxy > 0.9).mean()),
                           frac_max_gt_50=float((maxy > 0.5).mean()),
                           frac_near_uniform=float((maxy < 1.0 / K + 0.05).mean()),
                           mean_norm_entropy=float(ey.mean())),
        "joint_p": dict(mean_max=float(maxj.mean()), median_max=float(np.median(maxj)),
                        frac_max_gt_90=float((maxj > 0.9).mean()),
                        frac_max_gt_50=float((maxj > 0.5).mean()),
                        frac_near_uniform=float((maxj < 1.0 / K + 0.05).mean()),
                        mean_norm_entropy=float(ej.mean())),
    }
    res["agreement"] = {
        "allele_vs_joint_argmax": float((lab_y == lab_j).mean()),
        "allele_vs_clone_opt": float((lab_y == lab_opt).mean()),
        "expr_vs_joint_argmax": float((lab_x == lab_j).mean()),
        "allele_vs_expr_argmax": float((lab_y == lab_x).mean()),
        "pearson_flat_p_vs_p_y": float(np.corrcoef(Pj.ravel(), Py.ravel())[0, 1]),
        "pearson_flat_p_vs_p_x": float(np.corrcoef(Pj.ravel(), Px.ravel())[0, 1]),
        "pearson_flat_p_x_vs_p_y": float(np.corrcoef(Px.ravel(), Py.ravel())[0, 1]),
        "per_clone_pearson_p_vs_p_y": {int(c): float(np.corrcoef(Pj[:, i], Py[:, i])[0, 1])
                                       for i, c in enumerate(clone_ids)},
        "per_clone_pearson_p_vs_p_x": {int(c): float(np.corrcoef(Pj[:, i], Px[:, i])[0, 1])
                                       for i, c in enumerate(clone_ids)},
    }

    for name, lab in (("ALLELE_p_y", lab_y), ("JOINT_clone_opt", lab_opt)):
        vc = pd.Series(lab).value_counts().sort_index()
        clones_all = [int(c) for c in vc.index]
        clones_big = [int(c) for c in vc.index if vc[c] >= MIN_SPOTS]
        coh = perm_null(lab, nbr, clones_big, seed=hash(name) % 1000)
        big_mean = float(np.mean([coh[c]["excess"] for c in clones_big])) if clones_big else float("nan")
        nodes, Dt = tree_dist(tree_csv)
        mt = mantel(lab, xy, nodes, Dt, clones_big)
        res[name] = dict(
            spot_counts={int(c): int(vc[c]) for c in clones_all},
            clones_ge20=clones_big,
            coherence_per_clone={int(c): coh[c] for c in clones_big},
            mean_coherence_clones_ge20=big_mean,
            mantel=mt,
        )
    Path(out).write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    r = analyse(*sys.argv[1:7])
    print(json.dumps(r, indent=2, default=str))
