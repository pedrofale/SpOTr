"""BCSA2 premise measurement on Visium spots, ALLELE-ONLY clone calls (p_y_*).

  1. clone assignment = argmax over p_y_* (allele-only posterior) + confidence distribution
  2. spatial coherence per clone: frac of k=12 nearest spatial neighbours sharing the clone,
     minus that clone's overall share (excess over chance)
  3. permutation null for coherence, 999 label permutations, empirical p per clone
  4. Mantel: clone tree distance (branch events, and Hamming over GT segment sets)
     vs euclidean distance between clone spatial centroids, 999 permutations

Joint (p_*) and expression-only (p_x_*) calls are computed alongside for contrast only.
"""
import sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K_NN, N_PERM, MIN_SPOTS = 12, 999, 20


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


def tree_dist_events(edges_csv):
    e = pd.read_csv(edges_csv)
    nodes = sorted(set(e["from_clone"]) | set(e["to_clone"]))
    idx = {c: i for i, c in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf); np.fill_diagonal(D, 0.0)
    for _, r in e.iterrows():
        i, j = idx[r["from_clone"]], idx[r["to_clone"]]
        D[i, j] = D[j, i] = float(r["n_events"])
    for k in range(len(nodes)):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return np.array(nodes), D


def tree_dist_hamming(gt_csv):
    g = pd.read_csv(gt_csv).fillna("")
    nodes = g["clone"].to_numpy()
    sets = [set(s for s in str(x).split(",") if s) for x in g["GT"]]
    n = len(nodes)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = len(sets[i] ^ sets[j])
    return np.array(nodes), D


def coherence(labels, nbr_lab_of, clones):
    out = {}
    for c in clones:
        m = labels == c
        out[c] = (nbr_lab_of[m] == c).mean() - m.mean()
    return out


def coherence_with_null(labels, nbr, clones, n_perm=N_PERM, seed=0):
    obs = coherence(labels, labels[nbr], clones)
    r = np.random.default_rng(seed)
    null = {c: np.empty(n_perm) for c in clones}
    for t in range(n_perm):
        pl = r.permutation(labels); nb = pl[nbr]
        for c in clones:
            m = pl == c
            null[c][t] = (nb[m] == c).mean() - m.mean()
    return {int(c): dict(excess=float(obs[c]), null_mean=float(null[c].mean()),
                         null_sd=float(null[c].std()),
                         z=float((obs[c] - null[c].mean()) / max(null[c].std(), 1e-12)),
                         p_emp=float((np.sum(null[c] >= obs[c]) + 1) / (n_perm + 1)))
            for c in clones}


def mantel(labels, xy, nodes, Dtree, clones, n_perm=N_PERM, seed=1):
    pos = {int(c): i for i, c in enumerate(nodes)}
    cl = [int(c) for c in clones if int(c) in pos]
    if len(cl) < 3:
        return dict(n_clones=len(cl), r=None, p=None)
    cent = np.stack([xy[labels == c].mean(0) for c in cl])
    Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
    Dt = Dtree[np.ix_([pos[c] for c in cl], [pos[c] for c in cl])]
    iu = np.triu_indices(len(cl), 1)
    if Dt[iu].std() == 0:
        return dict(n_clones=len(cl), r=None, p=None, note="tree distances constant")
    r_obs = float(np.corrcoef(Dt[iu], Ds[iu])[0, 1])
    rr = np.random.default_rng(seed)
    null = np.array([np.corrcoef(Dt[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                     for p in (rr.permutation(len(cl)) for _ in range(n_perm))])
    return dict(n_clones=len(cl), clones=cl, r=r_obs,
                p=float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)),
                min_attainable_p=float(1.0 / math.factorial(len(cl))))


def run(tag, nbdir, prefix, spatial_dir, tree_csv, gt_csv, out):
    nbdir = Path(nbdir)
    cp = pd.read_csv(nbdir / "clone_post_2.tsv", sep="\t")
    ycols = sorted([c for c in cp.columns if c.startswith("p_y_")], key=lambda s: int(s.split("_")[-1]))
    xcols = sorted([c for c in cp.columns if c.startswith("p_x_")], key=lambda s: int(s.split("_")[-1]))
    pcols = sorted([c for c in cp.columns if c.startswith("p_") and c.split("_")[-1].isdigit()
                    and not c.startswith(("p_x_", "p_y_"))], key=lambda s: int(s.split("_")[-1]))
    cid = np.array([int(c.split("_")[-1]) for c in ycols]); K = len(ycols)

    gt = pd.read_csv(gt_csv).fillna("")
    normal_clones = set(int(c) for c, g in zip(gt["clone"], gt["GT"]) if str(g).strip() == "")

    v = cp[cp["cell"].str.startswith(prefix + "_")].copy()
    v["bc"] = v["cell"].str.replace(f"^{prefix}_", "", regex=True)
    coords = load_coords(spatial_dir)
    n_in_post = len(v)
    v = v[v["bc"].isin(coords.index)].reset_index(drop=True)
    xy = coords.loc[v["bc"]].to_numpy()

    Py, Px, Pj = v[ycols].to_numpy(), v[xcols].to_numpy(), v[pcols].to_numpy()
    lab = {"allele_p_y": cid[Py.argmax(1)], "joint_p": cid[Pj.argmax(1)],
           "expr_p_x": cid[Px.argmax(1)], "clone_opt": v["clone_opt"].to_numpy()}
    ent = lambda P: float((-(np.where(P > 0, P * np.log(P), 0)).sum(1) / np.log(K)).mean())

    _, nbr = cKDTree(xy).query(xy, k=K_NN + 1); nbr = nbr[:, 1:]
    nodes_e, De = tree_dist_events(tree_csv)
    nodes_h, Dh = tree_dist_hamming(gt_csv)

    res = dict(tag=tag, numbat_dir=str(nbdir), n_clones=K, normal_clones=sorted(normal_clones),
               n_spots_in_clone_post=int(n_in_post), n_spots_in_tissue_with_coords=int(len(v)),
               k_nn=K_NN, n_permutations=N_PERM)

    def conf(P):
        m = P.max(1)
        return dict(mean_max=float(m.mean()), median_max=float(np.median(m)),
                    q25_max=float(np.percentile(m, 25)), q75_max=float(np.percentile(m, 75)),
                    frac_max_gt_0p50=float((m > .5).mean()), frac_max_gt_0p90=float((m > .9).mean()),
                    frac_near_uniform_within_0p05_of_1overK=float((m < 1.0 / K + .05).mean()),
                    mean_normalised_entropy=ent(P))
    res["confidence"] = {"allele_p_y": conf(Py), "joint_p": conf(Pj), "expr_p_x": conf(Px),
                         "uniform_baseline_max": 1.0 / K}
    res["column_relationships"] = dict(
        pearson_all_entries_p_vs_p_y=float(np.corrcoef(Pj.ravel(), Py.ravel())[0, 1]),
        pearson_all_entries_p_vs_p_x=float(np.corrcoef(Pj.ravel(), Px.ravel())[0, 1]),
        pearson_all_entries_p_x_vs_p_y=float(np.corrcoef(Px.ravel(), Py.ravel())[0, 1]),
        argmax_agree_p_y_vs_p=float((lab["allele_p_y"] == lab["joint_p"]).mean()),
        argmax_agree_p_x_vs_p=float((lab["expr_p_x"] == lab["joint_p"]).mean()),
        argmax_agree_p_y_vs_p_x=float((lab["allele_p_y"] == lab["expr_p_x"]).mean()),
        mean_abs_diff_p_vs_p_y=float(np.abs(Pj - Py).mean()),
        mean_abs_diff_p_vs_p_x=float(np.abs(Pj - Px).mean()),
    )

    for name in ("allele_p_y", "joint_p"):
        L = lab[name]
        vc = pd.Series(L).value_counts().sort_index()
        big = [int(c) for c in vc.index if vc[c] >= MIN_SPOTS]
        coh = coherence_with_null(L, nbr, big, seed=7 if name == "allele_p_y" else 11)
        sub = [c for c in big if c not in normal_clones]
        res[name] = dict(
            spot_counts={int(c): int(vc[c]) for c in vc.index},
            clones_ge20_spots=big,
            coherence=coh,
            mean_coherence_all_clones_ge20=float(np.mean([coh[c]["excess"] for c in big])) if big else None,
            mean_coherence_tumour_subclones_only=float(np.mean([coh[c]["excess"] for c in sub])) if sub else None,
            mantel_branch_events=mantel(L, xy, nodes_e, De, big),
            mantel_GT_hamming=mantel(L, xy, nodes_h, Dh, big, seed=2),
        )
    Path(out).write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    print(json.dumps(run(*sys.argv[1:8]), indent=2, default=str))
