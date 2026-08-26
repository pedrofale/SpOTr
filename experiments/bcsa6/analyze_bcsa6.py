#!/usr/bin/env python
"""
BCSA6 / TumA1 : does lineage structure predict spatial position?

ALLELE-ONLY clone calls (p_y_* columns of clone_post_2.tsv).
No expression quantity (p_*, p_x_*, clone_opt, phi_mle, gexp_roll, exp_post) is used
anywhere in this script except for reporting how much the joint call would have differed.
"""
import json, sys, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr

NUMBAT = "/mnt/data/pedro/BCSA/BCSA6/numbat_out"
COORDS = "/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro/visium/BCSA6_TumA1_visium_CR/outs/spatial/tissue_positions.csv"
HERE = "/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa6"
PREFIX = "TumA1_"
K = 12
NPERM = 999
MIN_SPOTS = 20
RNG = np.random.default_rng(0)

out = {}

# ---------------------------------------------------------------- 1. posterior
cp = pd.read_csv(f"{NUMBAT}/clone_post_2.tsv", sep="\t")
n_clones = sum(c.startswith("p_y_") for c in cp.columns)
ycols = [f"p_y_{i}" for i in range(1, n_clones + 1)]
sp = cp[cp["cell"].str.startswith(PREFIX)].copy()
sp["barcode"] = sp["cell"].str.replace(PREFIX, "", regex=False)
print(f"clones in run: {n_clones};  spot rows in clone_post_2: {len(sp)};  "
      f"SS3 rows: {len(cp) - len(sp)}")

P = sp[ycols].to_numpy(float)
bad = ~np.isfinite(P).all(1)
print(f"rows with non-finite p_y: {bad.sum()}")
P = np.where(np.isfinite(P), P, 0.0)
rs = P.sum(1, keepdims=True)
P = np.where(rs > 0, P / np.maximum(rs, 1e-300), 1.0 / n_clones)

sp["clone_y"] = np.argmax(P, 1) + 1
sp["pmax_y"] = P.max(1)
ent = -(P * np.log(np.maximum(P, 1e-300))).sum(1)
sp["ent_y"] = ent / np.log(n_clones)          # 0 = certain, 1 = uniform

# --------------------------------------------------- 2. allele depth / usability
ad = pd.read_csv(f"{HERE}/allele_depth_TumA1.tsv", sep="\t")
ad["barcode"] = ad["cell"].str.replace(PREFIX, "", regex=False)
ad = ad[ad["total"] > 0]
per_spot = ad.groupby("barcode").agg(n_seg_cov=("seg", "nunique"),
                                     depth_tot=("total", "sum"))
per_spot["depth_per_seg"] = per_spot["depth_tot"] / per_spot["n_seg_cov"]
sp = sp.merge(per_spot, left_on="barcode", right_index=True, how="left")
sp[["n_seg_cov", "depth_tot", "depth_per_seg"]] = \
    sp[["n_seg_cov", "depth_tot", "depth_per_seg"]].fillna(0)

segs = pd.read_csv(f"{NUMBAT}/segs_consensus_2.tsv", sep="\t")
n_cnv_segs = segs["seg"].nunique()

out["n_spots_clone_post"] = int(len(sp))
out["n_clones"] = int(n_clones)
out["n_cnv_segs"] = int(n_cnv_segs)
out["n_spots_any_allele"] = int((sp["n_seg_cov"] > 0).sum())
out["allele_depth_quantiles_per_spot_total"] = {
    str(q): float(np.quantile(sp["depth_tot"], q)) for q in
    (0, .05, .1, .25, .5, .75, .9, .95, 1)}
out["segs_covered_per_spot_quantiles"] = {
    str(q): float(np.quantile(sp["n_seg_cov"], q)) for q in
    (0, .05, .1, .25, .5, .75, .9, .95, 1)}
out["reads_per_spot_per_covered_seg_quantiles"] = {
    str(q): float(np.quantile(sp.loc[sp.n_seg_cov > 0, "depth_per_seg"], q))
    for q in (0, .05, .1, .25, .5, .75, .9, .95, 1)}
rr = ad["total"].to_numpy()
out["reads_per_spot_seg_pair_quantiles"] = {
    str(q): float(np.quantile(rr, q)) for q in (0, .05, .25, .5, .75, .9, .95, .99, 1)}
out["frac_spot_seg_pairs_depth_lt5"] = float((rr < 5).mean())
out["frac_spot_seg_pairs_depth_lt10"] = float((rr < 10).mean())

# posterior confidence
out["pmax_y_quantiles"] = {str(q): float(np.quantile(sp["pmax_y"], q))
                           for q in (0, .05, .1, .25, .5, .75, .9, .95, 1)}
for t in (0.5, 0.8, 0.9, 0.95, 0.99):
    out[f"frac_spots_pmax_y_gt_{t}"] = float((sp["pmax_y"] > t).mean())
out["frac_spots_near_uniform_pmax_lt_2x_chance"] = float(
    (sp["pmax_y"] < 2.0 / n_clones).mean())
out["normalised_entropy_median"] = float(sp["ent_y"].median())

# how far the allele-only call is from the joint call (report only, never used)
if "clone_opt" in sp.columns:
    out["agreement_alleleonly_vs_joint_cloneopt"] = float(
        (sp["clone_y"] == sp["clone_opt"]).mean())

# ---------------------------------------------------------------- 3. coordinates
tp = pd.read_csv(COORDS)
tp = tp[tp["in_tissue"] == 1]
m = sp.merge(tp, on="barcode", how="inner")
print(f"spots merged with coordinates: {len(m)} / {len(sp)}")
out["n_spots_with_coords"] = int(len(m))
xy = m[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(float)

tree = cKDTree(xy)
d, idx = tree.query(xy, k=K + 1)
nbr = idx[:, 1:]                                   # drop self
out["median_nn1_dist_px"] = float(np.median(d[:, 1]))
out["median_nnK_dist_px"] = float(np.median(d[:, K]))

lab = m["clone_y"].to_numpy()

def coherence(labels):
    """per-clone excess coherence = mean neighbour-match fraction - clone share"""
    res = {}
    N = len(labels)
    nb = labels[nbr]
    match = (nb == labels[:, None]).mean(1)
    for c in np.unique(labels):
        sel = labels == c
        res[int(c)] = float(match[sel].mean() - sel.sum() / N)
    return res, match

obs, match_obs = coherence(lab)
counts = pd.Series(lab).value_counts().to_dict()
keep = [c for c in obs if counts[c] >= MIN_SPOTS]

# ------------------------------------------------- 4. permutation null (999)
perm = {c: np.empty(NPERM) for c in obs}
glob = np.empty(NPERM)
for b in range(NPERM):
    pl = RNG.permutation(lab)
    r, _ = coherence(pl)
    for c in obs:
        perm[c][b] = r.get(c, np.nan)
    glob[b] = np.mean([r[c] for c in keep if c in r])

obs_mean_keep = float(np.mean([obs[c] for c in keep]))
p_glob = float((1 + np.sum(glob >= obs_mean_keep)) / (NPERM + 1))

rows = []
for c in sorted(obs, key=lambda z: -counts[z]):
    v = perm[c][np.isfinite(perm[c])]
    p = float((1 + np.sum(v >= obs[c])) / (len(v) + 1)) if len(v) else np.nan
    rows.append(dict(clone=c, n_spots=int(counts[c]), share=counts[c] / len(lab),
                     excess_coherence=obs[c],
                     null_mean=float(v.mean()) if len(v) else np.nan,
                     null_sd=float(v.std()) if len(v) else np.nan,
                     p_perm=p, used=c in keep,
                     median_pmax_y=float(m.loc[m.clone_y == c, "pmax_y"].median()),
                     median_depth=float(m.loc[m.clone_y == c, "depth_tot"].median())))
coh = pd.DataFrame(rows)
coh.to_csv(f"{HERE}/coherence_TumA1.csv", index=False)

out["coherence_mean_over_clones_ge20"] = obs_mean_keep
out["coherence_p_global"] = p_glob
out["n_clones_ge20_spots"] = len(keep)
out["n_clones_signif_p05"] = int(((coh.used) & (coh.p_perm < 0.05)).sum())
out["n_clones_signif_bonferroni"] = int(
    ((coh.used) & (coh.p_perm < 0.05 / max(len(keep), 1))).sum())
out["global_excess_coherence_all_spots"] = float(
    match_obs.mean() - np.sum((np.array(list(counts.values())) / len(lab)) ** 2))

# ------------------------------------------------- 5. tree vs space (Mantel)
edges = pd.read_csv(f"{HERE}/bcsa6_clone_edges_full.csv")
nodes = pd.read_csv(f"{HERE}/bcsa6_clone_nodes.csv")
nodes["gt"] = nodes["gt"].fillna("")
gt = {int(r.clone): set(str(r.gt).split(",")) - {"", "nan"}
      for r in nodes.itertuples()}

INF = 1e9
D = np.full((n_clones + 1, n_clones + 1), INF)
np.fill_diagonal(D, 0)
for r in edges.itertuples():
    w = float(r.n_events)
    D[int(r.from_clone), int(r.to_clone)] = w
    D[int(r.to_clone), int(r.from_clone)] = w
for k in range(1, n_clones + 1):          # Floyd-Warshall
    D = np.minimum(D, D[:, [k]] + D[[k], :])

H = np.zeros((n_clones + 1, n_clones + 1))
for i in range(1, n_clones + 1):
    for j in range(1, n_clones + 1):
        H[i, j] = len(gt[i] ^ gt[j])

cent = m.groupby("clone_y")[["pxl_row_in_fullres", "pxl_col_in_fullres"]].mean()

def mantel(clist, Dmat, tag):
    if len(clist) < 4:
        return None
    C = cent.loc[clist].to_numpy(float)
    S = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    T = Dmat[np.ix_(clist, clist)]
    iu = np.triu_indices(len(clist), 1)
    t, s = T[iu], S[iu]
    r_p = pearsonr(t, s)[0]
    r_s = spearmanr(t, s)[0]
    cnt_p = cnt_s = 0
    for _ in range(NPERM):
        o = RNG.permutation(len(clist))
        Tp = T[np.ix_(o, o)][iu]
        if abs(pearsonr(Tp, s)[0]) >= abs(r_p): cnt_p += 1
        if abs(spearmanr(Tp, s)[0]) >= abs(r_s): cnt_s += 1
    return dict(metric=tag, n_clones=len(clist), pearson_r=float(r_p),
                p_pearson=float((1 + cnt_p) / (NPERM + 1)),
                spearman_r=float(r_s),
                p_spearman=float((1 + cnt_s) / (NPERM + 1)))

clist = sorted(keep)
out["mantel_clones_used"] = clist
out["mantel"] = [x for x in (mantel(clist, D, "tree_path_n_events"),
                             mantel(clist, H, "GT_hamming")) if x]
# malignant-only (drop clone 1 = normal root, and the off-root branches 18/19)
mal = [c for c in clist if c not in (1, 18, 19)]
out["mantel_malignant_only_clones"] = mal
out["mantel_malignant_only"] = [x for x in (mantel(mal, D, "tree_path_n_events"),
                                            mantel(mal, H, "GT_hamming")) if x]

pd.DataFrame(D[1:, 1:], index=range(1, n_clones + 1),
             columns=range(1, n_clones + 1)).to_csv(f"{HERE}/clone_tree_dist.csv")
cent.to_csv(f"{HERE}/clone_centroids.csv")
m[["barcode", "clone_y", "pmax_y", "ent_y", "n_seg_cov", "depth_tot",
   "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]].to_csv(
    f"{HERE}/spot_calls_TumA1.csv", index=False)

with open(f"{HERE}/results_bcsa6.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
print(coh.to_string(index=False))
