"""Agreement between a method's map and the truth, per clone, with the metrics of the spatial deconvolution benchmark of Li et al. (2022):

    PCC   Pearson correlation of the clone's proportions across spots
    SSIM  each column divided by its maximum; one global window: luminance x contrast x structure (k1 0.01, k2 0.03, L 1)
    RMSE  each column z-scored; root mean squared difference
    JS    each column divided by its sum; Jensen-Shannon divergence (natural log)

Proportions are each spot's clone composition among its true cancer cells with a clone (truth) and among the map's mass (method), on the
spots with at least one true such cell, for the clones with at least `min_cells` true cells; a spot the map leaves empty counts as zero for
every clone.
"""
import numpy as np

MIN_CELLS = 20


def ssim(a, b):
    a = a / max(a.max(), 1e-300); b = b / max(b.max(), 1e-300); C1, C2 = 0.01 ** 2, 0.03 ** 2; C3 = C2 / 2
    m1, m2 = a.mean(), b.mean(); s1, s2 = a.std(), b.std(); s12 = ((a - m1) * (b - m2)).mean()
    return float((2 * m1 * m2 + C1) / (m1 ** 2 + m2 ** 2 + C1) * (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2) * (s12 + C3) / (s1 * s2 + C3))


def js(a, b):
    p = a / max(a.sum(), 1e-300); q = b / max(b.sum(), 1e-300); m = (p + q) / 2
    kl = lambda x: float(np.sum(np.where(x > 0, x * np.log(np.maximum(x, 1e-300) / np.maximum(m, 1e-300)), 0.0)))
    return 0.5 * kl(p) + 0.5 * kl(q)


def rmse(a, b):
    z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
    return float(np.sqrt(((z(a) - z(b)) ** 2).mean()))


def per_clone(P_map, N_true, min_cells=MIN_CELLS):
    """P_map: spots x clones, the map's mass of each clone (any scale); N_true: spots x clones, the true number of cells of each clone.
    Returns {metric: [one value per scored clone]}."""
    P_map, N_true = np.asarray(P_map, float), np.asarray(N_true, float); h = N_true.sum(1) > 0
    Q = N_true[h] / N_true[h].sum(1, keepdims=True); P = P_map[h] / np.maximum(P_map[h].sum(1, keepdims=True), 1e-300)
    ks = [k for k in range(Q.shape[1]) if N_true[:, k].sum() >= min_cells and Q[:, k].std() > 0]; out = dict(PCC=[], SSIM=[], RMSE=[], JS=[])
    for k in ks:
        a, b = Q[:, k], P[:, k]
        out["PCC"].append(float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 else 0.0); out["SSIM"].append(ssim(a, b)); out["RMSE"].append(rmse(a, b))
        out["JS"].append(js(a, b))
    return out


def scores(P_map, N_true, min_cells=MIN_CELLS):
    """per_clone, averaged over the clones"""
    pc = per_clone(P_map, N_true, min_cells); return {k: float(np.mean(v)) if v else float("nan") for k, v in pc.items()} | dict(clones=len(pc["PCC"]))
