"""The deconvolution field's metrics (decision-real-data-metrics), with subclones in place of cell types.  2026-10-28.
Pedro, 2026-10-28: "The way we score our method should follow how cell2location, DestVI, Tangram score theirs, since we're doing the
same task"; "Yes please use the field's metrics."
Definitions follow the benchmark's own code (Li et al. 2022, Nat Methods; github.com/QuKunLab/SpatialBenchmarking,
Benchmarking/GenesMetrics.py, read 2026-10-28): each metric per feature -- here per clone, a column of per-spot proportions across spots
-- after a feature-wise scaling, then averaged over features:
  PCC   Pearson correlation of the raw columns
  SSIM  each column divided by its maximum; one global window: luminance x contrast x structure with k1 0.01, k2 0.03, L = 1
  RMSE  each column z-scored; root mean squared difference
  JS    each column divided by its sum; 0.5 KL(p || m) + 0.5 KL(q || m), m = (p + q) / 2, natural log
Proportions: each spot's clone composition among its traced cells (truth) and among the map's traced mass (map); scored on the spots
with at least one true traced cell, over clones with at least MIN_CELLS true cells; a spot the map leaves without traced mass counts as
zero for every clone.  AS (the benchmark's accuracy score): the mean over the four metrics of each arm's rank among the arms compared,
scaled to [0, 1] with 1 best.
Usage: imported (scores, accuracy_score).
"""
import numpy as np
MIN_CELLS = 20


def _ssim(a, b):
    a = a / max(a.max(), 1e-300); b = b / max(b.max(), 1e-300); C1, C2 = 0.01 ** 2, 0.03 ** 2; C3 = C2 / 2
    m1, m2 = a.mean(), b.mean(); s1, s2 = a.std(), b.std(); s12 = ((a - m1) * (b - m2)).mean()
    return float((2 * m1 * m2 + C1) / (m1 ** 2 + m2 ** 2 + C1) * (2 * s1 * s2 + C2) / (s1 ** 2 + s2 ** 2 + C2) * (s12 + C3) / (s1 * s2 + C3))


def _js(a, b):
    p = a / max(a.sum(), 1e-300); q = b / max(b.sum(), 1e-300); m = (p + q) / 2
    kl = lambda x: float(np.sum(np.where(x > 0, x * np.log(np.maximum(x, 1e-300) / np.maximum(m, 1e-300)), 0.0)))
    return 0.5 * kl(p) + 0.5 * kl(q)


def _rmse(a, b):
    z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
    return float(np.sqrt(((z(a) - z(b)) ** 2).mean()))


def scores(P_map, N_true, min_cells=MIN_CELLS):
    """P_map: spots x clones, the map's traced mass per clone (any scale); N_true: spots x clones, true traced cells per clone"""
    P_map, N_true = np.asarray(P_map, float), np.asarray(N_true, float); h = N_true.sum(1) > 0
    Q = N_true[h] / N_true[h].sum(1, keepdims=True); P = P_map[h] / np.maximum(P_map[h].sum(1, keepdims=True), 1e-300)
    ks = [k for k in range(Q.shape[1]) if N_true[:, k].sum() >= min_cells and Q[:, k].std() > 0]
    out = dict(PCC=[], SSIM=[], RMSE=[], JS=[])
    for k in ks:
        a, b = Q[:, k], P[:, k]
        out["PCC"].append(float(np.corrcoef(a, b)[0, 1]) if b.std() > 0 else 0.0); out["SSIM"].append(_ssim(a, b)); out["RMSE"].append(_rmse(a, b)); out["JS"].append(_js(a, b))
    return {k: float(np.mean(v)) for k, v in out.items()} | dict(clones=len(ks))


def accuracy_score(table):
    """table: {arm: scores()}; the benchmark's AS -- mean over metrics of the arm's rank (PCC, SSIM higher better; RMSE, JS lower better),
    scaled so the best arm on every metric gets 1"""
    arms = list(table); n = len(arms); r = np.zeros(n)
    for met, sign in [("PCC", 1), ("SSIM", 1), ("RMSE", -1), ("JS", -1)]:
        v = np.array([sign * table[a][met] for a in arms]); r += np.argsort(np.argsort(v)) / max(n - 1, 1)
    return dict(zip(arms, (r / 4).tolist()))
