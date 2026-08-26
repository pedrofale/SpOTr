"""Two reference baselines for unmatched clone mapping (eTRACER benchmark).

Both expose the common interface

    run(X_sc, X_st, cell_clone, clones, xy=None, seed=0) -> [n_bins, n_clones]

and use ONLY cell-side clone labels + the two expression matrices.  Neither touches
the phylogeny (C_tree), the spatial graph (C_space), the bin coordinates, or any
spatial clone label.

1. run_ot        Plain entropic (Sinkhorn) optimal transport between cells and bins on
                 the EXPRESSION cost alone -- no Gromov / structural term at all.  This
                 is the alpha = 0 limit of SpOTr, implemented independently here with
                 POT's log-domain Sinkhorn run to convergence (SpOTr uses its own
                 unrolled fixed-iteration JAX solver), so that the alpha = 0 entry in the
                 ablation cannot be an artefact of our solver.  Bin composition is the
                 clone-projected, column-normalised coupling.

2. run_centroid  Nearest-clone-centroid / correlation deconvolution.  Each clone gets a
                 mean expression profile over its cells; each bin is scored against every
                 clone by Pearson correlation across genes after removing the shared
                 average profile from both sides; scores are z-scored per bin and softmaxed.

Defaults were fixed a priori (see notes on each function) rather than tuned against the
barcoded bins, because in this benchmark spatial clone labels are held out.

Standalone: needs only numpy and POT (`import ot`).  No repo imports.
"""
from __future__ import annotations

import numpy as np

__all__ = ["run", "run_ot", "run_centroid"]


# --------------------------------------------------------------------------- helpers
def _onehot(cell_clone, clones) -> np.ndarray:
    """[n_cells, n_clones] one-hot assignment of cells to clones."""
    clones = [str(c) for c in np.asarray(clones).ravel()]
    index = {c: k for k, c in enumerate(clones)}
    cc = np.asarray(cell_clone).ravel().astype(str)
    A = np.zeros((cc.shape[0], len(clones)), dtype=np.float64)
    for i, c in enumerate(cc):
        k = index.get(c, -1)
        if k >= 0:
            A[i, k] = 1.0
    return A


def _row_normalise(P: np.ndarray) -> np.ndarray:
    """Rows to simplex; rows with no mass (or non-finite) fall back to uniform."""
    P = np.asarray(P, dtype=np.float64)
    P = np.where(np.isfinite(P), P, 0.0)
    P = np.maximum(P, 0.0)
    s = P.sum(axis=1, keepdims=True)
    bad = (s[:, 0] <= 0) | ~np.isfinite(s[:, 0])
    P = np.divide(P, np.where(s > 0, s, 1.0))
    if bad.any():
        P[bad] = 1.0 / P.shape[1]
    return P


def _expression_cost(X_sc, X_st, metric: str = "sqeuclidean") -> np.ndarray:
    """[n_cells, n_bins] expression cost, rescaled to max 1 (as SpOTr does)."""
    import ot as pot

    Xs = np.ascontiguousarray(np.asarray(X_sc, dtype=np.float64))
    Xt = np.ascontiguousarray(np.asarray(X_st, dtype=np.float64))
    if metric == "sqeuclidean":
        M = pot.dist(Xs, Xt, metric="sqeuclidean")
    elif metric in ("cosine", "correlation", "euclidean"):
        M = pot.dist(Xs, Xt, metric=metric)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    M = np.asarray(M, dtype=np.float64)
    mx = float(np.nanmax(M))
    if not np.isfinite(mx) or mx <= 0:
        mx = 1.0
    return M / mx


# --------------------------------------------------------------------------- 1. plain OT
def run_ot(
    X_sc,
    X_st,
    cell_clone,
    clones,
    xy=None,
    seed: int = 0,
    reg: float = 0.02,
    metric: str = "sqeuclidean",
    numItermax: int = 5000,
    stopThr: float = 1e-9,
    verbose: bool = False,
):
    """Entropic OT on expression cost alone -> per-bin clone composition.

    Marginals are uniform on both sides and the cost is the max-normalised squared
    euclidean distance in HVG space, i.e. exactly the setting SpOTr's expression term
    uses; reg = 0.02 matches SpOTr's eps.  The solver is POT's log-domain Sinkhorn run
    to convergence, which is deliberately a DIFFERENT implementation from SpOTr's
    unrolled fixed-T solver.

    `xy` and `seed` are accepted for interface compatibility and unused (the method is
    deterministic and spatially blind).
    """
    import ot as pot

    A = _onehot(cell_clone, clones)
    n, m = A.shape[0], np.asarray(X_st).shape[0]
    M = _expression_cost(X_sc, X_st, metric=metric)
    a = np.full(n, 1.0 / n, dtype=np.float64)
    b = np.full(m, 1.0 / m, dtype=np.float64)

    T = pot.sinkhorn(
        a, b, M, reg,
        method="sinkhorn_log",
        numItermax=numItermax,
        stopThr=stopThr,
        verbose=verbose,
    )
    T = np.asarray(T, dtype=np.float64)
    # each bin's clone composition = its column of the coupling, projected onto clones
    return _row_normalise(T.T @ A)


# --------------------------------------------------------------------------- 2. centroid
def run_centroid(
    X_sc,
    X_st,
    cell_clone,
    clones,
    xy=None,
    seed: int = 0,
    tau: float = 1.0,
    center_genes: bool = True,
):
    """Correlation-to-clone-centroid deconvolution -> per-bin clone composition.

    Clone centroid = mean log-normalised profile of its cells.  If `center_genes`
    (default) the shared average profile is removed per gene from BOTH sides -- clone
    centroids minus the mean centroid, bins minus the mean bin -- so the correlation
    measures clone-specific deviation rather than the dominant "how expressed is this
    gene at all" component, and the platform shift between scRNA and Stereo-seq is
    largely absorbed.  Scores are Pearson correlations across genes, z-scored within a
    bin (which makes tau scale-free), then softmaxed at temperature `tau`.

    tau only sharpens/flattens the composition; the argmax clone is invariant to it.
    Clones with no cells get score -inf (they can never be assigned).

    `xy` and `seed` are accepted for interface compatibility and unused.
    """
    A = _onehot(cell_clone, clones)                      # [n, K]
    Xs = np.asarray(X_sc, dtype=np.float64)
    Xt = np.asarray(X_st, dtype=np.float64)
    K = A.shape[1]

    counts = A.sum(axis=0)                               # [K]
    present = counts > 0
    cent = (A.T @ Xs) / np.maximum(counts, 1.0)[:, None]  # [K, G] clone mean profiles

    if center_genes:
        mu_c = cent[present].mean(axis=0, keepdims=True)   # mean over PRESENT clones
        cent = cent - mu_c
        Xt = Xt - Xt.mean(axis=0, keepdims=True)

    def _zrows(Z):
        Z = Z - Z.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(Z, axis=1, keepdims=True)
        return Z / np.maximum(nrm, 1e-12)

    R = _zrows(Xt) @ _zrows(cent).T                      # [m, K] Pearson correlation
    R[:, ~present] = -np.inf

    # z-score across clones within each bin, then softmax
    finite = R[:, present]
    mu = finite.mean(axis=1, keepdims=True)
    sd = finite.std(axis=1, keepdims=True)
    Z = np.full_like(R, -np.inf)
    Z[:, present] = (finite - mu) / np.maximum(sd, 1e-12)
    Z = Z / max(float(tau), 1e-12)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    P = np.exp(Z)
    return _row_normalise(P)


# --------------------------------------------------------------------------- default
def run(X_sc, X_st, cell_clone, clones, xy=None, seed: int = 0, **kwargs):
    """Common interface; defaults to the plain entropic OT baseline."""
    return run_ot(X_sc, X_st, cell_clone, clones, xy=xy, seed=seed, **kwargs)


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":  # pragma: no cover
    import time
    from pathlib import Path

    npz = Path(__file__).resolve().parent.parent / "etracer_LE_full.npz"
    d = np.load(npz, allow_pickle=True)
    X_sc, X_st = np.asarray(d["X_sc"]), np.asarray(d["X_st"])
    cell_clone = np.asarray(d["cell_clone"]).astype(str)
    bin_clone = np.asarray(d["bin_clone"]).astype(str)
    clones = np.asarray(d["clones"]).astype(str)
    xy, is_lab = np.asarray(d["xy"]), np.asarray(d["is_labelled"], bool)

    cl = list(clones)
    true_k = np.array([cl.index(c) if c in cl else -1 for c in bin_clone])
    maj = np.bincount(true_k[is_lab], minlength=len(clones)).argmax()
    print(f"majority-clone reference acc = {(true_k[is_lab] == maj).mean():.4f}"
          f"  ({int(is_lab.sum())} barcoded bins of {len(is_lab)})")

    for name, fn in (("run_ot", run_ot), ("run_centroid", run_centroid)):
        t0 = time.time()
        P = fn(X_sc, X_st, cell_clone, clones, xy=xy, seed=0)
        dt = time.time() - t0
        pred = P.argmax(axis=1)
        acc = float((pred[is_lab] == true_k[is_lab]).mean())
        print(f"{name:13s} shape={P.shape} rowsum=[{P.sum(1).min():.6f},{P.sum(1).max():.6f}] "
              f"wall={dt:.1f}s | barcoded-bin argmax acc = {acc:.4f} "
              f"| distinct predicted clones = {len(np.unique(pred))}")
