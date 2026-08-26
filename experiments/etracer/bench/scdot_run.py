"""scDOT adapted to the eTRACER unmatched clone-mapping benchmark.

scDOT (Nguyen et al.) is an expression-only optimal-transport deconvolution method:
it (a) solves a non-negative least squares problem of each spatial spot against a
cell-type signature/marker matrix, and (b) learns an entropy-regularised OT coupling
between spots and single cells whose cost matrix is trained to agree with the NNLS
deconvolution.  Here the "cell types" are the 45 lineage clones, so the deconvolution
output is exactly the per-bin clone composition the benchmark scores.

Upstream implementation used verbatim from
``benchmark/scDOT/{scdot.py, ddn.py}``.  Only the glue is local:

  * a 4-attribute AnnData shim (``train_scdot`` touches nothing but ``.X``,
    ``.obsm['cell_type']`` and ``.uns['markers']``);
  * construction of the signature matrix from the clone means of ``X_sc``
    (``signature="mean"``, the default) or from per-clone one-vs-rest Wilcoxon
    marker genes (``signature="markers"``, the recipe used by this repo's
    BCSA2/BCSA3 scDOT baselines);
  * collapsing scDOT's spot x cell mapping to spot x clone proportions.

Interface
---------
    run(X_sc, X_st, cell_clone, clones, xy=None, seed=0) -> [n_bins, n_clones]

Sees only the two expression matrices and the cell-side clone labels.  It never
touches C_tree, C_space, xy, bin_clone or is_labelled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCDOT_DIR = Path("/home/pedro/projects/spotr/phylospatial_mapping/benchmark/scDOT")

_UPSTREAM = None


def _upstream():
    """Import benchmark/scDOT/scdot.py once, standalone (no package install)."""
    global _UPSTREAM
    if _UPSTREAM is None:
        if str(SCDOT_DIR) not in sys.path:
            sys.path.insert(0, str(SCDOT_DIR))
        spec = importlib.util.spec_from_file_location(
            "scdot_upstream", str(SCDOT_DIR / "scdot.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _UPSTREAM = mod
    return _UPSTREAM


class _AD:
    """Minimal AnnData stand-in: scDOT only reads .X, .obsm and .uns."""

    def __init__(self, X):
        self.X = np.asarray(X, dtype=float)
        self.obsm = {}
        self.uns = {}
        self.obs = {}

    @property
    def shape(self):
        return self.X.shape


def _one_hot(cell_clone, clones):
    index = {str(c): j for j, c in enumerate(clones)}
    oh = np.zeros((len(cell_clone), len(clones)), dtype=float)
    for i, c in enumerate(cell_clone):
        j = index.get(str(c))
        if j is not None:
            oh[i, j] = 1.0
    return oh


def _clone_means(X_sc, oh):
    """Signature matrix [n_clones, n_genes] = mean expression per clone."""
    n = oh.sum(0)
    sig = oh.T @ X_sc
    nz = n > 0
    sig[nz] /= n[nz, None]
    return sig


def _wilcoxon_markers(X_sc, oh, n_top=10):
    """Per-clone one-vs-rest Mann-Whitney z-scores -> binary top-n marker matrix + 1.

    Reproduces the signature this repo's BCSA scDOT baselines feed to scDOT
    (scanpy rank_genes_groups(method='wilcoxon') top-10 indicator, +1).
    """
    from scipy.stats import rankdata

    ranks = rankdata(X_sc, axis=0)               # [n_cells, n_genes]
    total = ranks.sum(0)
    n_all = X_sc.shape[0]
    markers = np.zeros((oh.shape[1], X_sc.shape[1]), dtype=float)
    for j in range(oh.shape[1]):
        m = oh[:, j] > 0
        n1 = int(m.sum())
        n2 = n_all - n1
        if n1 == 0 or n2 == 0:
            continue
        u = ranks[m].sum(0) - n1 * (n1 + 1) / 2.0
        z = (u - n1 * n2 / 2.0) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        top = np.argsort(-z)[:n_top]
        markers[j, top] = 1.0
    _ = total
    return markers + 1.0


def run(
    X_sc,
    X_st,
    cell_clone,
    clones,
    xy=None,
    seed=0,
    signature="mean",
    iters=30,
    lr=0.1,
    mode="nnls",
    verbose=False,
):
    """Predict per-bin clone composition with scDOT.

    Parameters
    ----------
    signature : {"mean", "markers"}
        Cell-type profile handed to scDOT's NNLS step.  "mean" = clone means of
        X_sc; "markers" = top-10 Wilcoxon marker indicator (+1) per clone.
    mode : {"nnls", "ot"}
        "nnls" returns what upstream ``train_scdot`` returns (its NNLS-derived
        spot x cell mapping, collapsed to clones) -- the faithful scDOT output.
        "ot" instead collapses the learned OT coupling itself.
    """
    import torch

    up = _upstream()
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_sc = np.ascontiguousarray(np.asarray(X_sc, dtype=float))
    X_st = np.ascontiguousarray(np.asarray(X_st, dtype=float))
    oh = _one_hot(cell_clone, clones)

    if signature == "mean":
        sig = _clone_means(X_sc, oh)
    elif signature == "markers":
        sig = _wilcoxon_markers(X_sc, oh)
    else:
        raise ValueError(f"unknown signature {signature!r}")

    sc_ad = _AD(X_sc)
    sc_ad.obsm["cell_type"] = oh
    st_ad = _AD(X_st)
    st_ad.uns["markers"] = np.asarray(sig, dtype=float)

    model, P = up.train_scdot(sc_ad, st_ad, lr=lr, iters=iters)

    if mode == "ot":
        with torch.no_grad():
            coupling = model.OT(model.M)             # [n_bins, n_cells]
            coupling = coupling / coupling.sum(0)
        P = coupling.numpy()
    elif mode != "nnls":
        raise ValueError(f"unknown mode {mode!r}")

    # spot x cell mapping -> spot x clone composition
    prop = np.nan_to_num(np.asarray(P, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    prop = np.clip(prop, 0.0, None) @ oh
    rs = prop.sum(1, keepdims=True)
    dead = rs[:, 0] <= 0
    if dead.any():                                   # bin got no mass at all
        prop[dead] = 1.0 / prop.shape[1]
        rs = prop.sum(1, keepdims=True)
    prop = prop / rs
    if verbose:
        print(f"[scdot_run] signature={signature} mode={mode} -> {prop.shape}")
    return prop


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/pedro/projects/spotr/phylospatial_mapping/"
                                     "experiments/etracer/etracer_LE_full.npz")
    ap.add_argument("--signature", default="mean", choices=["mean", "markers"])
    ap.add_argument("--mode", default="nnls", choices=["nnls", "ot"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    t0 = time.time()
    prop = run(d["X_sc"], d["X_st"], d["cell_clone"], d["clones"],
               xy=d["xy"], seed=0, signature=a.signature, mode=a.mode,
               iters=a.iters, verbose=True)
    wall = time.time() - t0

    clones = [str(c) for c in d["clones"]]
    lab = d["is_labelled"]
    truth = np.array([str(c) for c in d["bin_clone"]])
    pred = np.array(clones)[prop.argmax(1)]
    acc = float((pred[lab] == truth[lab]).mean())
    maj = float(max((truth[lab] == c).mean() for c in clones))
    print(f"signature={a.signature} mode={a.mode} iters={a.iters}")
    print(f"wall_time_s={wall:.1f}  shape={prop.shape}  rowsum={prop.sum(1).min():.6f}"
          f"..{prop.sum(1).max():.6f}")
    print(f"barcoded-bin argmax accuracy = {acc:.4f}  (majority baseline {maj:.4f}, "
          f"n={int(lab.sum())})")
    print(f"distinct argmax clones = {len(set(pred))}/{len(clones)}")
    if a.out:
        np.save(a.out, prop)
