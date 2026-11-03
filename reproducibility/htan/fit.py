"""SpOTr and cell2location on HT260C1 (environment benchmark_env, GPU; cell2location runs in its own environment through common/c2l.py).

Labels: every nucleus's cell type, every cancer nucleus's clone (A, B1, B2).  Every cancer nucleus has a clone, so the deconvolution has no
type for cancer cells without one.  B1's sub-lineages are not given to any method.

SpOTr, with the Methods' settings: signatures from cell2location's regression model on these labels and every fitted gene; the reference cell
types those with more than one nucleus; the tree: Ward linkage within each clone on the nuclei's copy-number distances, scaled so that its
deepest split equals the shallowest distance between clones, the clones joined by their mean distances (copynumber.clone_tree); beta 1/2.
Two fits: alpha 0.2 (Fig. 4, Fig. S3) and alpha 0, the expression alone (Fig. S3).
cell2location: the package with its documented workflow on the same labels, 10 cells per spot, 30,000 epochs.

Writes to $SPOTR_RESULTS/htan/: spotr_alpha0.2.npz and spotr_alpha0.0.npz (phi: the map, nuclei x spots; proportions: spots x labels; labels),
c2l.npz (cell2location's abundances, spots x labels; types), c2l_signatures.npz.

Usage: python fit.py [spotr] [c2l]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); from common import paths, c2l  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent)); import copynumber as cn  # noqa: E402

ALPHAS, BETA = (0.2, 0.0), 0.5


def data():
    import anndata as ad
    sc_ = ad.read_h5ad(paths.DATA / "htan" / "sc.h5ad"); st_ = ad.read_h5ad(paths.DATA / "htan" / "st.h5ad")
    ct = sc_.obs["cell_type"].astype(str).to_numpy(); clone = sc_.obs["clone"]
    labels = np.where(clone.notna(), clone.astype(str), ct)
    return sc_, st_, labels


def tree(sc_):
    """the tree SpOTr is given, as a DataFrame of distances between the cancer nuclei"""
    m = sc_.obs["clone"].notna().to_numpy(); names = sc_.obs_names[m]; cl = sc_.obs["clone"][m].astype(str).to_numpy()
    k = np.array([sorted(set(cl)).index(c) for c in cl]); d = cn.profile_distances(sc_.obsm["cn_profile"][m])
    return pd.DataFrame(cn.clone_tree(d, k), index=names, columns=names)


def signatures(sc_, labels):
    X = sc_.X.toarray() if hasattr(sc_.X, "toarray") else np.asarray(sc_.X)
    S = c2l.signatures(X, labels, paths.out("htan", "c2l_signatures.npz")); return pd.DataFrame(S, index=sc_.var_names)


def fit_spotr(signature_frame=None):
    import spotr
    sc_, st_, labels = data(); S = signatures(sc_, labels) if signature_frame is None else signature_frame
    n = pd.Series(sc_.obs["cell_type"].astype(str)).value_counts(); ref = sc_.obs["cell_type"].astype(str).map(n).to_numpy() > 1
    sc_ = sc_[ref].copy()                                                                     # cell types with a single nucleus left out
    spotr.SpOTr.setup_anndata(sc_, labels_key="cell_type", clone_key="clone", tree=tree(sc_))
    model = spotr.SpOTr(sc_, st_, spatial_key="spatial", signatures=S)
    for a in ALPHAS:
        f = paths.out("htan", f"spotr_alpha{a}.npz")
        if f.exists():
            continue
        model.train(alpha=a, beta=BETA); P = model.get_proportions()
        np.savez_compressed(f, phi=model.get_mapping().to_numpy().astype(np.float16), proportions=P.to_numpy(np.float32), labels=np.array(P.columns),
                            cells=np.array(sc_.obs_names[model.setup["mapped"]]))
        print(f"SpOTr alpha {a}: written {f}", flush=True)
    return model


def fit_c2l():
    sc_, st_, labels = data(); X = sc_.X.toarray(); Y = st_.X.toarray()
    w, types = c2l.deconvolve(X, labels, Y, n_cells=10.0, path=paths.out("htan", "c2l.npz"), epochs=30000)
    print(f"cell2location: {w.shape[0]} spots x {len(types)} labels", flush=True)


if __name__ == "__main__":
    import os, warnings; warnings.filterwarnings("ignore"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for step in (sys.argv[1:] or ["spotr", "c2l"]):
        {"spotr": fit_spotr, "c2l": fit_c2l}[step]()
