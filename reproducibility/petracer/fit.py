"""SpOTr and the compared methods on the PEtracer problem that prepare.py writes; each method's estimate of every spot's composition.

Every dissociated cell has one label: its clone when it has one, otherwise its cell type (the malignant cells without a lineage readout under
"Malignant").
  spotr          SpOTr (spotr.SpOTr) with the lineage tree, alpha 0.2 and beta 1/2; reference signatures from cell2location's regression
                 model on the labelled dissociated cells (common.c2l.signatures), the malignant cells without a clone as a reference type
                 of their own.  The spots' proportions of every reference type and clone (get_proportions), and the map.
  cell2location  the package (common.c2l.deconvolve), every label a reference type, N_cells_per_location the mean number of cells per spot,
                 30,000 epochs; each label's abundance.
  destvi         scvi-tools' CondSCVI and DestVI (common.baselines.destvi), every label a reference type; each label's proportion.
  tangram        Tangram's cell mapping of every dissociated cell (common.baselines.tangram, 1,000 epochs, the spots' true numbers of cells as
                 its density prior); each spot's mapped cells per label.
  scdot          scDOT (common.baselines.scdot) with every label as a cell type; its map's mapped cells per label.

Writes $SPOTR_RESULTS/petracer/<section>/<method>.npz: comp (spots x labels) and labels; for SpOTr also the map (cells x spots, float16)
and the costs of the mapping.  A method whose file exists is not refitted.

Usage: python fit.py [section=M3-4] [spotr] [cell2location] [destvi] [tangram] [scdot]   (environment benchmark_env, GPU; cell2location in
its own environment, see common/paths.py)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, baselines, c2l  # noqa: E402

METHODS = ["spotr", "cell2location", "destvi", "tangram", "scdot"]
ALPHA, BETA = 0.2, 0.5


def load(section):
    import anndata as ad
    d = paths.DATA / "petracer" / section
    return ad.read_h5ad(d / "sc.h5ad"), ad.read_h5ad(d / "st.h5ad")


def labels_of(sc):
    """each dissociated cell's label: its clone, else its cell type"""
    cl = sc.obs["clone"].astype(object); return np.where(cl.notna(), cl.astype(str), sc.obs["cell_type"].astype(str)).astype(str)


def dense(X):
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, float)


def onehot(lab):
    keys = sorted(set(lab.tolist())); return (lab[:, None] == np.array(keys)[None, :]).astype(float), keys


def spotr_model(sc, st, sig):
    """SpOTr set up on the dissociated cells with the lineage tree and the given signatures (DataFrame genes x labels)"""
    import spotr
    spotr.SpOTr.setup_anndata(sc, labels_key="cell_type", clone_key="clone", tree=sc.uns["tree"])
    return spotr.SpOTr(sc, st, spatial_key="spatial", signatures=sig)


def signature_table(sc, out):
    import pandas as pd
    lab = labels_of(sc); S = c2l.signatures(dense(sc.X), lab, out / "c2l_signatures.npz"); keys = sorted(S)
    return pd.DataFrame(np.stack([S[k] for k in keys], 1), index=sc.var_names, columns=keys)


def fit(method, sc, st, out):
    Xc, Y, lab = dense(sc.X), dense(st.X), labels_of(sc)
    if method == "spotr":
        m = spotr_model(sc, st, signature_table(sc, out)); m.train(alpha=ALPHA, beta=BETA); P = m.get_proportions()
        return dict(comp=P.to_numpy(), labels=list(P.columns), phi=m.get_mapping().to_numpy().astype(np.float16),
                    costs=np.array([m.costs_["expression"], m.costs_["structure"]]))
    if method == "cell2location":
        w, types = c2l.deconvolve(Xc, lab, Y, n_cells=float(st.obs["n_cells"].mean()), path=out / "c2l_deconvolution.npz", epochs=30000)
        return dict(comp=w, labels=types)
    if method == "destvi":
        w, names = baselines.destvi(Xc, lab, Y, seed=0); return dict(comp=w, labels=names)
    H, keys = onehot(lab)
    if method == "tangram":
        P = baselines.tangram(Xc, Y, mass=st.obs["n_cells"].to_numpy(float), epochs=1000, seed=0); return dict(comp=P.T @ H, labels=keys)
    if method == "scdot":
        P = baselines.scdot(Xc, lab, Y, seed=0); return dict(comp=P.T @ H, labels=keys)
    raise ValueError(f"unknown method {method}")


def main(args):
    import warnings; warnings.filterwarnings("ignore")
    section = next((a.split("=")[1] for a in args if a.startswith("section=")), "M3-4"); todo = [a for a in args if a in METHODS] or METHODS
    sc, st = load(section); out = paths.RESULTS / "petracer" / section; out.mkdir(parents=True, exist_ok=True)
    for method in todo:
        f = out / f"{method}.npz"
        if f.exists():
            continue
        t0 = time.time(); r = fit(method, sc.copy(), st, out)
        np.savez_compressed(f, **{k: (np.asarray(v, np.float32) if k == "comp" else np.asarray(v)) for k, v in r.items()})
        print(f"{method}: {len(r['labels'])} labels, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
