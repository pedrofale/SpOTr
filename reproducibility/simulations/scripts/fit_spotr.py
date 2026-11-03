"""SpOTr on one simulated tumor, through the package's interface (environment benchmark_env; GPU).

The dissociated cells as AnnData: their counts, each cell's type ("tumour", or its non-tumor type me<e>) and, for the tumor cells whose lineage
is read, their clone (c<k>, or "pooled" for the smaller clades); the tree as the true distances between those cells.  The spots: their counts
and positions.  The signatures: cell2location's regression on the dissociated cells, one per label (each non-tumor type, each clone, and the
tumor cells without a lineage readout), cached in the seed folder as c2l_signatures.npz.

Modes (each writes <out>.npz and <out>.json):
  fit      SpOTr at --alpha and --beta: the map (phi, traced cells x spots), each spot's proportions (proportions, spots x labels) and the
           expression and structural costs.
  grid     each alpha of --alphas at --beta: phi_<alpha>, the costs, and the elbow (the alpha that minimises the sum of the two costs, each
           normalised between the smallest and the largest alpha).
  weights  at --alpha and --beta, with the default clade weights (phi_default) and with weight 0 on the dispersed clones (phi_weights).
  tuning   each setting of --alphas x --betas: phi_<alpha>_<beta>, and each setting's mean per-clone Pearson correlation with the truth.

Usage: python fit_spotr.py <mode> <seed folder> <out.npz> [--alpha 0.2] [--beta 0.5] [--alphas 0,0.1,...] [--betas 0.25,...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths, c2l, metrics  # noqa: E402  (paths puts the package on the path)
import world  # noqa: E402


def model(d):
    """the SpOTr model of the seed folder d, and its problem"""
    import anndata as ad, pandas as pd, spotr
    q, t = np.load(d / "problem.npz", allow_pickle=True), np.load(d / "trees.npz"); lab = world.labels(q, t); tr = q["traced"].astype(bool)
    names = np.array([f"cell{i}" for i in range(len(lab))]); genes = [f"g{j}" for j in range(q["Xc"].shape[1])]
    ctype = np.where(q["kind"] == 0, "tumour", lab)
    sc = ad.AnnData(q["Xc"].astype(np.float32), obs=pd.DataFrame({"cell_type": ctype, "clone": np.where(tr, lab, None)}, index=names),
                    var=pd.DataFrame(index=genes))
    st = ad.AnnData(q["Y"].astype(np.float32), obs=pd.DataFrame(index=[f"spot{i}" for i in range(len(q["Y"]))]), var=pd.DataFrame(index=genes))
    st.obsm["spatial"] = q["spots"].astype(float)
    tree = pd.DataFrame(t["D_true"], index=names[t["rows"]], columns=names[t["rows"]])
    spotr.SpOTr.setup_anndata(sc, labels_key="cell_type", clone_key="clone", tree=tree)
    S = c2l.signatures(q["Xc"], np.where(tr, lab, ctype), d / "c2l_signatures.npz")
    return spotr.SpOTr(sc, st, signatures=pd.DataFrame(S, index=genes)), q


def clone_pcc(phi, q):
    """the mean over clones of the Pearson correlation between each clone's true and mapped share of the spots"""
    tr = q["traced"].astype(bool); pc = metrics.per_clone(np.asarray(phi, float).T @ np.eye(int(q["K"]))[q["clone"][tr]], q["Ncl"].astype(float))["PCC"]
    return float(np.mean(pc))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "grid", "weights", "tuning"]); ap.add_argument("folder"); ap.add_argument("out")
    ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.4,0.95"); ap.add_argument("--betas", default="0.25,0.5,0.75"); a = ap.parse_args()
    d = Path(a.folder); m, q = model(d); out = Path(a.out); arrays, info = {}, dict(mode=a.mode)
    f16 = lambda: m.get_mapping().to_numpy().astype(np.float16)
    if a.mode == "fit":
        m.train(alpha=a.alpha, beta=a.beta); P = m.get_proportions()
        arrays = dict(phi=f16(), proportions=P.to_numpy(np.float32), labels=np.array(P.columns)); info.update(alpha=a.alpha, beta=a.beta, costs=m.costs_)
    elif a.mode == "grid":
        alphas = [float(x) for x in a.alphas.split(",")]; costs = {}
        for al in alphas:
            m.train(alpha=al, beta=a.beta); arrays[f"phi_{al}"] = f16(); costs[al] = m.costs_
        E = np.array([costs[al]["expression"] for al in alphas]); S = np.array([costs[al]["structure"] for al in alphas])
        e = (E - E[0]) / max(E[-1] - E[0], 1e-30); s = (S - S[-1]) / max(S[0] - S[-1], 1e-30)
        info.update(alphas=alphas, beta=a.beta, costs={str(k): v for k, v in costs.items()}, elbow=alphas[int(np.argmin(e + s))])
    elif a.mode == "weights":
        dispersed = [int(k) for k in np.load(d / "sim.npz")["migrating"]]
        m.train(alpha=a.alpha, beta=a.beta); arrays["phi_default"] = f16()
        m.train(alpha=a.alpha, beta=a.beta, clade_weights={f"c{k}": 0.0 for k in dispersed}); arrays["phi_weights"] = f16()
        arrays["dispersed"] = np.array(dispersed, int); info.update(alpha=a.alpha, beta=a.beta, dispersed=dispersed)
    else:
        score = {}
        for b in [float(x) for x in a.betas.split(",")]:
            for al in [float(x) for x in a.alphas.split(",")]:
                m.train(alpha=al, beta=b); arrays[f"phi_{al}_{b}"] = f16(); score[f"alpha {al} beta {b}"] = clone_pcc(m.get_mapping().to_numpy(), q)
                print(f"alpha {al} beta {b}: mean per-clone PCC {score[f'alpha {al} beta {b}']:.4f}", flush=True)
        info.update(clone_PCC=score, best=max(score, key=score.get))
    np.savez_compressed(out, **arrays); json.dump(info, open(out.with_suffix(".json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in info.items() if k not in ("costs",)}, default=float)[:400])


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
