"""SpOTr's joint engine on HT260C1 v4 (445_: the microenvironment look-alikes dropped, every other tumour nucleus called from RNA + ATAC copy
number, the uncalled rest dropped; no untraced tumour, f = 1).  Arms as 439_ / 441_: no tree; the latest (tumour-part expression cost beta 0.5,
distance matching alpha 0.2 on the within-clone copy-number tree, 419_.within_tree).  Profiles: cell2location signatures from v4's labels
(379_.signatures, tag htan_v4_c2l).  Exp-unmatched-oct12 section 177.

Composition per spot: every microenvironment type's abundance and the tumour abundance split by clone through the map (saved with the map).
Scores as 436_: purity against the copy-number references (B vs A, B1 vs B2; B1b vs B1a only over the v2 B1 nuclei, the ones 414_ split) and
against the authors' clones.

Usage: python 446_htan_v4_run.py   (benchmark_env, GPU; writes spotr_out/full_composition/HT260C1_v4_<arm>.npz and v4_scores.json)
"""
import os, sys, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; OUT = O / "full_composition"; DATA = ROOT / "data" / "htan"; SEC = "HT260C1-Th1K1Fc2U1Z1Bs1"
sys.path.insert(0, str(ROOT / "src"))
import spotr.c2l_gw as M


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def problem():
    from scipy.spatial.distance import pdist, squareform
    m379, m419 = mod("m379", "379_c2l_reference.py"), mod("m419", "419_check_new_variant.py"); m379.OUT = O / "htan_v4"; m379.OUT.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(DATA / f"{SEC}_v4.npz", allow_pickle=True)); tr = z["traced"].astype(bool); ti = np.where(tr)[0]; ct = np.asarray(z["cell_type"]).astype(str)
    lab3 = np.where(z["sn_clone"].astype(str)[ti] == "A", "A", z["b_sub"].astype(str)[ti]); labs = ct.astype(object); labs[ti] = lab3; labs = labs.astype(str)
    Xc = z["Xc_fit"].astype(float); Y = z["Ys_fit"].astype(float); S = m379.signatures(Xc, labs, "htan_v4_c2l")
    keep = sorted(e for e in set(ct.tolist()) if e != "tumour" and (ct == e).sum() > 1); G = Y.shape[1]
    SUBS = ["B1", "B2", "A"]; c3 = np.array([SUBS.index(l) for l in lab3]); d = np.asarray(z["d_cn"], float)
    Dk3 = np.array([[d[np.ix_(c3 == a, c3 == b)].mean() if a != b else 0.0 for b in range(3)] for a in range(3)]); dW = m419.within_tree(d, c3, Dk3)
    xy = np.asarray(z["xy"], float); D = squareform(pdist(xy))
    args = (Y, np.stack([S[e] for e in keep]), np.zeros((1, G)), np.array([1.0]), np.stack([S[l] for l in lab3]), np.zeros(len(ti), int), D, xy)
    return dict(z=z, ti=ti, lab3=lab3, keep=keep, args=args, c3=c3, dW=dW, Xc=Xc)


def scorer(z, ti, lab3):
    m415, m406 = mod("m415", "415_deeper_trees.py"), mod("m406", "406_score_authors_labels.py"); s415 = m415.section("HT260C1")
    hb = dict(zip(s415["z"]["barcodes"].astype(str)[s415["ti"]], s415["new"])); bt = z["barcodes"].astype(str)[ti]
    grp = np.array([hb.get(b, "B1new") if l == "B1" else l for b, l in zip(bt, lab3)]); print({g: int((grp == g).sum()) for g in ("A", "B1a", "B1b", "B1new", "B2")}, flush=True)
    C = s415["contrasts"]; con = {"B1b vs B1a": C["B1b vs B1a"], "B1b vs B1a (parent removed)": C["B1b vs B1a (parent removed)"],
                                  "B vs A": (["B1a", "B1b", "B1new", "B2"], ["A"], C["B vs A"][2]), "B1 vs B2": (["B1a", "B1b", "B1new"], ["B2"], C["B1 vs B2"][2])}
    cl_auth = m406.labels(SEC, "HT260C1", z["spot_barcodes"].astype(str)); oh = np.stack([lab3 == s for s in ("B1", "B2", "A")], 1).astype(float)
    def sc(P):
        P = np.asarray(P, float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); G = {g: P[grp == g].sum(0) for g in ("A", "B1a", "B1b", "B1new", "B2")}
        s = {k: v["purity"] for k, v in m415.gscore(G, dict(contrasts=con)).items()}; W = P.T @ oh
        s["authors' clones"] = m406.score(np.stack([W[:, 0] + W[:, 1], W[:, 2]], 1), cl_auth)["purity"]; return s
    return sc


def main():
    import warnings; warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True); p = problem(); ti, lab3, keep = p["ti"], p["lab3"], p["keep"]; sc = scorer(p["z"], ti, lab3); print("problem ready", flush=True)
    fs = OUT / "v4_scores.json"; res = json.load(open(fs)) if fs.exists() else {}; t0 = time.time()
    for nm, kw in [("notree", dict(structure="compact", alpha_tree=0.0)),
                   ("resid_dm0.2", dict(structure="dm", alpha_tree=0.2, tree_d=p["dW"], cell_cost="resid", cell_metric="sqeuclidean", X_cells=p["Xc"][ti], beta=0.5))]:
        fn = OUT / f"HT260C1_v4_{nm}.npz"
        if fn.exists():
            continue
        r = M.fit(*p["args"], verbose=True, clone=p["c3"], mode="clones", **kw); Ab = np.asarray(M.abund(r["th"]), float); phi = np.asarray(r["phi"], float)
        tum = Ab[:, len(keep)]; share = phi / np.maximum(phi.sum(0, keepdims=True), 1e-300); cols = {e: Ab[:, i] for i, e in enumerate(keep)}
        cols.update({k: share[lab3 == k].sum(0) * tum for k in ("A", "B1", "B2")}); Ln = sorted(cols)
        np.savez_compressed(fn, comp=np.stack([cols[l] for l in Ln], 1).astype(np.float32), labels=np.array(Ln), phi=phi.astype(np.float16))
        res[nm] = sc(phi); json.dump(res, open(fs, "w"), indent=1)
        print(f"v4 {nm} done [{time.time() - t0:.0f}s]: " + json.dumps({k: round(v, 3) for k, v in res[nm].items()}) + f"; tumour cells {tum.sum():.0f}", flush=True)
    print("ALL DONE 446", flush=True)


if __name__ == "__main__":
    main()
