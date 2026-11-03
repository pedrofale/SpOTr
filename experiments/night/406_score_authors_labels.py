"""Every map of HTAN scored against the authors' own per-spot clones (405_: digitised from Mo et al. 2024 Extended Data Fig. 9g with their clone
table; their clone 1 is our B, clone 2 our A on both sections).

Per map: the B share of the tumour mass per spot; balanced purity over the authors' territories (of the tumour mass a map puts on clone 1's spots,
the share that is B; on clone 2's, the share that is A; their mean -- 0.5 without spatial information); and the AUC of the per-spot B share between
the authors' clone 1 and clone 2 spots (spots with tumour mass).  Maps: full depth (SpOTr's arms, cell2location, exact OT, Tangram, DestVI where
present) and the thinning series (390_).

Usage: python 406_score_authors_labels.py   (benchmark_env; writes spotr_out/authors_microregions/scores.json)
"""
import json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; O = HERE / "spotr_out"; A = O / "authors_microregions"


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def score(P2, clone):
    """P2: spots x (B, A) masses; clone: per spot 'clone_1' / 'clone_2' / ''"""
    from sklearn.metrics import roc_auc_score
    P2 = np.asarray(P2, float); c1, c2 = clone == "clone_1", clone == "clone_2"; tot = P2.sum(1)
    p1 = P2[c1, 0].sum() / max(P2[c1].sum(), 1e-300); p2 = P2[c2, 1].sum() / max(P2[c2].sum(), 1e-300); has = (c1 | c2) & (tot > 1e-9 * tot.max())
    auc = roc_auc_score(c1[has].astype(int), P2[has, 0] / tot[has]) if has.any() else np.nan
    return dict(purity=float((p1 + p2) / 2), purity_clone1=float(p1), purity_clone2=float(p2), auc=float(auc), spots=int(has.sum()))


def labels(sec, key, sb):
    d = pd.read_csv(A / f"{key}_spots.tsv", sep="\t").set_index("barcode").reindex(sb); return d["clone"].fillna("").astype(str).to_numpy()


def main():
    res = {}
    # HT260C1 (A, B1, B2: B = B1 + B2), full depth
    z = np.load(ROOT / "data" / "htan" / "HT260C1-Th1K1Fc2U1Z1Bs1_v2.npz", allow_pickle=True); sb = z["spot_barcodes"].astype(str); cl = labels("HT260C1-Th1K1Fc2U1Z1Bs1", "HT260C1", sb)
    tr = np.where(z["traced"])[0]; c = z["sn_clone"].astype(str)[tr]; lab3 = np.where(c == "A", "A", z["b_sub"].astype(str)[tr]); oh = np.stack([lab3 == s for s in ("B1", "B2", "A")], 1).astype(float)
    two = lambda P: np.stack([P[:, 0] + P[:, 1], P[:, 2]], 1); S = json.load(open(O / "htan_v2" / "scores.json")); ms = {k: np.asarray(v, float) for k, v in S["masses"].items()}
    for k in ("lineagesubnb", "notreesubnb"):
        ms[k] = np.load(O / "htan_v2" / f"spotr_{k}_phi.npy").astype(float).T @ oh
    names = {"lineagesubnb": "SpOTr copy-number tree", "notreesubnb": "SpOTr without tree", "c2l": "cell2location", "exact_ot": "exact OT", "tangram": "Tangram", "destvi": "DestVI"}
    names["destvi"] = "DestVI (first run)"; res["HT260C1 full depth"] = {nm: score(two(ms[k]), cl) for k, nm in names.items() if k in ms}
    fdd = O / "htan_v2" / "destvi_documented.npz"
    if fdd.exists():
        o = np.load(fdd, allow_pickle=True); ty = [str(t) for t in o["types"]]; res["HT260C1 full depth"]["DestVI (documented)"] = score(np.stack([o["w"][:, ty.index("B1")] + o["w"][:, ty.index("B2")], o["w"][:, ty.index("A")]], 1), cl)
    # K3 (B, A), full depth
    m387 = mod("m387", "387_htan_ht268b1.py"); z3, ti, lab = m387.load("K3"); sb3 = z3["spot_barcodes"].astype(str); cl3 = labels("HT268B1-Th1K3Fc2U1Z1Bs1", "K3", sb3); oh2 = np.stack([lab == "B", lab == "A"], 1).astype(float)
    r = {}
    for nm, f in [("SpOTr clone labels", "spotr_clones_phi.npy"), ("SpOTr without tree", "spotr_notree_phi.npy"), ("exact OT", "exact_ot.npy"), ("Tangram", "tangram.npy")]:
        P = np.load(O / "ht268b1" / "K3" / f).astype(float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); r[nm] = score(P.T @ oh2, cl3)
    fc = O / "c2l_package" / "htan_HT268B1K3v2_out.npz"
    if fc.exists():
        o = np.load(fc, allow_pickle=True); ty = [str(t) for t in o["types"]]; r["cell2location"] = score(np.stack([o["w"][:, ty.index("B")], o["w"][:, ty.index("A")]], 1), cl3)
    for nm, fd in [("DestVI (first run)", O / "ht268b1" / "K3" / "destvi.npz"), ("DestVI (documented)", O / "ht268b1" / "K3" / "destvi_documented.npz")]:
        if fd.exists():
            o = np.load(fd, allow_pickle=True); ty = [str(t) for t in o["types"]]; r[nm] = score(np.stack([o["w"][:, ty.index("B")], o["w"][:, ty.index("A")]], 1), cl3)
    res["K3 full depth"] = r
    # thinning series (390_'s maps)
    for sec_key, short, oh_, cl_, arms in [("K3", "K3", oh2, cl3, ["notree", "clones", "clone_tree"]), ("HT260C1", "HT260C1", oh, cl, ["notree", "clones", "clone_tree", "lineage"])]:
        for p_ in ("1.0", "0.3", "0.1", "0.03", "0.01"):
            d = O / "thinning" / short / f"p{p_}"; rr = {}
            for a in arms + ["exact_ot", "tangram"]:
                f = d / (f"spotr_{a}_phi.npy" if a not in ("exact_ot", "tangram") else f"{a}.npy")
                if f.exists():
                    P = np.load(f).astype(float); P = P / np.maximum(P.sum(1, keepdims=True), 1e-300); M = P.T @ oh_
                    rr[a] = score(M if M.shape[1] == 2 else two(M), cl_)
            res[f"{sec_key} thinning p={p_}"] = rr
    json.dump(res, open(A / "scores.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k}: " + "; ".join(f"{m} {s['purity']:.3f} (AUC {s['auc']:.3f})" for m, s in v.items()), flush=True)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    main()
