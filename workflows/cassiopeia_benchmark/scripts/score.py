"""Scores of every method on one simulated problem (run in the spotr environment).

clone_tv          per-spot subclone error: the mean over spots holding tumour cells of the total variation distance between the true clone composition
                  of the section's tumour cells in the spot and the composition the method places there (true clones throughout: maps of cells
                  use each cell's true clone; cell2location's abundances and DestVI's proportions per reference label are converted through the labelled cells' true clones)
subclade_tv_l1-3  the same at the resolution of the sub-clades of each true clone's subtree (levels 1-3), against the dissociated cells' own
                  true spots (maps of cells only)
spread_ratio      SpOTr: the median over clones of the learned spread over the spread of the clone's cells' true positions

Usage: python score.py <params.json> <problem.npz> <trees.npz> <spotr.npz> <out.json> [c2l.npz] [destvi.npz] [exact_ot.npz] [tangram.npz]
"""
import sys, json
from pathlib import Path
import numpy as np


def tv(P, Ncl):
    has = Ncl.sum(1) > 0; T = Ncl[has] / Ncl[has].sum(1, keepdims=True); Q = P[has] / np.maximum(P[has].sum(1, keepdims=True), 1e-300)
    return float(0.5 * np.abs(T - Q).sum(1).mean())


def main(pfile, prob, trees, spotr, out, *extra):
    import spotr.c2l_gw as M
    q, t, s = np.load(prob), np.load(trees), np.load(spotr); rec = json.load(open(spotr.replace(".npz", ".json")))["record"]
    K = int(q["K"]); Ncl = q["Ncl"].astype(float); tr = q["traced"]; ct = q["clone"][tr]; onehot = np.eye(K)[ct]; xyt = q["xy_true"][tr]; spots = q["spots"]
    major = np.where(ct < K - 1, ct, -1); lev = M.nested_levels(t["D_true"], major, K - 1); used = s["clone_used"]; scores = {}
    def cellmap(name, phi):
        phi = np.asarray(phi, float); r = dict(clone_tv=tv(phi.T @ onehot, Ncl)); r.update(M.subclade_errors(phi, lev, xyt, spots)); scores[name] = r
    for k in s.files:
        if k.startswith("phi_"):
            a = k[4:]; cellmap(a, s[k])
            if "weights" in rec.get(a, {}):
                l = np.sqrt(np.asarray(rec[a]["weights"]["s2"], float)); tru = np.array([np.sqrt(((xyt[used == c] - xyt[used == c].mean(0)) ** 2).sum(1).mean() / 2.0) for c in range(len(l))])
                ok = tru > 0; scores[a]["spread_ratio"] = float(np.median(l[ok] / tru[ok]))
    for f in extra:
        name = Path(f).stem
        if name in ("c2l", "destvi"):                                                              # abundances / proportions per reference label
            z = np.load(f, allow_pickle=True); types = [str(x) for x in z["types"]]; w = z["w"].astype(float); lab = np.array([f"c{c}" if c >= 0 else "pooled" for c in used])
            P = np.zeros((len(w), K))
            for L in np.unique(lab):
                if L in types:
                    P += w[:, [types.index(L)]] * onehot[lab == L].mean(0)[None, :]
            scores[name] = dict(clone_tv=tv(P, Ncl))
        else:
            P = np.load(f)["P"].astype(float)[np.where(tr)[0]]; cellmap(name, P / np.maximum(P.sum(1, keepdims=True), 1e-300))
    json.dump(scores, open(out, "w"), indent=1); print({k: round(v["clone_tv"], 3) for k, v in scores.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
