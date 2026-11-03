"""SpOTr's lineage-resolution readout on one simulated problem (run in the spotr environment), as experiments/night/351_.

Every split of the tree given to SpOTr (average linkage on its distances) whose daughters each hold >= 20 traced cells is tested on a map that
does not know it -- below the clone level the clone-label map, above it the map without the tree: the energy distance between the daughters'
mapped distributions against the same map with the node's cells permuted between the daughters (199 permutations); resolved if p < 0.05 and
the parent split is resolved.  The same test on the cells' true positions says which splits are truly separated.

Usage: python readout.py <params.json> <problem.npz> <trees.npz> <spotr.npz> <out.json>
"""
import sys, json
import numpy as np
MIN, N_PERM, ALPHA = 20, 199, 0.05


def splits(d):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    n = len(d); Z = linkage(squareform(np.asarray(d, float), checks=False), method="average"); mem = {i: [i] for i in range(n)}; out = []
    for j, (a, b, _, _) in enumerate(Z):
        L, R = mem.pop(int(a)), mem.pop(int(b)); out.append((np.array(L), np.array(R))); mem[n + j] = L + R
    return out


def energy_map(P, a, b, D, rng):
    idx = np.concatenate([a, b]); na = len(a); Q = P[idx]; m = len(idx); Ws = np.zeros((N_PERM + 1, m)); Ws[0, :na] = 1
    for i in range(1, N_PERM + 1):
        Ws[i, rng.permutation(m)[:na]] = 1
    A = Ws @ Q / na; B = (1 - Ws) @ Q / (m - na); AD, BD = A @ D, B @ D; e = 2 * (AD * B).sum(1) - (AD * A).sum(1) - (BD * B).sum(1)
    return float((1 + (e[1:] >= e[0]).sum()) / (N_PERM + 1))


def energy_true(x, a, b, rng):
    from scipy.spatial.distance import cdist
    idx = np.concatenate([a, b]); na = len(a); Dc = cdist(x[idx], x[idx]); m = len(idx)
    def st(sel):
        return 2 * Dc[np.ix_(sel, ~sel)].mean() - Dc[np.ix_(sel, sel)].mean() - Dc[np.ix_(~sel, ~sel)].mean()
    s0 = np.zeros(m, bool); s0[:na] = True; obs = st(s0); null = []
    for _ in range(N_PERM):
        s = np.zeros(m, bool); s[rng.permutation(m)[:na]] = True; null.append(st(s))
    return float((1 + (np.array(null) >= obs).sum()) / (N_PERM + 1))


def main(pfile, prob, trees, spotr, out):
    from scipy.spatial.distance import pdist, squareform
    p = json.load(open(pfile)); q, t, s = np.load(prob), np.load(trees), np.load(spotr); d = t["D_rec"] if p["tree"] == "reconstructed" else t["D_true"]
    clone = s["clone_used"]; xyt = q["xy_true"][q["traced"]]; D = squareform(pdist(q["spots"].astype(float))); rng = np.random.default_rng(0)
    P0, Pc = s["phi_notree"].astype(float), s["phi_clones"].astype(float); rows = []; sp = [x for x in splits(d) if min(len(x[0]), len(x[1])) >= MIN]
    for a, b in sp:
        cl = set(clone[np.concatenate([a, b])].tolist()); within = len(cl) == 1 and -1 not in cl
        rows.append(dict(n=len(a) + len(b), within=within, p=energy_map(Pc if within else P0, a, b, D, rng), p_true=energy_true(xyt, a, b, rng)))
    sets = [set(np.concatenate(x).tolist()) for x in sp]; res = [False] * len(rows)
    for i in np.argsort([-len(x) for x in sets]):
        par = [j for j in range(len(sets)) if j != i and sets[i] < sets[j]]; pj = min(par, key=lambda j: len(sets[j])) if par else None
        res[i] = rows[i]["p"] < ALPHA and (pj is None or res[pj])
    tp = sum(r_ and x["p_true"] < ALPHA for r_, x in zip(res, rows)); fp = sum(r_ and x["p_true"] >= ALPHA for r_, x in zip(res, rows)); fn = sum((not r_) and x["p_true"] < ALPHA for r_, x in zip(res, rows))
    o = dict(splits=len(rows), resolved=int(sum(res)), within=int(sum(x["within"] for x in rows)), within_resolved=int(sum(r_ and x["within"] for r_, x in zip(res, rows))),
             tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
    json.dump(o, open(out, "w"), indent=1); print(o)


if __name__ == "__main__":
    main(*sys.argv[1:6])
