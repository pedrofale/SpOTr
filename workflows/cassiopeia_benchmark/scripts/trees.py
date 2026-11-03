"""The trees given to SpOTr (run in the cassiopeia environment): distances among the traced dissociated cells in the true tree and in the tree
Cassiopeia reconstructs from their barcodes (VanillaGreedySolver; branch length = sites edited along the branch, after Cassiopeia's ancestral
reconstruction), and each tree's clones by the truth's own rule: the tree cut into its first clone_founders lineages, or n_clones of them (average linkage on its
distances), lineages below clone_min of the cells pooled (-1); with the true tree, the true clones.

Usage: python trees.py <params.json> <sim.npz> <problem.npz> <out.npz>
"""
import sys, json
import numpy as np


def dist_from_parents(parent, depth, leaf_rows, n):
    """patristic distances among the leaves listed in leaf_rows (node index -> row), from parent pointers in pre-order and node depths"""
    kids = [[] for _ in range(len(parent))]
    for v in range(len(parent)):
        if parent[v] >= 0:
            kids[parent[v]].append(v)
    D = np.zeros((n, n)); below = [None] * len(parent)
    for v in range(len(parent) - 1, -1, -1):                                                        # children before parents (reverse pre-order)
        lists = [below[c] for c in kids[v] if below[c] is not None and len(below[c])]
        if v in leaf_rows:
            lists.append(np.array([leaf_rows[v]]))
        for a in range(len(lists)):
            for b in range(a + 1, len(lists)):
                La, Lb = lists[a], lists[b]; d = depth_leaf[La][:, None] + depth_leaf[Lb][None, :] - 2.0 * depth[v]
                D[np.ix_(La, Lb)] = d; D[np.ix_(Lb, La)] = d.T
        below[v] = np.concatenate(lists) if lists else None
        for c in kids[v]:
            below[c] = None
    return D


def clones_of(D, F, mmin):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=max(F, 1), criterion="maxclust") - 1
    u, cnt = np.unique(lab, return_counts=True); big = u[cnt >= mmin]; big = big[np.argsort(-cnt[np.isin(u, big)])]
    out = np.full(len(lab), -1)
    for k, b in enumerate(big):
        out[lab == b] = k
    return out


def main(pfile, sim, prob, out):
    global depth_leaf
    import pandas as pd, cassiopeia as cas
    p = json.load(open(pfile)); z = np.load(sim, allow_pickle=True); q = np.load(prob)
    rows = np.where(q["traced"])[0]; leaf = q["leaf"][rows]; n = len(rows); K = int(z["K"]) - 1; F = int(p["n_clones"]) if "n_clones" in p else int(p["clone_founders"]); mmin = float(p["clone_min"]) * n
    # the true tree
    lrow = {int(z["leaf_node"][l]): i for i, l in enumerate(leaf)}; depth_leaf = z["time"][z["leaf_node"][leaf]]
    D_true = dist_from_parents(z["parent"], z["time"], lrow, n)
    # the reconstructed tree, from the traced cells' barcodes
    names = [f"c{i}" for i in range(n)]; cm = pd.DataFrame(z["characters"][leaf], index=names)
    tree = cas.data.CassiopeiaTree(character_matrix=cm, missing_state_indicator=-1); cas.solver.VanillaGreedySolver().solve(tree)
    tree.reconstruct_ancestral_characters(); nodes = list(tree.depth_first_traverse_nodes(source=tree.root, postorder=False)); idx = {v: i for i, v in enumerate(nodes)}
    par = np.array([idx[tree.parent(v)] if v != tree.root else -1 for v in nodes]); dep = np.zeros(len(nodes))
    for v in nodes[1:]:
        a, b = np.array(tree.get_character_states(tree.parent(v))), np.array(tree.get_character_states(v))
        dep[idx[v]] = dep[idx[tree.parent(v)]] + float(np.sum((a != b) & (a != -1) & (b != -1)))
    rrow = {idx[nm]: i for i, nm in enumerate(names)}; depth_leaf = dep[[idx[nm] for nm in names]]
    D_rec = dist_from_parents(par, dep, rrow, n)
    ct = q["clone"][rows]; true_clones = np.where(ct < K, ct, -1)                                        # with the true tree, the true clones themselves
    np.savez_compressed(out, rows=rows, D_true=D_true, D_rec=D_rec, clone_true_tree=true_clones, clone_rec_tree=clones_of(D_rec, F, mmin), K=K)
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score
    ct = q["clone"][rows]; tru = np.where(ct < K, ct, -1); ok = tru >= 0
    iu = np.triu_indices(n, 1); print(f"trees for {n} traced cells; reconstructed against true distances: Spearman {spearmanr(D_true[iu], D_rec[iu])[0]:.2f}; "
                                      f"clones vs truth (ARI over traced cells in major clones): true tree {adjusted_rand_score(tru[ok], clones_of(D_true, F, mmin)[ok]):.2f}, "
                                      f"reconstructed {adjusted_rand_score(tru[ok], clones_of(D_rec, F, mmin)[ok]):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
