"""The tree given to SpOTr (environment benchmark_env; numpy only): the true tree's distances (sums of branch lengths in time) between the
traced dissociated cells, and their true clones, the pooled smaller clades as -1.

Writes trees.npz: rows (the traced cells' rows in problem.npz), D_true (their distances), clone_true_tree, K (the number of clones, not pooled).

Usage: python tree.py <sim.npz> <problem.npz> <out.npz>
"""
import sys

import numpy as np


def leaf_distances(parent, depth, leaf_rows, n):
    """distances between the leaves listed in leaf_rows ({node: row}), from parent pointers in pre-order (a parent before its children) and each
    node's depth"""
    kids = [[] for _ in range(len(parent))]
    for v in range(len(parent)):
        if parent[v] >= 0:
            kids[parent[v]].append(v)
    depth_row = np.zeros(n)
    for v, r in leaf_rows.items():
        depth_row[r] = depth[v]
    D = np.zeros((n, n)); below = [None] * len(parent)
    for v in range(len(parent) - 1, -1, -1):                                                        # children before parents
        lists = [below[c] for c in kids[v] if below[c] is not None and len(below[c])]
        if v in leaf_rows:
            lists.append(np.array([leaf_rows[v]]))
        for a in range(len(lists)):
            for b in range(a + 1, len(lists)):
                La, Lb = lists[a], lists[b]; d = depth_row[La][:, None] + depth_row[Lb][None, :] - 2.0 * depth[v]
                D[np.ix_(La, Lb)] = d; D[np.ix_(Lb, La)] = d.T
        below[v] = np.concatenate(lists) if lists else None
        for c in kids[v]:
            below[c] = None
    return D


def main(sim, prob, out):
    z = np.load(sim, allow_pickle=True); q = np.load(prob); rows = np.where(q["traced"])[0]; leaf = q["leaf"][rows]; n = len(rows); K = int(z["K"]) - 1
    D = leaf_distances(z["parent"], z["time"], {int(z["leaf_node"][l]): i for i, l in enumerate(leaf)}, n)
    ct = q["clone"][rows]; np.savez_compressed(out, rows=rows, D_true=D, clone_true_tree=np.where(ct < K, ct, -1), K=K)
    print(f"tree distances between {n} traced cells, {K} clones")


if __name__ == "__main__":
    main(*sys.argv[1:4])
