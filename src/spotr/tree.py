"""Trees for SpOTr, read from a Newick string, a networkx DiGraph or a TreeData object, with the distances between their leaves and clones cut from
them.

The leaves of every tree are cell names (the obs_names of the dissociated AnnData). The branch lengths come from the Newick lengths, from an edge
attribute (`length_key`), or from a node attribute with the depth or time of each node (`depth_key`, as pycea and treedata store them). Every
branch has length 1 when none of these is given.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------------------------------------------- reading trees
def parse_newick(s: str):
    """Return the names, parents and branch lengths of every node in pre-order, with each parent before its children. Internal nodes may be unnamed.
    """
    import sys
    s = s.strip().rstrip(";"); names, parent, length = [], [], []; pos = [0]
    def node(p):
        v = len(names); names.append(""); parent.append(p); length.append(0.0)
        if pos[0] < len(s) and s[pos[0]] == "(":
            pos[0] += 1
            while True:
                node(v)
                if s[pos[0]] == ",":
                    pos[0] += 1; continue
                if s[pos[0]] == ")":
                    pos[0] += 1; break
                raise ValueError(f"unexpected '{s[pos[0]]}' at {pos[0]} in the Newick string")
        j = pos[0]
        while j < len(s) and s[j] not in ",():":
            j += 1
        names[v] = s[pos[0]:j].strip().strip("'\""); pos[0] = j
        if pos[0] < len(s) and s[pos[0]] == ":":
            j = pos[0] + 1
            while j < len(s) and s[j] not in ",()":
                j += 1
            length[v] = float(s[pos[0] + 1:j]); pos[0] = j
        return v
    old = sys.getrecursionlimit(); sys.setrecursionlimit(max(old, 100000))
    try:
        node(-1)
    finally:
        sys.setrecursionlimit(old)
    return names, np.asarray(parent), np.asarray(length, float)


def from_networkx(G, length_key: str = "length", depth_key: str | None = None):
    """Return the names, parents and branch lengths of every node in pre-order, from a rooted networkx DiGraph with edges from parent to child.
    """
    import networkx as nx
    root = [n for n in G.nodes if G.in_degree(n) == 0]
    if len(root) != 1:
        raise ValueError(f"the tree must have one root, found {len(root)}")
    order = list(nx.dfs_preorder_nodes(G, root[0])); idx = {n: i for i, n in enumerate(order)}
    parent = np.full(len(order), -1); length = np.zeros(len(order))
    for u, v, d in G.edges(data=True):
        parent[idx[v]] = idx[u]
        if depth_key is not None and depth_key in G.nodes[v] and depth_key in G.nodes[u]:
            length[idx[v]] = float(G.nodes[v][depth_key]) - float(G.nodes[u][depth_key])
        else:
            length[idx[v]] = float(d.get(length_key, 1.0))
    return [str(n) for n in order], parent, length


# ---------------------------------------------------------------------------------------------------------------- distances
def leaf_distances(parent, length, leaf_rows: dict) -> np.ndarray:
    """Return the patristic distances between the leaves in `leaf_rows` (node index -> row), from parent pointers in pre-order and branch lengths.
    The distances between the leaves below different children of each internal node are set at once.
    """
    parent = np.asarray(parent); n_nodes = len(parent); depth = np.zeros(n_nodes)
    for v in range(n_nodes):
        if parent[v] >= 0:
            depth[v] = depth[parent[v]] + float(length[v])
    n = len(leaf_rows); D = np.zeros((n, n)); below = [None] * n_nodes; kids = [[] for _ in range(n_nodes)]
    row_depth = np.zeros(n)
    for v, r in leaf_rows.items():
        row_depth[r] = depth[v]
    for v in range(n_nodes):
        if parent[v] >= 0:
            kids[parent[v]].append(v)
    for v in range(n_nodes - 1, -1, -1):                                                        # children before parents (reverse pre-order)
        lists = [below[c] for c in kids[v] if below[c] is not None and len(below[c])]
        if v in leaf_rows:
            lists.append(np.array([leaf_rows[v]]))
        for a in range(len(lists)):
            for b in range(a + 1, len(lists)):
                La, Lb = lists[a], lists[b]; d = row_depth[La][:, None] + row_depth[Lb][None, :] - 2.0 * depth[v]
                D[np.ix_(La, Lb)] = d; D[np.ix_(Lb, La)] = d.T
        below[v] = np.concatenate(lists) if lists else None
        for c in kids[v]:
            below[c] = None
    return D


def tree_distances(tree, cells, tree_key: str | None = None, length_key: str = "length", depth_key: str | None = None) -> np.ndarray:
    """Return the tree distances between `cells`, given as leaf names and in that order, for a Newick string, a networkx DiGraph or a TreeData
    object. For a TreeData object, `tree_key` selects one of its `.obst` trees (default the only or first one).
    """
    if isinstance(tree, str):
        names, parent, length = parse_newick(tree)
    elif hasattr(tree, "obst"):                                                                 # treedata.TreeData (pycea)
        keys = list(tree.obst.keys()); G = tree.obst[tree_key if tree_key is not None else keys[0]]
        names, parent, length = from_networkx(G, length_key, depth_key)
    elif hasattr(tree, "edges") and hasattr(tree, "in_degree"):
        names, parent, length = from_networkx(tree, length_key, depth_key)
    else:
        raise TypeError("tree must be a Newick string, a networkx DiGraph or a TreeData object")
    where = {nm: i for i, nm in enumerate(names) if nm != ""}
    missing = [c for c in cells if str(c) not in where]
    if missing:
        raise ValueError(f"{len(missing)} cells are not leaves of the tree, e.g. {missing[:3]}")
    return leaf_distances(parent, length, {where[str(c)]: r for r, c in enumerate(cells)})


def cut_clones(D: np.ndarray, n_clones: int) -> np.ndarray:
    """Return clones (0 .. n_clones - 1) as the n_clones groups of an average-linkage clustering of the tree distances."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    return fcluster(linkage(squareform(D, checks=False), method="average"), t=int(n_clones), criterion="maxclust") - 1


def subtree(parent, length, keep: dict):
    """Return the names, parents and branch lengths in pre-order of the tree induced on the leaves in `keep` (node index -> name). The induced tree
    contains the nodes on the paths from these leaves to the root, and every node with a single child is merged with it, adding the branch
    lengths.
    """
    parent = np.asarray(parent); on = np.zeros(len(parent), bool)
    for v in keep:
        u = int(v)
        while u >= 0 and not on[u]:
            on[u] = True; u = int(parent[u])
    kids = [[] for _ in parent]
    for v in range(len(parent)):
        if on[v] and parent[v] >= 0:
            kids[parent[v]].append(v)
    names, npar, nlen = [], [], []; stack = [(int(np.where(parent < 0)[0][0]), -1, 0.0)]
    while stack:
        v, p_new, acc = stack.pop()
        while len(kids[v]) == 1 and v not in keep:
            c = kids[v][0]; acc += float(length[c]); v = c
        i = len(names); names.append(keep.get(v, "")); npar.append(p_new); nlen.append(acc)
        for c in reversed(kids[v]):
            stack.append((c, i, float(length[c])))
    return names, np.array(npar), np.array(nlen)


def to_newick(parent, length, names) -> str:
    """Return the tree as a Newick string, with leaves and internal nodes by name and branch lengths after ':'."""
    parent = np.asarray(parent); kids = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p >= 0:
            kids[p].append(v)
    def rec(v):
        lab = names[v] if names[v] else ""
        s = f"({','.join(rec(c) for c in kids[v])}){lab}" if kids[v] else lab
        return s + (f":{float(length[v])!r}" if parent[v] >= 0 else "")                         # full precision
    return rec(int(np.where(parent < 0)[0][0])) + ";"
