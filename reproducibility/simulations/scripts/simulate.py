"""The lineage, the barcodes, the positions of the tumor cells and their clones (environment cassiopeia).

Lineage. We grow a lineage with the BirthDeathFitnessSimulator of Cassiopeia to as many leaves as the tissue layout (layout.py) has tumor sites,
and simulate it again if it dies out.
Barcodes. We simulate the barcodes with the Cas9LineageTracingDataSimulator, with the edit rate set so that an expected fraction
barcode_saturation of the sites is edited at the leaves.
Positions. We cut the tumor sites of the layout along the tree from the root down. We cut the sites of each clade along their principal axis,
turned by an angle from Normal(0, territory_jitter), into one contiguous part per child clade with the number of cells of that clade
(territories()), so that every clade occupies a territory within that of its parent.
Clones. The clones are the clades at the earliest time at which n_clones clades contain at least a fraction clone_min of the cells, ordered from
the largest, with the smaller clades pooled as the last clone index.
Dispersal. We disperse a fraction `migrate` of the clones. To disperse a clone, we exchange the positions of its cells with those of random tumor
cells of the clones that are not dispersed, or of any cells outside the clone when the dispersed clones contain most cells.

Usage: python simulate.py <params.json> <seed> <layout.npz> <out.npz>
"""
import json
import sys

import numpy as np


def territories(parent, leaf_node, sites, rng, jitter=0.0):
    """Assign the clades of the tree to nested territories on the sites, with the nodes in pre-order (each parent before its children)."""
    ch = [[] for _ in parent]
    for v, u in enumerate(parent):
        if u >= 0:
            ch[u].append(v)
    leaf_of = {int(l): i for i, l in enumerate(leaf_node)}; size = np.zeros(len(parent), int)
    for v in range(len(parent) - 1, -1, -1):
        size[v] = 1 if v in leaf_of else sum(size[c] for c in ch[v])
    xy = np.zeros((len(leaf_node), 2)); stack = [(int(np.where(parent < 0)[0][0]), np.arange(len(sites)))]
    while stack:
        v, S = stack.pop()
        if v in leaf_of:
            xy[leaf_of[v]] = sites[S[0]]; continue
        cs = [c for c in ch[v] if size[c] > 0]
        if len(cs) == 1:
            stack.append((cs[0], S)); continue
        X = sites[S] - sites[S].mean(0); u = np.linalg.svd(X, full_matrices=False)[2][0]; th = rng.normal(0.0, jitter)
        u = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ u; o = S[np.argsort(X @ u)]; rng.shuffle(cs); k = 0
        for c in cs:
            stack.append((c, o[k:k + size[c]])); k += size[c]
    return xy


def clones(parent, time, leaf_node, n_clones, clone_min):
    """Return the clone of each leaf (0 for the largest and the last index for the pooled smaller clades) and the number of clones including the
    pooled one.
    """
    n = len(leaf_node); root = int(np.where(parent < 0)[0][0]); nch = np.bincount(parent[parent >= 0], minlength=len(parent))
    cuts = np.sort(time[np.where(nch > 0)[0]])
    def clades(cut):                                                                                # the highest ancestor of each leaf at or after the cut
        v = leaf_node.copy()
        while True:
            u_ = parent[v]; mv = (u_ != root) & (u_ >= 0) & (time[np.maximum(u_, 0)] >= cut)
            if not mv.any():
                return v
            v[mv] = u_[mv]
    for cut in cuts:
        anc = clades(cut)
        if int((np.unique(anc, return_counts=True)[1] >= clone_min * n).sum()) >= n_clones:
            break
    _, inv, cnt = np.unique(anc, return_inverse=True, return_counts=True); big = np.where(cnt >= clone_min * n)[0]; big = big[np.argsort(-cnt[big])]
    clone = np.full(n, len(big), int)
    for k, b in enumerate(big):
        clone[inv == b] = k
    return clone, len(big) + 1


def main(pfile, seed, layout, out):
    from cassiopeia.mixins.errors import TreeSimulatorError
    from cassiopeia.simulator import BirthDeathFitnessSimulator, Cas9LineageTracingDataSimulator
    p = json.load(open(pfile)); seed = int(seed); rng = np.random.default_rng(seed); np.random.seed(seed)
    L = np.load(layout); sites = L["xy"][L["type"] == 0]; fr, fsd = float(p["fitness_rate"]), float(p["fitness_sd"])
    for attempt in range(50):
        try:
            bd = BirthDeathFitnessSimulator(birth_waiting_distribution=lambda scale: np.random.exponential(scale), initial_birth_scale=1.0,
                                            death_waiting_distribution=lambda: np.random.exponential(float(p["death_scale"])),
                                            mutation_distribution=lambda: 1 if np.random.uniform() < fr else 0, fitness_distribution=lambda: np.random.normal(0, fsd),
                                            fitness_base=np.e, num_extant=len(sites), random_seed=seed + 1000 * attempt)
            tree = bd.simulate_tree(); break
        except TreeSimulatorError:
            continue
    leaves = list(tree.leaves); n = len(leaves); depth = float(np.mean([tree.get_time(l) for l in leaves]))
    rate = -np.log(1.0 - float(p["barcode_saturation"])) / depth
    Cas9LineageTracingDataSimulator(number_of_cassettes=int(p["cassettes"]), size_of_cassette=int(p["sites"]), mutation_rate=rate, number_of_states=100,
                                    random_seed=seed).overlay_data(tree)
    cm = tree.character_matrix.loc[leaves].to_numpy(int)
    nodes = list(tree.depth_first_traverse_nodes(source=tree.root, postorder=False)); idx = {v: i for i, v in enumerate(nodes)}
    parent = np.array([idx[tree.parent(v)] if v != tree.root else -1 for v in nodes]); time = np.array([tree.get_time(v) for v in nodes], float)
    leaf_node = np.array([idx[l] for l in leaves])
    xy = territories(parent, leaf_node, sites, rng, float(p["territory_jitter"]))
    clone, K = clones(parent, time, leaf_node, int(p["n_clones"]), float(p["clone_min"])); nbig = K - 1
    mig = np.array([], int)
    if float(p["migrate"]) > 0 and nbig > 1:
        mig = rng.choice(nbig, size=max(1, int(round(float(p["migrate"]) * nbig))), replace=False)
        for k in mig:
            i = np.where(clone == k)[0]; pop = np.where(~np.isin(clone, mig))[0]
            if len(pop) < len(i):                                                                   # when the dispersed clones contain most cells, exchange
                pop = np.where(clone != k)[0]; i = rng.choice(i, size=min(len(i), len(pop)), replace=False)   # with any cell outside the clone
            j = rng.choice(pop, size=len(i), replace=False); xy[i], xy[j] = xy[j].copy(), xy[i].copy()
    np.savez_compressed(out, leaves=np.array(leaves), xy=xy, clone=clone, K=K, migrating=mig, characters=cm, parent=parent, time=time, leaf_node=leaf_node,
                        depth=depth, edit_rate=rate, newick=tree.get_newick(record_branch_lengths=True))
    print(f"simulated {n} cells, {K - 1} clones (+pooled), depth {depth:.2f}, edited sites {np.mean(cm > 0):.2f}, dispersed clones {len(mig)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
