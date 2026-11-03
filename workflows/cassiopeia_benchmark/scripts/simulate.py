"""Lineage, barcodes and positions with Cassiopeia (run in the cassiopeia environment).

Tree: BirthDeathFitnessSimulator until it holds as many leaves as the tissue layout (layout.py) has tumour cells.  Barcodes: Cas9LineageTracingDataSimulator, with the edit rate set so that a share
barcode_saturation of the sites is edited at the leaves.  Positions: ClonalSpatialDataSimulator on a disc (spatial: clonal) or
BrownianSpatialDataSimulator (spatial: brownian), rescaled so that cells are 12 um apart on average.  Migration: the cells of a share
`migrate` of the major clones swap places with random cells of other clones.  Within-clone mixing: a share `mix_within` of each major clone's
cells trade places among themselves (absent: 0).  Clones: the clades of the tree's first clone_founders lineages
holding at least clone_min of the cells; the rest pooled; or, with n_clones, the clades at the earliest cut where exactly n_clones of them
hold at least clone_min of the cells.  Positions (spatial: territories, 2026-11-02): the layout's tumour sites cut along the tree from the root
down, every clade a contiguous territory within its parent's (territories()).

Usage: python simulate.py <params.json> <seed> <layout.npz> <out.npz>
"""
import sys, json
import numpy as np


def territories(parent, leaf_node, sites, rng, jitter=0.0):
    """the tree's clades as nested territories on the tumour sites: from the root down, a clade's sites are cut along their principal axis
    (turned by a N(0, jitter) angle) into one contiguous part per child clade, of the child's number of cells, in random order"""
    ch = [[] for _ in parent]
    for v, u in enumerate(parent):
        if u >= 0:
            ch[u].append(v)
    leaf_of = {int(l): i for i, l in enumerate(leaf_node)}; size = np.zeros(len(parent), int)
    for v in range(len(parent) - 1, -1, -1):                                                        # nodes in pre-order: reversed, children first
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


def main(pfile, seed, layout, out):
    import cassiopeia as cas
    from cassiopeia.simulator import BirthDeathFitnessSimulator, Cas9LineageTracingDataSimulator, ClonalSpatialDataSimulator, BrownianSpatialDataSimulator
    p = json.load(open(pfile)); seed = int(seed); rng = np.random.default_rng(seed); np.random.seed(seed); n_leaves = int(np.load(layout)["n_tumour"])
    fr, fsd = float(p["fitness_rate"]), float(p["fitness_sd"])
    from cassiopeia.mixins.errors import TreeSimulatorError
    for attempt in range(50):                                                                       # a lineage that dies out is redrawn (seed + 1000 x attempt)
        try:
            bd = BirthDeathFitnessSimulator(birth_waiting_distribution=lambda scale: np.random.exponential(scale), initial_birth_scale=1.0,
                                            death_waiting_distribution=lambda: np.random.exponential(float(p["death_scale"])),
                                            mutation_distribution=lambda: 1 if np.random.uniform() < fr else 0, fitness_distribution=lambda: np.random.normal(0, fsd),
                                            fitness_base=np.e, num_extant=n_leaves, random_seed=seed + 1000 * attempt)
            tree = bd.simulate_tree(); break
        except TreeSimulatorError:
            continue
    leaves = list(tree.leaves); n = len(leaves)
    depth = float(np.mean([tree.get_time(l) for l in leaves]))
    rate = -np.log(1.0 - float(p["barcode_saturation"])) / depth                                    # P(edited at a leaf) = saturation
    Cas9LineageTracingDataSimulator(number_of_cassettes=int(p["cassettes"]), size_of_cassette=int(p["sites"]), mutation_rate=rate, number_of_states=100,
                                    random_seed=seed).overlay_data(tree)
    cm = tree.character_matrix.loc[leaves].to_numpy(int)
    # the true tree as parent pointers with times, for the expression layer and the distances
    nodes = list(tree.depth_first_traverse_nodes(source=tree.root, postorder=False)); idx = {v: i for i, v in enumerate(nodes)}
    parent = np.array([idx[tree.parent(v)] if v != tree.root else -1 for v in nodes]); time = np.array([tree.get_time(v) for v in nodes], float)
    leaf_node = np.array([idx[l] for l in leaves])
    if p["spatial"] == "territories":                                                               # the layout's tumour sites split along the tree
        L = np.load(layout); xy = territories(parent, leaf_node, L["xy"][L["type"] == 0], rng, float(p.get("territory_jitter", 0.0)))
    elif p["spatial"] == "clonal":
        side = int(np.ceil(np.sqrt(n / (np.pi / 4)) * 1.6)); ClonalSpatialDataSimulator(shape=(side, side), random_seed=seed).overlay_data(tree)
    else:
        BrownianSpatialDataSimulator(dim=2, diffusion_coefficient=1.0, scale_unit_area=True, random_seed=seed).overlay_data(tree)
    if p["spatial"] != "territories":
        xy = np.array([tree.get_attribute(l, "spatial") for l in leaves], float)[:, :2]
    from scipy.spatial import cKDTree
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]; xy = (xy - xy.mean(0)) * (12.0 / max(np.mean(nn), 1e-12))  # cells 12 um apart on average
    # clones: the clades of the first clone_founders lineages (the tree's lineages at the time it first holds that many) that hold at least
    # clone_min of the cells; or, with n_clones (2026-11-03), the clades at the earliest such cut where exactly n_clones of them do
    root = int(np.where(parent < 0)[0][0]); nch = np.bincount(parent[parent >= 0], minlength=len(parent))
    cuts = time[np.where(nch > 0)[0]]; cuts.sort(); nchs = nch[np.where(nch > 0)[0]][np.argsort(time[np.where(nch > 0)[0]], kind="stable")]
    def clades(cut):                                                                                # each leaf's highest ancestor below the cut
        v = leaf_node.copy()
        while True:
            u_ = parent[v]; mv = (u_ != root) & (u_ >= 0) & (time[np.maximum(u_, 0)] >= cut)
            if not mv.any():
                return v
            v[mv] = u_[mv]
    nbig = lambda a: int((np.unique(a, return_counts=True)[1] >= float(p["clone_min"]) * n).sum())
    if "n_clones" in p:
        for cut in cuts:
            anc = clades(cut)
            if nbig(anc) >= int(p["n_clones"]):
                break
    else:
        lin = 1
        for cut, k_ in zip(cuts, nchs):
            lin += int(k_) - 1
            if lin >= int(p["clone_founders"]):
                break
        anc = clades(cut)
    u, inv, cnt = np.unique(anc, return_inverse=True, return_counts=True); big = np.where(cnt >= float(p["clone_min"]) * n)[0]; big = big[np.argsort(-cnt[big])]
    clone = np.full(n, len(big), int)
    for k, b in enumerate(big):
        clone[inv == b] = k
    K = len(big) + 1                                                                                # last index: the pooled small clades
    if float(p["migrate"]) > 0 and len(big) > 1:                                                    # migration: swap places with random other cells
        mig = rng.choice(len(big), size=max(1, int(round(float(p["migrate"]) * len(big)))), replace=False)
        for k in mig:
            i = np.where(clone == k)[0]; pop = np.where(~np.isin(clone, mig))[0]
            if len(pop) < len(i):                                                                   # the migrating clones hold most cells (2026-11-02):
                pop = np.where(clone != k)[0]; i = rng.choice(i, size=min(len(i), len(pop)), replace=False)   # swap with any cell outside the clone
            j = rng.choice(pop, size=len(i), replace=False)                                         # (unchanged wherever the original worked)
            xy[i], xy[j] = xy[j].copy(), xy[i].copy()
    else:
        mig = np.array([], int)
    mw = float(p.get("mix_within", 0.0))
    if mw > 0:                                                                                      # within-clone mixing (Pedro, 2026-11-02: when the tree helps
        for k in range(len(big)):                                                                   # over clone labels): a share of each major clone's cells trade
            i = np.where(clone == k)[0]; s_ = rng.choice(i, size=int(round(mw * len(i))), replace=False)   # places among themselves -- the clone's
            xy[s_] = xy[rng.permutation(s_)]                                                        # territory kept, its sub-lineage patches dissolved
    np.savez_compressed(out, leaves=np.array(leaves), xy=xy, clone=clone, K=K, migrating=mig, characters=cm, parent=parent, time=time, leaf_node=leaf_node,
                        depth=depth, edit_rate=rate, newick=tree.get_newick(record_branch_lengths=True))
    print(f"simulated {n} cells, {K - 1} clones (+pooled), depth {depth:.2f}, edited sites {np.mean(cm > 0):.2f}, migrating clones {len(mig)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
