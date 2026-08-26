"""Confusable-expression testbed: the regime where the tree CORRECTS expression.

Earlier testbeds made expression correct-but-uninformative at the fine level. Then the tree
had only refinement to offer, downweighting it where it was useless bought nothing, and
per-clade weighting had no work to do.

The premise here is different: expression does not respect the tree. Clades that share an
expression program are CONFUSABLE, so per-cell assignment is only weakly right and clones
get mapped into each other's territory. Two things follow.

POOLING. If each cell independently picks the right territory 55% of the time, 45% land
wrong. The GW term forces a clade to move together, so the clade makes ONE 55/45 decision
instead of n independent ones -- a majority vote of 100 cells at 55% is right ~84% of the
time. GW converts many weak decisions into one strong collective one. That is a genuine
information gain, not refinement, and no earlier testbed contained it.

HEADROOM FOR PER-CLADE ALPHA. Pooling is only correct for a clade that really is spatially
coherent. For a DISPERSED clade, forcing coherence drags its cells to one wrong place -- the
tree is actively harmful. So over-weighting now costs first-order, not second-order, and
alpha_k has something real to do:

    compact clades    territory follows the tree   -> pooling is right   -> alpha_k HIGH
    dispersed clades  cells scattered tissue-wide  -> pooling is wrong   -> alpha_k LOW

Clade territories are placed at BM-derived clade centroids, so the BETWEEN-clade layout
matches the tree's own distance structure. That is what makes the correct alignment a strict
optimum rather than one of K! ties: GW is rigid up to global isometry once the layout is
generic, and expression only has to fix the global frame.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import cassiopeia as cas
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src" / "spotr"
sys.path.insert(0, str(SRC.parent))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def split_to(tree, root, n_parts, min_size):
    parts = [root]
    while len(parts) < n_parts:
        parts.sort(key=lambda x: -len(tree.leaves_in_subtree(x)))
        big = parts.pop(0)
        kids = tree.children(big)
        if not kids:
            parts.append(big); break
        parts.extend(kids)
    return [p for p in parts if len(tree.leaves_in_subtree(p)) >= min_size]


def main(n_cells=2600, n_clades=12, n_prog=6, n_genes=240, seed=0,
         confuse=0.90, frac_dispersed=0.5, compact_sd=0.22, prog_strength=2.0,
         expr_noise=1.0, cells_per_spot=3.0, batch_sd=0.3, spot_noise=0.3,
         out="sim_conf.npz"):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    utils = _load("utils"); space_sim = _load("space_simulator")

    ts = cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution=lambda scale: np.random.exponential(scale),
        initial_birth_scale=0.5,
        death_waiting_distribution=lambda: np.random.exponential(1.5),
        mutation_distribution=lambda: 0,
        fitness_distribution=lambda: np.random.normal(0, 0.5),
        num_extant=n_cells, random_seed=seed)
    tree = ts.simulate_tree()
    leaves = list(tree.leaves)
    pos = {l: i for i, l in enumerate(leaves)}

    clade_nodes = split_to(tree, tree.root, n_clades, min_size=30)[:n_clades]
    clade_of = {}
    for cn in clade_nodes:
        for l in tree.leaves_in_subtree(cn):
            clade_of[l] = cn
    keep = [l for l in leaves if l in clade_of]
    K = len(clade_nodes)
    ci = {l: i for i, l in enumerate(keep)}
    n = len(keep)
    clade_lab = np.array([clade_of[l] for l in keep])
    sizes = np.array([(clade_lab == c).sum() for c in clade_nodes])
    print(f"[sim] tree {len(leaves)} leaves -> {K} clades (sizes {sizes.min()}-{sizes.max()}), "
          f"{n} cells covered")

    # ---- which clades are spatially coherent ----
    n_disp = int(round(frac_dispersed * K))
    is_disp = np.zeros(K, dtype=bool)
    is_disp[rng.choice(K, size=n_disp, replace=False)] = True
    print(f"[sim] {K - n_disp} compact clades (tree pooling correct), "
          f"{n_disp} dispersed (tree pooling harmful)")

    # ---- spatial layout: clade centroids from BM on the tree ----
    # the between-clade arrangement must follow the tree's own distance structure, otherwise
    # the correct alignment is one of K! ties instead of a strict optimum
    sp = space_sim.BrownianSpatialDataSimulator(2, 1.)
    sp.overlay_data(tree, brownian_motion=True, lambda_brownian=1.)
    xy0 = tree.cell_meta[["spatial_0", "spatial_1"]].loc[leaves].to_numpy(float)
    cen = np.stack([xy0[[pos[l] for l in tree.leaves_in_subtree(cn)]].mean(0)
                    for cn in clade_nodes])
    cen = (cen - cen.mean(0)) / cen.std(0).mean()          # unit layout scale
    lo, hi = cen.min(0) - 1.0, cen.max(0) + 1.0

    xy = np.zeros((n, 2))
    for k, cn in enumerate(clade_nodes):
        rows = [ci[l] for l in tree.leaves_in_subtree(cn) if l in ci]
        if is_disp[k]:
            xy[rows] = rng.uniform(lo, hi, size=(len(rows), 2))       # scattered tissue-wide
        else:
            xy[rows] = cen[k] + rng.normal(0, compact_sd, size=(len(rows), 2))

    # ---- expression: shared programs, so clades are CONFUSABLE rather than distinct ----
    # clades are paired; a pair's program-weight vectors are mixed towards their mean by
    # `confuse`, so within a pair expression is nearly uninformative while the pair as a whole
    # is still distinguishable from other pairs. Territories of a pair are far apart, so
    # confusing them is expensive -- exactly the mis-mapping the tree is supposed to correct.
    progs = np.zeros((n_prog, n_genes))
    for p, gi in enumerate(np.array_split(rng.permutation(n_genes), n_prog)):
        progs[p, gi] = 1.0
    W = rng.dirichlet(np.ones(n_prog) * 0.6, size=K)
    order = np.argsort(-sizes)                     # pair by size so pairs are comparable
    pairs = [(order[i], order[i + 1]) for i in range(0, K - 1, 2)]
    for i, j in pairs:
        m = 0.5 * (W[i] + W[j])
        W[i] = confuse * m + (1 - confuse) * W[i]
        W[j] = confuse * m + (1 - confuse) * W[j]
    pair_of = {}
    for pi, (i, j) in enumerate(pairs):
        pair_of[i] = pi; pair_of[j] = pi

    E = rng.normal(0.0, expr_noise, size=(n, n_genes))
    for k, cn in enumerate(clade_nodes):
        rows = [ci[l] for l in tree.leaves_in_subtree(cn) if l in ci]
        # per-cell program sampling, so individual cells are ambiguous even within a clade
        mix = rng.dirichlet(W[k] * 25.0, size=len(rows))
        E[rows] += prog_strength * (mix @ progs)

    # ---- disjoint populations ----
    is_sc = np.zeros(n, dtype=bool)
    for c in np.unique(clade_lab):
        idx = np.where(clade_lab == c)[0]
        is_sc[rng.choice(idx, size=max(1, len(idx) // 2), replace=False)] = True
    sp_i, sc_i = np.where(~is_sc)[0], np.where(is_sc)[0]

    blo, bhi = xy[sp_i].min(0), xy[sp_i].max(0)
    nb = max(2, int(np.sqrt(len(sp_i) / cells_per_spot)))
    bx = np.clip(((xy[sp_i] - blo) / np.maximum(bhi - blo, 1e-9) * nb).astype(int), 0, nb - 1)
    bid = bx[:, 0] * nb + bx[:, 1]
    ub, inv = np.unique(bid, return_inverse=True)
    m = len(ub)
    M = np.zeros((m, len(sp_i))); M[inv, np.arange(len(sp_i))] = 1.0
    cnt = M.sum(1, keepdims=True)
    X_st = (M @ E[sp_i]) / cnt
    xy_st = (M @ xy[sp_i]) / cnt
    X_st = X_st * np.exp(rng.normal(0, batch_sd, n_genes))[None, :] + \
        rng.normal(0, spot_noise, size=X_st.shape)
    X_sc = E[sc_i]
    print(f"[sim] disjoint split: {len(sc_i)} cells for scRNA-seq, {len(sp_i)} cells "
          f"composing {m} spots ({cnt.mean():.1f} cells/spot)")

    st_clade = np.array([pd.Series(clade_lab[sp_i][inv == k]).value_counts().index[0]
                         for k in range(m)])

    W_tree = utils.cophenetic_distance_matrix(tree).loc[leaves, leaves].to_numpy()
    ii = [pos[keep[i]] for i in sc_i]
    C_tree = W_tree[np.ix_(ii, ii)]
    C_space = squareform(pdist(xy_st))

    # ---- design checks ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    print("\n[check] is expression CONFUSABLE (weak per-cell), not just uninformative?")
    acc = cross_val_score(LogisticRegression(max_iter=600, class_weight="balanced"),
                          X_sc, clade_lab[sc_i], cv=3, scoring="accuracy").mean()
    print(f"  expression -> clade, all {K} clades      accuracy = {acc:.3f}   "
          f"(chance {1.0/K:.3f})")
    pl = np.array([pair_of.get(list(clade_nodes).index(c), -1) for c in clade_lab[sc_i]])
    pacc = cross_val_score(LogisticRegression(max_iter=600, class_weight="balanced"),
                           X_sc, pl, cv=3, scoring="accuracy").mean()
    print(f"  expression -> PAIR                       accuracy = {pacc:.3f}   "
          f"(chance {1.0/len(pairs):.3f})")
    within = []
    for i, j in pairs:
        sel = np.isin(clade_lab[sc_i], [clade_nodes[i], clade_nodes[j]])
        if sel.sum() < 20:
            continue
        within.append(cross_val_score(
            LogisticRegression(max_iter=600, class_weight="balanced"),
            X_sc[sel], (clade_lab[sc_i][sel] == clade_nodes[i]).astype(int),
            cv=3, scoring="balanced_accuracy").mean())
    print(f"  expression -> which of a confusable PAIR  accuracy = {np.mean(within):.3f}   "
          f"(chance 0.500)  <- the ambiguity the tree must resolve")

    print("\n[check] are compact clades compact and dispersed ones dispersed?")
    for nm, msk in [("compact", ~is_disp), ("dispersed", is_disp)]:
        sd = [xy[[ci[l] for l in tree.leaves_in_subtree(clade_nodes[k]) if l in ci]].std(0).mean()
              for k in np.where(msk)[0]]
        print(f"  {nm:<10} mean within-clade spatial sd = {np.mean(sd):.3f}")
    dpair = np.mean([np.linalg.norm(cen[i] - cen[j]) for i, j in pairs])
    print(f"  mean territory distance within a confusable pair = {dpair:.3f} "
          f"(layout scale 1.0) -- confusing a pair is expensive")

    np.savez_compressed(
        HERE / out,
        X_sc=X_sc.astype(np.float32), X_st=X_st.astype(np.float32),
        C_tree=(C_tree / C_tree.max()).astype(np.float32),
        C_space=(C_space / max(C_space.max(), 1e-9)).astype(np.float32),
        xy_sc=xy[sc_i], xy_st=xy_st,
        cell_clade=clade_lab[sc_i], spot_clade=st_clade,
        clades=np.array(clade_nodes), is_dispersed=is_disp,
        pair_id=np.array([pair_of.get(k, -1) for k in range(K)]))
    print(f"\n[sim] wrote {out}: {len(sc_i)} cells x {m} spots, {K} clades")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in [("n-cells", 2600), ("n-clades", 12), ("n-prog", 6), ("n-genes", 240), ("seed", 0)]:
        ap.add_argument(f"--{k}", type=int, default=v, dest=k.replace("-", "_"))
    for k, v in [("confuse", 0.90), ("frac-dispersed", 0.5), ("compact-sd", 0.22),
                 ("prog-strength", 2.0), ("expr-noise", 1.0), ("cells-per-spot", 3.0),
                 ("batch-sd", 0.3), ("spot-noise", 0.3)]:
        ap.add_argument(f"--{k}", type=float, default=v, dest=k.replace("-", "_"))
    ap.add_argument("--out", default="sim_conf.npz")
    main(**vars(ap.parse_args()))
