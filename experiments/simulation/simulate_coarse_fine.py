"""Coarse-to-fine testbed: the regime in which a second-order term can actually work.

The previous crossed design asked for clades with "clear tree-space concordance but NO
expression signal". Those clades are UNIDENTIFIABLE. GW is second-order: it says "send
this clade to *a* compact region" without saying which one. Permuting the
expression-invisible clades across the leftover regions leaves the cost unchanged, so
their expected accuracy is chance at every alpha -- while alpha > 0 costs accuracy on the
clades that do have expression signal. Hence ARI fell monotonically with alpha. The
simulation was not broken; it was correctly reporting an unidentifiable regime.

Here expression is AMBIGUOUS rather than ABSENT:

  every clade       dedicated COARSE marker genes    -> expression pins the REGION,
                                                        which breaks the GW symmetry
  expr-fine clades  per-SUBCLADE marker genes         -> expression also pins position
                    within-clade lambda = 0              WITHIN the region; the tree there
                                                         is pure noise -> alpha_k LOW
  tree-fine clades  no per-subclade markers           -> only the tree pins position
                    within-clade lambda = 1              within the region -> alpha_k HIGH

Expression carries WHICH region; the tree carries WHERE WITHIN it. That is the only
regime in which a relational term adds information, and it is the regime the real data
sits in (best alpha on eTRACER is 0.075 -- a refiner, not a source).

The metric has to change with it. ARI on clade identity measures only the COARSE level,
which expression saturates by construction, so a refiner cannot show up there at all.
Primary metric is SPATIAL ERROR: distance between a cell's true position and the
barycentre of its coupling row, in units of tissue diameter.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
import cassiopeia as cas
from scipy.spatial.distance import pdist, squareform
from scipy import stats

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src" / "spotr"
sys.path.insert(0, str(SRC.parent))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def split_to(tree, root, n_parts, min_size):
    """Repeatedly split the largest subtree until we have n_parts of at least min_size."""
    parts = [root]
    while len(parts) < n_parts:
        parts.sort(key=lambda x: -len(tree.leaves_in_subtree(x)))
        big = parts.pop(0)
        kids = tree.children(big)
        if not kids:
            parts.append(big); break
        parts.extend(kids)
    return [p for p in parts if len(tree.leaves_in_subtree(p)) >= min_size]


def main(n_cells=3000, n_clades=12, n_sub=4, n_genes=300, seed=0,
         sep=2.0, sub_sep=2.0, coarse_strength=2.5, fine_strength=2.0, n_coarse=8, n_fine=6,
         expr_noise=1.0, cells_per_spot=3.0, batch_sd=0.3, spot_noise=0.3,
         out="sim_cf.npz"):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    utils = _load("utils"); space_sim = _load("space_simulator")

    # unbalanced tree: a balanced ultrametric tree makes the GW linearisation additive and
    # double-centering annihilates it (|L|/|F| ~ 4e-6), so alpha cannot do anything
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

    # ---- two-level partition: clades, then subclades within each clade ----
    clade_nodes = split_to(tree, tree.root, n_clades, min_size=8 * n_sub)
    clade_nodes = clade_nodes[:n_clades]
    sub_of, clade_of = {}, {}
    for cn in clade_nodes:
        subs = split_to(tree, cn, n_sub, min_size=6)
        for si, sn in enumerate(subs):
            for l in tree.leaves_in_subtree(sn):
                sub_of[l] = f"{cn}|{si}"
        for l in tree.leaves_in_subtree(cn):
            clade_of[l] = cn
    keep = [l for l in leaves if l in sub_of]
    print(f"[sim] tree {len(leaves)} leaves -> {len(clade_nodes)} clades x ~{n_sub} subclades, "
          f"{len(keep)} cells covered")

    # ---- which clades are expression-fine and which are tree-fine ----
    grp = np.array([0, 1] * (len(clade_nodes) // 2 + 1))[:len(clade_nodes)]
    rng.shuffle(grp)                                 # 1 = tree-fine, 0 = expr-fine
    is_tree_fine = {cn: bool(grp[i]) for i, cn in enumerate(clade_nodes)}
    print(f"[sim] {int((grp==0).sum())} expr-fine clades (subclade markers, within-lambda 0), "
          f"{int((grp==1).sum())} tree-fine clades (no subclade markers, within-lambda 1)")

    # ---- space: Brownian motion on the tree, then rescale the two levels separately ----
    # BM on a tree is hierarchical by construction: clades separate, and subclades separate
    # WITHIN clades. Decompose into clade centroid + within-clade deviation so `sep` sets how
    # far apart regions sit relative to their own extent, and so within-clade lambda can be
    # set per clade without touching the between-clade layout.
    sp = space_sim.BrownianSpatialDataSimulator(2, 1.)
    sp.overlay_data(tree, brownian_motion=True, lambda_brownian=1.)
    xy0 = tree.cell_meta[["spatial_0", "spatial_1"]].loc[leaves].to_numpy(float)

    cen_all = np.zeros_like(xy0); dev_all = np.zeros_like(xy0)
    for cn in clade_nodes:
        ii = [pos[l] for l in tree.leaves_in_subtree(cn)]
        c = xy0[ii].mean(0)
        cen_all[ii] = c
        dev_all[ii] = xy0[ii] - c
    cs = cen_all[[pos[l] for l in keep]].std(0).mean()
    ds = dev_all[[pos[l] for l in keep]].std(0).mean()
    cen_all *= (sep / max(cs, 1e-9)); dev_all *= (1.0 / max(ds, 1e-9))

    # within-clade lambda: 1 keeps the tree-driven deviation, 0 replaces it with iid noise of
    # matched scale. The clade centroid is untouched either way, so the COARSE layout -- the
    # thing expression identifies -- is identical for both clade types.
    # Within a clade, plain BM gives only r ~ 0.18 between cophenetic and spatial distance --
    # BM is a random walk, so two cells at equal tree distance land anywhere. "Clear tree-space
    # concordance" needs the SUBCLADE level separated the same way clades are: apply the same
    # centroid/deviation split one level down, with sub_sep setting subregion separation
    # relative to within-subclade jitter.
    xy = cen_all.copy()
    for cn in clade_nodes:
        ii = [pos[l] for l in tree.leaves_in_subtree(cn)]
        dev = dev_all[ii]
        if is_tree_fine[cn]:
            sl = np.array([sub_of.get(l, "") for l in tree.leaves_in_subtree(cn)])
            sc_, rs_ = np.zeros_like(dev), np.zeros_like(dev)
            for s in np.unique(sl):
                jj = np.where(sl == s)[0]
                sc_[jj] = dev[jj].mean(0); rs_[jj] = dev[jj] - dev[jj].mean(0)
            a_ = sc_.std(0).mean(); b_ = max(rs_.std(0).mean(), 1e-9)
            dev = sub_sep * sc_ / max(a_, 1e-9) + rs_ / b_
            dev /= max(dev.std(0).mean(), 1e-9)
            xy[ii] += dev
        else:
            nz = rng.normal(size=dev.shape)
            nz *= dev.std(0, keepdims=True) / np.maximum(nz.std(0, keepdims=True), 1e-9)
            xy[ii] += nz

    # ---- expression: coarse markers for every clade, fine markers only for expr-fine ----
    ci = {l: i for i, l in enumerate(keep)}
    n = len(keep)
    E = rng.normal(0.0, expr_noise, size=(n, n_genes))
    gene_pool = rng.permutation(n_genes)
    gp = 0
    coarse_genes, fine_genes = {}, {}
    for cn in clade_nodes:
        g = gene_pool[gp:gp + n_coarse]; gp += n_coarse
        coarse_genes[cn] = g
        rows = [ci[l] for l in tree.leaves_in_subtree(cn) if l in ci]
        E[np.ix_(rows, g)] += coarse_strength
    for cn in clade_nodes:
        if is_tree_fine[cn]:
            continue
        subs = sorted({sub_of[l] for l in tree.leaves_in_subtree(cn) if l in sub_of})
        for s in subs:
            g = gene_pool[gp:gp + n_fine]; gp += n_fine
            fine_genes[s] = g
            rows = [ci[l] for l in keep if sub_of[l] == s]
            E[np.ix_(rows, g)] += fine_strength
    assert gp <= n_genes, f"need more genes: used {gp} of {n_genes}"
    print(f"[sim] expression: {gp}/{n_genes} genes are markers "
          f"({n_coarse}/clade coarse, {n_fine}/subclade fine on expr-fine clades only)")

    xy_keep = xy[[pos[l] for l in keep]]
    clade_lab = np.array([clade_of[l] for l in keep])
    sub_lab = np.array([sub_of[l] for l in keep])

    # ---- disjoint populations: half become the scRNA-seq, half compose the spots ----
    # If spots are built from the SAME cells, every cell has a spot containing itself and
    # expression matching is near-exact regardless of how uninformative expression is.
    is_sc = np.zeros(n, dtype=bool)
    for s in np.unique(sub_lab):                        # stratify by subclade
        idx = np.where(sub_lab == s)[0]
        is_sc[rng.choice(idx, size=max(1, len(idx) // 2), replace=False)] = True

    sp_i = np.where(~is_sc)[0]
    lo, hi = xy_keep[sp_i].min(0), xy_keep[sp_i].max(0)
    nb = max(2, int(np.sqrt(len(sp_i) / cells_per_spot)))
    bx = np.clip(((xy_keep[sp_i] - lo) / np.maximum(hi - lo, 1e-9) * nb).astype(int), 0, nb - 1)
    bid = bx[:, 0] * nb + bx[:, 1]
    ub, inv = np.unique(bid, return_inverse=True)
    m = len(ub)
    M = np.zeros((m, len(sp_i))); M[inv, np.arange(len(sp_i))] = 1.0
    cnt = M.sum(1, keepdims=True)
    X_st = (M @ E[sp_i]) / cnt
    xy_st = (M @ xy_keep[sp_i]) / cnt
    gene_batch = np.exp(rng.normal(0.0, batch_sd, size=n_genes))
    X_st = X_st * gene_batch[None, :] + rng.normal(0.0, spot_noise, size=X_st.shape)

    sc_i = np.where(is_sc)[0]
    X_sc = E[sc_i]
    print(f"[sim] disjoint split: {len(sc_i)} cells for scRNA-seq, {len(sp_i)} cells "
          f"composing {m} spots ({cnt.mean():.1f} cells/spot); no cell is in its own spot")

    # spot answer key = dominant clade / subclade among its member cells
    def dom(lab):
        return np.array([pd.Series(lab[sp_i][inv == k]).value_counts().index[0] for k in range(m)])
    st_clade, st_sub = dom(clade_lab), dom(sub_lab)

    # ---- cost matrices ----
    W = utils.cophenetic_distance_matrix(tree)
    W = W.loc[leaves, leaves].to_numpy()
    ii = [pos[keep[i]] for i in sc_i]
    C_tree = W[np.ix_(ii, ii)]
    C_space = squareform(pdist(xy_st))

    # ---- design checks ----
    print("\n[check] does the design do what it says?")
    for name, mask in [("expr-fine", grp == 0), ("tree-fine", grp == 1)]:
        rs = []
        for cn in np.array(clade_nodes)[mask]:
            jj = np.where(clade_lab[sc_i] == cn)[0]
            if len(jj) < 8:
                continue
            dt = squareform(C_tree[np.ix_(jj, jj)], checks=False)
            dsp = pdist(xy_keep[sc_i][jj])
            rs.append(np.corrcoef(dt, dsp)[0, 1])
        print(f"  within-clade tree-vs-space r   {name:<10} = {np.mean(rs):+.3f}  (n={len(rs)})")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    for name, mask in [("expr-fine", grp == 0), ("tree-fine", grp == 1)]:
        cacc, facc = [], []
        for cn in np.array(clade_nodes)[mask]:
            y = (clade_lab[sc_i] == cn).astype(int)
            if y.sum() < 15:
                continue
            cacc.append(cross_val_score(LogisticRegression(max_iter=400, class_weight="balanced"),
                                        X_sc, y, cv=3, scoring="balanced_accuracy").mean())
            jj = np.where(y == 1)[0]                 # within this clade: can expression tell subclades?
            ysub = sub_lab[sc_i][jj]
            if len(np.unique(ysub)) > 1:
                facc.append(cross_val_score(LogisticRegression(max_iter=400, class_weight="balanced"),
                                            X_sc[jj], ysub, cv=3, scoring="balanced_accuracy").mean())
        print(f"  expression -> CLADE    (coarse) {name:<10} = {np.mean(cacc):.3f}")
        print(f"  expression -> SUBCLADE (fine)   {name:<10} = {np.mean(facc):.3f}")

    np.savez_compressed(
        HERE / out,
        X_sc=X_sc.astype(np.float32), X_st=X_st.astype(np.float32),
        C_tree=(C_tree / C_tree.max()).astype(np.float32),
        C_space=(C_space / max(C_space.max(), 1e-9)).astype(np.float32),
        xy_st=xy_st, xy_sc=xy_keep[sc_i],
        cell_clade=clade_lab[sc_i], cell_sub=sub_lab[sc_i],
        spot_clade=st_clade, spot_sub=st_sub,
        clades=np.array(clade_nodes),
        is_tree_fine=np.array([is_tree_fine[c] for c in clade_nodes], dtype=bool),
    )
    print(f"\n[sim] wrote {out}: {len(sc_i)} cells x {m} spots, {len(clade_nodes)} clades")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in [("n-cells", 3000), ("n-clades", 12), ("n-sub", 4), ("n-genes", 300),
                 ("seed", 0), ("n-coarse", 8), ("n-fine", 6)]:
        ap.add_argument(f"--{k}", type=int, default=v, dest=k.replace("-", "_"))
    for k, v in [("sep", 2.0), ("sub-sep", 2.0), ("coarse-strength", 2.5), ("fine-strength", 2.0),
                 ("expr-noise", 1.0), ("cells-per-spot", 3.0), ("batch-sd", 0.3),
                 ("spot-noise", 0.3)]:
        ap.add_argument(f"--{k}", type=float, default=v, dest=k.replace("-", "_"))
    ap.add_argument("--out", default="sim_cf.npz")
    main(**vars(ap.parse_args()))
