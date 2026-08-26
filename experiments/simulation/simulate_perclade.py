"""Positive control: can the algorithm recover per-clade spatial concordance?

The spatial simulator applies Pagel's lambda PER CLADE (`rescale_clades`), scaling the
within-clade covariance block. lambda = 1 keeps the clade's internal phylogenetic structure
in its positions (spatially concordant); lambda = 0 removes it, so the clade still occupies
its own territory but its internal tree predicts nothing (discordant).

So per-clade lambda IS per-clade concordance, by construction. If SpOTr's per-clade phi
cannot recover a known lambda here, the null on real data is an algorithm limitation rather
than a data limitation.
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



def _simulate_expression_alpha(utils, tree, alpha, n_traits, n_genes, seed, clade_labels,
                               alpha_per_clade=None, marker_clades=None, marker_strength=3.0):
    """utils.simulate_expression with the alpha blend actually applied.

    The stock ExpressionSimulator accepts `alpha` and never uses it, so expression is always
    fully tree-determined and the GW term is always redundant. Here the external component
    is the tree-driven activation matrix permuted across cells -- same distribution and
    scale, no phylogenetic structure -- so alpha varies ONLY how much lineage information
    expression carries.

        alpha = 1 -> expression fully tree-determined (tree term redundant)
        alpha = 0 -> expression carries no lineage information (tree is the only source)
    """
    import numpy as _np, pandas as _pd, anndata as _ad
    _vs = _load("visium_simulator")
    _es = _load("expression_simulator")

    class ExpressionSimulatorV2(_es.ExpressionSimulator):
        def sample_combined_expression(self, tree, alpha=0.5, clades=None, rates=None,
                                       noise_std=0.5):
            rng = getattr(self, "_rng", None) or _np.random.default_rng()
            bm_act = self.sample_brownian_motion(clades=clades, rates=rates)
            bm_prop = _np.exp(bm_act) / _np.sum(_np.exp(bm_act), axis=1).values[:, None]
            perm = rng.permutation(bm_prop.shape[0])
            ext_prop = _pd.DataFrame(_np.asarray(bm_prop)[perm],
                                     index=bm_prop.index, columns=bm_prop.columns)
            # alpha may be a scalar OR a per-cell vector, so different clades can differ in
            # how much lineage information their EXPRESSION carries. That is what makes
            # per-clade weighting meaningful: a clade with strong expression signal needs no
            # tree, one without it needs the tree.
            amap = getattr(self, "_alpha_map", None)
            if amap is not None:
                lab = _pd.Series(getattr(self, "_clade_labels")).reindex(list(self.leaf_names))
                av = lab.map(amap).fillna(float(alpha)).to_numpy(float)
                al = _np.clip(av, 0, 1)[:, None]
            else:
                al = float(_np.clip(alpha, 0.0, 1.0))
            mixed = al * bm_prop + (1.0 - al) * ext_prop
            clone_expression = mixed.dot(self.trait_signatures)
            sp_expr, sp_act = self.sample_spatial_effects(tree)
            expression = clone_expression + sp_expr + self.sample_expression_noise(noise_std=noise_std)
            return expression, mixed, sp_act, None

    sig = _np.zeros((n_traits, n_genes))
    for k, gi in enumerate(_np.array_split(_np.arange(n_genes), n_traits)):
        sig[k, gi] = 1
    sig = _pd.DataFrame(sig, index=[f"T{i}" for i in range(n_traits)],
                        columns=[f"G{i}" for i in range(n_genes)])
    bs = n_traits // 2
    cov = _np.zeros((n_traits, n_traits))
    cov[:bs, :bs] = _np.ones((bs, bs)) * 0.8 + _np.eye(bs) * 0.2
    cov[bs:, bs:] = _np.ones((n_traits - bs, n_traits - bs)) * 0.8 + _np.eye(n_traits - bs) * 0.2
    cov = _pd.DataFrame(cov, index=sig.index, columns=sig.index)
    smap = _vs.prior(200, _vs.circ_equation, decay_factor=.0005,
                     radius=50, center_x=100, center_y=100)
    prog = _np.zeros((n_genes,)); prog[_np.random.choice(n_genes, size=5)] = 1.

    ex = ExpressionSimulatorV2(cov, sig, smap, prog)
    ex._rng = _np.random.default_rng(seed)
    if alpha_per_clade is not None:
        ex._alpha_map = alpha_per_clade          # resolved lazily; leaf_names exists by then
        ex._clade_labels = clade_labels
    ex.overlay_data(tree, alpha=alpha, clades=None, rates=None)

    expr = tree.cell_meta[[f"G{i}" for i in range(ex.n_genes)]].loc[ex.leaf_names]

    # The expr_alpha blend alone gives only a weak clade signal (balanced accuracy ~0.57 vs
    # 0.49 at chance). To make "clear expression signal" mean something, give each
    # expression-driven clade its own dedicated marker genes.
    if marker_clades:
        _rng2 = _np.random.default_rng(seed + 1)
        lab = _pd.Series(clade_labels).reindex(list(ex.leaf_names))
        pool = list(expr.columns)
        for ci, cl in enumerate(sorted(marker_clades)):
            genes = _rng2.choice(pool, size=6, replace=False)
            rows = (lab == cl).to_numpy()
            expr.loc[rows, list(genes)] = expr.loc[rows, list(genes)].to_numpy() + marker_strength
    acts = tree.cell_meta[[f"T{i}" for i in range(ex.n_traits)]].loc[ex.leaf_names]
    sp_act = tree.cell_meta[ex.spatial_activation_name].loc[ex.leaf_names]
    obs_expr = _np.random.normal(expr, 0.1)
    ad = _ad.AnnData(_pd.DataFrame(obs_expr, index=expr.index, columns=expr.columns))
    ad.raw = ad.copy()
    ad.layers["expression"] = expr.values
    ad.uns["trait_signatures"] = ex.trait_signatures
    ad.obsm["trait_activations"] = acts
    for t in ex.trait_names:
        ad.obs[t] = acts[t]
    ad.obs["spatial_activations"] = sp_act
    ad.obs["library_id"] = "simulated_puck"
    ad.obsm["spatial"] = tree.cell_meta[["spatial_0", "spatial_1"]].loc[expr.index].to_numpy()

    # simulate_visium requires clade_level2; take it from the caller's partition so it is
    # defined for every cell regardless of tree shape
    lab = _pd.Series(clade_labels).reindex(ad.obs_names).astype(str)
    ad.obs["clade_level0"] = "root"
    ad.obs["clade_level1"] = lab.values
    ad.obs["clade_level2"] = lab.values
    return ad


def main(n_cells=1024, depth=5, n_genes=200, n_traits=8, seed=0, spread=1.0,
         expr_alpha=0.2, batch_sd=0.5, spot_noise=0.5, out="sim_inputs.npz"):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    utils = _load("utils"); space_sim = _load("space_simulator")

    # CompleteBinarySimulator gives a perfectly balanced ultrametric tree, where every leaf
    # has an IDENTICAL average distance to every other leaf. The GW linearisation is then
    # additive f(i)+g(j) and double-centering annihilates it -- |L|/|F| ~ 4e-6, i.e. the GW
    # term is numerically zero and alpha cannot do anything. A birth-death tree is unbalanced
    # (row-mean sd/mean 4.3e-2, vs 1.9e-1 for the real eTRACER tree) and keeps the term alive.
    ts = cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution=lambda scale: np.random.exponential(scale),
        initial_birth_scale=0.5,
        death_waiting_distribution=lambda: np.random.exponential(1.5),
        mutation_distribution=lambda: 0,
        fitness_distribution=lambda: np.random.normal(0, 0.5),
        num_extant=n_cells, random_seed=seed)
    tree = ts.simulate_tree()

    # Depth is not uniform on an unbalanced tree, so partition by repeatedly splitting the
    # largest subtree until we have the requested number of clades.
    n_clades = 2 ** depth
    parts = [tree.root]
    while len(parts) < n_clades:
        parts.sort(key=lambda x: -len(tree.leaves_in_subtree(x)))
        big = parts.pop(0)
        kids = tree.children(big)
        if not kids:
            parts.append(big); break
        parts.extend(kids)
    clade_nodes = [x for x in parts if len(tree.leaves_in_subtree(x)) >= 8]
    clades = [tree.leaves_in_subtree(n) for n in clade_nodes]
    covered = sum(len(c) for c in clades)
    print(f"[sim] tree: {len(tree.leaves)} leaves, {len(clade_nodes)} clades "
          f"(sizes {min(len(c) for c in clades)}-{max(len(c) for c in clades)}), "
          f"covering {covered}/{len(tree.leaves)} leaves")

    # true per-clade concordance, cycled so it is uncorrelated with clade identity/size
    # CROSSED design. Per-clade weighting only has anything to recover if clades differ in
    # WHICH modality is informative:
    #   expression-driven : expression carries lineage (expr_alpha 1), no spatial structure
    #                       (lambda 0)  -> the GW term is unnecessary, alpha_k should be LOW
    #   tree-driven       : expression carries none (expr_alpha 0), strong spatial structure
    #                       (lambda 1)  -> the GW term is required, alpha_k should be HIGH
    # Varying lambda alone (the earlier design) leaves every clade wanting the same weight,
    # which is why a global alpha matched the oracle.
    n_cl = len(clades)
    grp = np.array([0, 1] * (n_cl // 2 + 1))[:n_cl]
    rng.shuffle(grp)
    lams = np.where(grp == 1, 1.0, 0.0).astype(float)
    expr_alpha_per_clade = {clade_nodes[i]: (0.0 if grp[i] == 1 else 1.0) for i in range(n_cl)}
    print(f"[sim] crossed design: {int((grp==0).sum())} expression-driven "
          f"(expr_alpha 1, lambda 0), {int((grp==1).sum())} tree-driven "
          f"(expr_alpha 0, lambda 1)")
    print(f"[sim] {len(tree.leaves)} cells, {len(clades)} clades of {len(clades[0])} "
          f"| lambda counts {dict(zip(*np.unique(lams, return_counts=True)))}")

    # `rescale_clade` scales the whole within-clade covariance block INCLUDING its diagonal,
    # so every within-clade distance scales by sqrt(lam) and the tree-vs-space CORRELATION is
    # unchanged -- lam only destroys concordance at exactly 0. To get a graded knob we mix the
    # tree-driven deviation with iid noise explicitly, preserving the clade centroid (and hence
    # the between-clade layout) while grading only the within-clade phylogenetic signal.
    sp = space_sim.BrownianSpatialDataSimulator(2, 1.)
    sp.overlay_data(tree, brownian_motion=True, lambda_brownian=1.)
    xy0 = tree.cell_meta[["spatial_0", "spatial_1"]].loc[tree.leaves].to_numpy(float)
    pos = {l: i for i, l in enumerate(tree.leaves)}
    xy_mix = xy0.copy()
    for node, leaves, lam in zip(clade_nodes, clades, lams):
        ii = [pos[l] for l in leaves]
        blk = xy0[ii]
        cen = blk.mean(0)
        dev = blk - cen
        noise = rng.normal(size=dev.shape)
        noise *= dev.std(0, keepdims=True) / np.maximum(noise.std(0, keepdims=True), 1e-9)
        xy_mix[ii] = cen + np.sqrt(lam) * dev + np.sqrt(1.0 - lam) * noise
    tree.cell_meta["spatial_0"] = pd.Series(xy_mix[:, 0], index=tree.leaves)
    tree.cell_meta["spatial_1"] = pd.Series(xy_mix[:, 1], index=tree.leaves)

    cell_clade = pd.Series(index=tree.leaves, dtype=object)
    for node, leaves in zip(clade_nodes, clades):
        cell_clade.loc[leaves] = node

    # ---- sanity check: does lambda actually produce the intended concordance? ----
    xy_cell = tree.cell_meta[["spatial_0", "spatial_1"]].loc[tree.leaves].to_numpy(float)
    W = utils.cophenetic_distance_matrix(tree)          # ~500x faster, numerically identical
    W = W.loc[tree.leaves, tree.leaves].to_numpy()
    rows = []
    for node, leaves in zip(clade_nodes, clades):
        ii = [tree.leaves.index(l) for l in leaves]
        dt = squareform(W[np.ix_(ii, ii)], checks=False)
        ds = pdist(xy_cell[ii])
        rows.append(dict(clade=node, lam=float(lams[clade_nodes.index(node)]),
                         measured_r=float(np.corrcoef(dt, ds)[0, 1]), n=len(leaves)))
    chk = pd.DataFrame(rows)
    from scipy import stats
    pr, pp = stats.pearsonr(chk.lam, chk.measured_r)
    print(f"[sim] SANITY: corr(true lambda, measured within-clade concordance) = "
          f"{pr:+.3f} (p = {pp:.2e})")
    print(chk.groupby("lam").measured_r.mean().to_string())

    # ---- expression + spot binning ----
    # expression_mode sets how much of expression the TREE explains:
    #   "tree"     alpha=1.0  -- expression fully tree-determined; the GW term is then
    #                            redundant and SpOTr's premise is inverted
    #   "mix"      alpha=0.2  -- mostly external factors; the tree carries information
    #                            expression does not, which is SpOTr's actual use case
    #   "external" alpha=0.0  -- no tree signal in expression at all
    ad_sc = _simulate_expression_alpha(utils, tree, expr_alpha, n_traits, n_genes, seed,
                                       cell_clade, alpha_per_clade=expr_alpha_per_clade,
                                       marker_clades=[clade_nodes[i] for i in range(n_cl) if grp[i] == 0])
    # simulate_visium bins on a fixed 0.05 grid and Brownian spread grows only
    # logarithmically with cell count, so adding cells packs them into existing spots rather
    # than creating new ones -- scaling the coordinates is what actually controls spot count.
    # It must happen AFTER simulate_expression, whose spatial_map is a fixed 200x200 grid
    # that scaled coordinates overflow (IndexError). Pure rescaling, so correlations and the
    # OT problem (which normalises C_space) are unaffected.
    ad_sc.obsm["spatial"] = np.asarray(ad_sc.obsm["spatial"], dtype=float) * spread
    _rng3 = np.random.default_rng(seed + 2)

    # THE LEAK: simulate_visium builds each spot from the SAME cells that form the
    # single-cell side, so every cell has a spot containing itself and expression matching is
    # near-exact (ARI ~0.81 at alpha=0) no matter how uninformative expression is per clade.
    # Real unmatched data has two disjoint populations. Split the cells: half become the
    # scRNA-seq, half compose the spots. No cell then appears in its own spot.
    _rs = np.random.default_rng(seed + 3)
    all_cells = np.array(ad_sc.obs_names)
    is_sc = np.zeros(len(all_cells), dtype=bool)
    for c in np.unique(cell_clade.reindex(all_cells).astype(str)):
        idx = np.where(cell_clade.reindex(all_cells).astype(str).to_numpy() == c)[0]
        pick = _rs.choice(idx, size=max(1, len(idx) // 2), replace=False)
        is_sc[pick] = True                     # stratified so both halves keep every clade
    # simulate_visium writes the 'spot' assignment onto the object it is GIVEN, so hand it
    # the persistent spot_source rather than a temporary copy
    spot_source = ad_sc[~is_sc].copy()
    ad_st = utils.simulate_visium(spot_source, obs_model="normal", sigma=0.1)
    Xst = np.asarray(ad_st.X, dtype=float)
    gene_batch = np.exp(_rng3.normal(0.0, batch_sd, size=Xst.shape[1]))
    ad_st.X = (Xst * gene_batch[None, :]
               + _rng3.normal(0.0, spot_noise, size=Xst.shape)).astype(np.float32)
    ad_sc = ad_sc[is_sc].copy()
    print(f"[sim] disjoint split: {is_sc.sum()} cells for scRNA-seq, "
          f"{(~is_sc).sum()} cells composing {ad_st.shape[0]} spots "
          f"(no cell appears in its own spot)")

    spot_of = spot_source.obs["spot"].astype(str)
    # a size-based partition of an unbalanced tree can leave a few small subtrees uncovered;
    # drop those cells rather than carrying NaN clade labels downstream
    order = [c for c in tree.leaves
             if c in set(ad_sc.obs_names) and pd.notna(cell_clade.get(c))]
    ad_sc = ad_sc[order].copy()
    cl = cell_clade.loc[order].to_numpy()
    clades_u = sorted(set(cl), key=str)      # birth-death node names are not integers
    A = np.zeros((len(order), len(clades_u)), dtype=np.float32)
    for i, c in enumerate(cl):
        A[i, clades_u.index(c)] = 1.0

    # each spot's dominant clade = the answer key
    spot_ids = list(ad_st.obs_names.astype(str))
    sp_cells = [c for c in spot_source.obs_names if pd.notna(cell_clade.get(c))]
    dom = pd.Series(spot_of.loc[sp_cells].values, index=sp_cells).to_frame("spot")
    dom["clade"] = cell_clade.loc[sp_cells].to_numpy()
    spot_clade = (dom.groupby("spot").clade
                  .agg(lambda s: s.value_counts().index[0]).reindex(spot_ids))

    keep = spot_clade.notna().to_numpy()
    ad_st = ad_st[keep].copy(); spot_clade = spot_clade[keep]
    xy = np.asarray(ad_st.obsm["spatial"], dtype=float)

    Wsub = W[np.ix_([tree.leaves.index(c) for c in order], [tree.leaves.index(c) for c in order])]
    np.savez_compressed(
        HERE / out,
        X_sc=np.asarray(ad_sc.X, dtype=np.float32), X_st=np.asarray(ad_st.X, dtype=np.float32),
        C_tree=(Wsub / Wsub.max()).astype(np.float32),
        C_space=(squareform(pdist(xy)) / max(pdist(xy).max(), 1e-9)).astype(np.float32),
        xy=xy, A=A, clones=np.array(clades_u), cell_clone=cl,
        bin_clone=spot_clade.to_numpy().astype(str),
        true_lambda=np.array([lams[clade_nodes.index(c)] for c in clades_u]),
        expr_alpha_clade=np.array([expr_alpha_per_clade[c] for c in clades_u]),
        measured_r=chk.set_index("clade").measured_r.reindex(clades_u).to_numpy(),
    )
    print(f"[sim] expr_alpha={expr_alpha}; wrote {out}: {len(order)} cells x {keep.sum()} spots, {len(clades_u)} clades")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in [("n-cells", 1024), ("depth", 5), ("n-genes", 200), ("seed", 0)]:
        ap.add_argument(f"--{k}", type=int, default=v, dest=k.replace("-", "_"))
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--expr-alpha", type=float, default=0.2, dest="expr_alpha")
    ap.add_argument("--out", default="sim_inputs.npz")
    ap.add_argument("--batch-sd", type=float, default=0.5, dest="batch_sd")
    ap.add_argument("--spot-noise", type=float, default=0.5, dest="spot_noise")
    main(**vars(ap.parse_args()))
