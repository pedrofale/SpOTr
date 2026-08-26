"""Build SpOTr inputs for the FULL eTRACER late-escape (LE) task.

Unlike prep_stage.py, the spatial side is NOT restricted to barcoded bins: the bin universe is
every LE bin that has both expression and coordinates (35,835 of them), of which only ~20%
carry a clone barcode. Those unlabelled bins are the whole point -- the model must map cells
onto them too, and the lineage loss is masked to the labelled minority (`is_labelled`).

Cell side: the 45 clones that have BOTH an sc ML tree (sc_trees_LE) and barcoded LE bins.
A cell is usable iff it is a leaf of its clone's tree (so the tree term is defined) AND it
resolves to a column of the exported sc matrix.  Clones are capped, not composition-matched:
proportional matching is capped at ~3.9k cells by C4 and starves the clones that cover most of
the mask, whereas the loss here is masked and alpha is per-clade, so a clone's cells only need
to support its own clade.

Writes etracer_LE_full.npz consumed by run_le_full.py.
"""
import argparse, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from ete3 import Tree
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
CACHE = HERE / "recon" / "cache"
STAGE = "LE"          # module default; main() overrides via the --stage flag
BASE_LABELLED = 0.211          # the briefed barcoded fraction, for the sampling guard rail
AWK = shutil.which("mawk") or shutil.which("gawk") or "awk"


# --------------------------------------------------------------------------- id matching
def key_mat(x):
    """export/sc/barcodes.tsv id -> (tumour, animal, 16bp). Verified injective on 91,411."""
    x = re.sub(r"-\d+$", "", x)
    p = x.split("_")
    if len(p) > 1 and p[0] == p[1]:
        p = p[1:]
    a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])


def key_lin(x):
    """sc_cell_lineage.csv cellBC -> same triple (drops the merge-artefact middle token)."""
    p = x.split("_")
    a = [t for t in p if re.fullmatch(r"a\d+", t)]
    return (p[0], a[0] if a else "", p[-1])


def clone_of(path):
    """C100-44.phy.treefile -> C100.  The -N suffix counts leaves INCLUDING `synthetic`."""
    return re.sub(r"-\d+$", "", path.name.replace(".phy.treefile", ""))


# --------------------------------------------------------------------------- tree distance
def cophenetic_bl(t, leaves):
    """Patristic (branch-length) distance over the given leaves only. Validated to 2e-16
    against ete3.get_distance; multifurcations and the polytomous root are handled."""
    idx = {l: i for i, l in enumerate(leaves)}
    assert len(idx) == len(leaves), "duplicate leaf name"
    have = {l.name for l in t.iter_leaves()}
    missing = [l for l in leaves if l not in have]
    assert not missing, f"leaves absent from tree would get all-zero rows: {missing[:5]}"
    D = np.zeros((len(leaves), len(leaves)))
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node._acc = [(idx[node.name], 0.0)] if node.name in idx else []
            continue
        kids = [c._acc for c in node.children]
        blen = [max(c.dist, 0.0) for c in node.children]
        for a in range(len(kids)):
            for b_ in range(a + 1, len(kids)):
                for i, di in kids[a]:
                    for j, dj in kids[b_]:
                        d = di + blen[a] + dj + blen[b_]
                        D[i, j] = D[j, i] = d
        node._acc = [(i, d + blen[k]) for k, kk in enumerate(kids) for i, d in kk]
    return D


# --------------------------------------------------------------------------- mtx extraction
def extract(mtx, row_ids, col_ids, tag, force=False):
    """Stream a genes x cells MatrixMarket file, keep (row_ids, col_ids), return cells x genes.

    row_ids / col_ids are 1-based original indices, in the order wanted in the output.
    Full mmread is not an option here: the sc matrix is 5.4 GB / 399M nnz.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{tag}.npz"
    if out.exists() and not force:
        z = np.load(out)
        if z["row_ids"].shape == row_ids.shape and (z["row_ids"] == row_ids).all() \
           and z["col_ids"].shape == col_ids.shape and (z["col_ids"] == col_ids).all():
            print(f"[prep] {tag}: reusing cache {out.name}")
            return coo_matrix((z["v"], (z["i"], z["j"])),
                              shape=(len(col_ids), len(row_ids))).tocsr()
    with tempfile.TemporaryDirectory() as td:
        gf, cf, of = Path(td) / "g", Path(td) / "c", Path(td) / "o"
        np.savetxt(gf, np.c_[row_ids, np.arange(1, len(row_ids) + 1)], fmt="%d")
        np.savetxt(cf, np.c_[col_ids, np.arange(1, len(col_ids) + 1)], fmt="%d")
        prog = r'''
        BEGIN{ while((getline l < GF)>0){split(l,a," "); G[a[1]]=a[2]}
               while((getline l < CF)>0){split(l,a," "); C[a[1]]=a[2]} }
        /^%/ {next}
        !h   {h=1; next}
        ($1 in G) && ($2 in C) { print G[$1], C[$2], $3 }
        '''
        print(f"[prep] {tag}: streaming {mtx.name} ({mtx.stat().st_size/1e9:.2f} GB) ...")
        with open(of, "w") as fh:
            subprocess.run([AWK, "-v", f"GF={gf}", "-v", f"CF={cf}", prog, str(mtx)],
                           stdout=fh, check=True)
        if of.stat().st_size == 0:
            raise RuntimeError(f"{tag}: extraction produced no entries -- index mismatch?")
        d = pd.read_csv(of, sep=" ", header=None, dtype=np.int64).to_numpy()
    i, j, v = d[:, 1] - 1, d[:, 0] - 1, d[:, 2].astype(np.float32)   # -> cells x genes
    np.savez_compressed(out, i=i, j=j, v=v, row_ids=row_ids, col_ids=col_ids)
    X = coo_matrix((v, (i, j)), shape=(len(col_ids), len(row_ids))).tocsr()
    print(f"[prep] {tag}: {X.shape[0]} obs x {X.shape[1]} genes, {X.nnz} nnz")
    return X


# --------------------------------------------------------------------------- main
MIN_PER_CLONE = 12


def main(n_bins=5000, n_cells=3000, n_hvg=2000, seed=0, force=False,
         match_composition=True, stage="LE", restrict_clones=None,
         max_bins_per_clone=None, out_suffix="full", stratified=False):
    global STAGE
    STAGE = stage
    n_cells_max = n_cells
    rng = np.random.default_rng(seed)

    # ---- 1. bin universe: ALL LE bins with expression + coordinates -------------------
    md = pd.read_csv(HERE / f"export/spatial_{STAGE}/metadata.csv", index_col=0)
    bar = [l.strip() for l in open(HERE / f"export/spatial_{STAGE}/barcodes.tsv")]
    md.index = [str(int(float(i))) for i in md.index]
    assert list(md.index) == bar, "metadata.csv row order != barcodes.tsv"
    bins_all = pd.DataFrame(dict(matrix_id=np.arange(1, len(bar) + 1),      # 1-based mtx column
                                 bc=bar, x=md.x.astype(int).to_numpy(),
                                 y=md.y.astype(int).to_numpy()))
    bins_all["cellid"] = bins_all.x.astype(str) + "_" + bins_all.y.astype(str)

    # the literal string "NA" is a real value in this file, not a missing marker
    lin_sp = pd.read_csv(B / f"spatial/spatial_lineage_{STAGE}.csv",
                         keep_default_na=False, na_values=[])
    lin_sp = lin_sp[lin_sp.lineageGrp != "NA"]
    lin_sp["lineageGrp"] = lin_sp.lineageGrp.astype(str)
    bins_all["clone_raw"] = bins_all.cellid.map(dict(zip(lin_sp.CellID, lin_sp.lineageGrp)))

    # ---- 2. the 45 clones with BOTH an sc tree and barcoded LE bins -------------------
    tree_files = {clone_of(f): f for f in (B / f"single_cell/sc_trees_{STAGE}").glob("*.phy.treefile")}
    shared = sorted(set(tree_files) & set(lin_sp.lineageGrp), key=lambda c: int(c[1:]))
    print(f"[prep] {len(tree_files)} sc trees, {lin_sp.lineageGrp.nunique()} spatial clones "
          f"-> {len(shared)} shared")

    # a bin is "labelled" only if its clone is one we can actually model (has a tree)
    bins_all["clone"] = np.where(bins_all.clone_raw.isin(shared), bins_all.clone_raw, "NA")
    bins_all["labelled"] = bins_all.clone != "NA"
    frac_pop = bins_all.labelled.mean()
    print(f"[prep] bin universe {len(bins_all)}, labelled {int(bins_all.labelled.sum())} "
          f"({100*frac_pop:.2f}%)  [{int(bins_all.clone_raw.notna().sum())} carry any barcode; "
          f"{int((bins_all.clone_raw.notna() & ~bins_all.labelled).sum())} belong to tree-less clones]")

    # ---- 2b. optional restriction to well-covered clones, with a per-clone bin cap -------
    # Per-clade alpha is only estimable for a clade that HAS labelled bins. In the full-tissue
    # run 10 of 45 clones drew zero and 8 of 23 drew fewer than five, alpha never differentiated
    # (span 0.091-0.109), and the alpha-vs-premise correlation climbed monotonically as starved
    # clones were dropped (+0.171 at >=1 bin, +0.286 at >=5, +0.543 at >=10). Restricting by
    # COVERAGE -- a criterion fixed before looking at any alpha -- keeps 23 of the LE clones and
    # 94% of the barcoded bins.
    #
    # The per-clone cap addresses a second problem in the same step: C4, C2 and C3 hold 6,021 of
    # 7,357 retained bins, so without a cap they dominate the mask and force the composition
    # mismatch that balanced OT punishes.
    if restrict_clones:
        keep_cl = set(Path(restrict_clones).read_text().split())
        shared = [c for c in shared if c in keep_cl]
        before = len(bins_all)
        bins_all.loc[~bins_all.clone.isin(shared), "clone"] = "NA"
        bins_all["labelled"] = bins_all.clone != "NA"
        frac_pop = bins_all.labelled.mean()
        print(f"[prep] restricted to {len(shared)} well-covered clones; labelled bins "
              f"{int(bins_all.labelled.sum())} of {before} ({100*frac_pop:.2f}%)")
    if max_bins_per_clone:
        rows = []
        for c, g in bins_all[bins_all.labelled].groupby("clone"):
            rows.append(g if len(g) <= max_bins_per_clone
                        else g.sample(max_bins_per_clone, random_state=seed))
        kept = pd.concat(rows).index
        drop = bins_all.index[bins_all.labelled].difference(kept)
        bins_all.loc[drop, "clone"] = "NA"
        bins_all.loc[drop, "labelled"] = False
        frac_pop = bins_all.labelled.mean()
        vc = bins_all[bins_all.labelled].clone.value_counts()
        print(f"[prep] capped at {max_bins_per_clone} bins/clone -> {int(bins_all.labelled.sum())} "
              f"labelled ({100*frac_pop:.2f}%), per-clone min {vc.min()} median "
              f"{int(vc.median())} max {vc.max()}")

    # ---- 3. bin selection -------------------------------------------------------------
    # Uniform sampling preserves the labelled fraction, which is what a realistic target looks
    # like. But once the per-clone cap has cut the labelled population to a few percent, a
    # uniform draw keeps almost none of it (capping to 400/clone took the population to 6.4%
    # and a 5,000-bin uniform sample then retained 318 labelled bins, fewer than without the
    # cap). When the point of the run is to estimate a per-clade parameter, every clade needs
    # bins, so stratify instead: keep ALL capped labelled bins and add unlabelled ones.
    # The model is never told which bins are labelled, so this changes what can be SCORED, not
    # what the model sees -- but it does raise the labelled fraction above the true 20.5%, and
    # that is a difference from the deployed setting worth stating.
    if stratified:
        lab_idx = np.where(bins_all.labelled.to_numpy())[0]
        unl_idx = np.where(~bins_all.labelled.to_numpy())[0]
        n_unl = max(0, min(len(unl_idx), n_bins - len(lab_idx)))
        take = np.sort(np.concatenate([lab_idx, rng.choice(unl_idx, size=n_unl, replace=False)]))
        bins_ = bins_all.iloc[take].reset_index(drop=True)
        print(f"[prep] STRATIFIED: all {len(lab_idx)} labelled bins + {n_unl} unlabelled "
              f"= {len(bins_)} ({100*len(lab_idx)/len(bins_):.1f}% labelled, vs "
              f"{100*frac_pop:.1f}% in the tissue -- deliberately enriched so every clade has "
              f"bins to estimate alpha_k from)")
    elif n_bins < len(bins_all):
        take = np.sort(rng.choice(len(bins_all), size=n_bins, replace=False))
        print(f"[prep] SUBSAMPLED {n_bins} of {len(bins_all)} bins UNIFORMLY AT RANDOM over the "
              f"whole tissue (labelled and unlabelled alike), so the labelled fraction is "
              f"preserved in expectation; memory (dense C_space is m^2) forces this.")
        bins_ = bins_all.iloc[take].reset_index(drop=True)
    else:
        bins_ = bins_all.copy()
        print(f"[prep] using all {len(bins_)} bins, no subsampling")

    # ---- 4. cell side: tree leaf AND resolves to an sc matrix column ------------------
    bcs = [l.strip() for l in open(HERE / "export/sc/barcodes.tsv")]
    mk = {key_mat(b): i + 1 for i, b in enumerate(bcs)}                 # -> 1-based mtx column
    assert len(mk) == len(bcs), "key_mat collides"
    lin_sc = pd.read_csv(B / "single_cell/sc_cell_lineage.csv")
    lin_sc["lineageGrp"] = lin_sc.lineageGrp.astype(str)
    lin_sc["matrix_id"] = lin_sc.cellBC.map(key_lin).map(mk)

    avail = {}
    for c in shared:
        t = Tree(str(tree_files[c]), format=1)
        leaves = {l.name for l in t.iter_leaves()} - {"synthetic"}      # rooting outgroup
        g = lin_sc[(lin_sc.lineageGrp == c) & lin_sc.cellBC.isin(leaves) & lin_sc.matrix_id.notna()]
        avail[c] = g.drop_duplicates("cellBC")
    n_av = {c: len(g) for c, g in avail.items()}
    print(f"[prep] usable cells (tree leaf & in matrix): {sum(n_av.values())} over {len(shared)} clones "
          f"(min {min(n_av.values())}, max {max(n_av.values())})")

    # Cell-side clone composition MUST track the labelled-bin composition. Balanced OT gives
    # every cell equal mass, so a uniform-per-clone reference IMPOSES a uniform-per-clone spot
    # composition, and the GW term then enforces that wrong composition through the tree's
    # block geometry. On this dataset that is not a subtlety: at composition TVD 0.63 the real
    # tree scored WORSE than no tree, and matching (TVD 0.011) flipped it to real > shuffled at
    # 5/5 alpha values (experiments/etracer/NOTES.md, "The critical condition").
    #
    # Water-filling caps give a near-uniform reference -- TVD 0.774 against these bins, worse
    # than the known-catastrophic case -- so proportional matching is the default here.
    target = (bins_[bins_.labelled].clone.value_counts(normalize=True)
              if "labelled" in bins_.columns else
              pd.Series({c: 1.0 / len(shared) for c in shared}))
    target = target.reindex(shared).fillna(0.0)
    if match_composition and target.sum() > 0:
        target = target / target.sum()
        # largest total whose per-clone demand is satisfiable from the usable pool
        feasible = min((n_av[c] / target[c]) for c in shared if target[c] > 0)
        n_tot = int(min(n_cells_max, feasible))
        want = {c: int(round(target[c] * n_tot)) for c in shared}
        # every clone needs at least a few cells or its alpha_k is unestimable
        for c in shared:
            want[c] = max(want[c], min(MIN_PER_CLONE, n_av[c]))
        rows = [avail[c].sample(min(want[c], n_av[c]), random_state=seed).assign(clone=c)
                for c in shared if want[c] > 0]
        cells = pd.concat(rows, ignore_index=True)
        got = cells.clone.value_counts(normalize=True).reindex(shared).fillna(0.0)
        tvd = 0.5 * float(np.abs(got - target).sum())
        print(f"[prep] composition-matched -> {len(cells)} cells, TVD vs labelled bins {tvd:.3f} "
              f"(floor {MIN_PER_CLONE}/clone; uncapped feasible max {int(feasible)})")
        # 0.63 was catastrophic (real tree worse than none), 0.011 was fine. Matching cannot
        # always reach 0.011: at EE, C18 holds 59% of the mask but supplies 244 cells, so the
        # match is capped by availability and lands at 0.374. That is still far better than the
        # 0.703 an unmatched cap gives, so warn rather than abort, and only fail near the danger
        # zone.
        if tvd >= 0.15:
            print(f"[prep] WARNING composition TVD {tvd:.3f} > 0.15 -- matching is capped by "
                  f"cell availability, not by choice. 0.63 was catastrophic in earlier work.")
        assert tvd < 0.50, f"composition matching failed: TVD {tvd:.3f} is near the danger zone"
    else:
        caps = np.arange(1, max(n_av.values()) + 1)
        tot = np.array([sum(min(v, int(k)) for v in n_av.values()) for k in caps])
        ok = np.where(tot <= n_cells_max)[0]
        cap = int(caps[ok[-1]]) if len(ok) else int(caps[0])
        rows = [(avail[c] if len(avail[c]) <= cap else avail[c].sample(cap, random_state=seed))
                .assign(clone=c) for c in shared]
        cells = pd.concat(rows, ignore_index=True)
        print(f"[prep] per-clone cap {cap} -> {len(cells)} cells (NOT composition-matched)")
    cells["matrix_id"] = cells.matrix_id.astype(int)

    # ---- 5. expression: shared gene space, log-normalised, HVGs from the spatial side --
    gsc = np.array([l.strip() for l in open(HERE / "export/sc/features.tsv")])
    gst = np.array([l.strip() for l in open(HERE / f"export/spatial_{STAGE}/features.tsv")])
    genes = sorted(set(gsc) & set(gst))
    rsc = np.array([{g: i + 1 for i, g in enumerate(gsc)}[g] for g in genes])
    rst = np.array([{g: i + 1 for i, g in enumerate(gst)}[g] for g in genes])

    Xsc = extract(HERE / "export/sc/matrix.mtx", rsc,
                  cells.matrix_id.to_numpy(), f"sc_{STAGE}_{len(cells)}c_{seed}", force)
    Xst = extract(HERE / f"export/spatial_{STAGE}/matrix.mtx", rst,
                  bins_.matrix_id.to_numpy(), f"st_{STAGE}_{len(bins_)}b_{seed}", force)
    assert Xsc.shape == (len(cells), len(genes)) and Xst.shape == (len(bins_), len(genes))
    assert (np.asarray(Xsc.sum(1)).ravel() > 0).all(), "cell with zero counts in shared genes"
    assert (np.asarray(Xst.sum(1)).ravel() > 0).all(), "bin with zero counts in shared genes"

    asc, ast = sc.AnnData(Xsc), sc.AnnData(Xst)
    asc.var_names = ast.var_names = genes
    for a in (asc, ast):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    sc.pp.highly_variable_genes(ast, n_top_genes=min(n_hvg, len(genes)))
    hvg = list(ast.var_names[ast.var.highly_variable])
    asc, ast = asc[:, hvg].copy(), ast[:, hvg].copy()
    dense = lambda X: np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    X_sc, X_st = dense(asc.X), dense(ast.X)
    print(f"[prep] shared genes {len(genes)} -> HVG {len(hvg)} (selected on the spatial bins)")

    # ---- 6. structural costs ---------------------------------------------------------
    n = len(cells)
    C_tree = np.zeros((n, n), dtype=np.float32)
    pos = {cb: i for i, cb in enumerate(cells.cellBC)}
    within_max = 0.0
    for clone, g in cells.groupby("clone"):
        t = Tree(str(tree_files[clone]), format=1)
        leaves = list(g.cellBC)
        sub = cophenetic_bl(t, leaves)
        ii = [pos[c] for c in leaves]
        C_tree[np.ix_(ii, ii)] = sub
        within_max = max(within_max, float(sub.max()))
    # separate clones are independent lineages: no shared ancestry to measure, so the
    # cross-clone entry is a constant beyond every within-clone distance
    same = cells.clone.to_numpy()[:, None] == cells.clone.to_numpy()[None, :]
    C_tree[~same] = 2.0 * within_max
    np.fill_diagonal(C_tree, 0.0)
    C_tree = 0.5 * (C_tree + C_tree.T)
    print(f"[prep] C_tree: within-clone max {within_max:.3f}, cross-clone {2*within_max:.3f}")

    xy = bins_[["x", "y"]].to_numpy(float)
    C_space = squareform(pdist(xy)).astype(np.float32)
    C_space /= max(C_space.max(), 1e-12)

    clones = np.array(shared)
    cell_clone = cells.clone.to_numpy().astype(str)
    bin_clone = bins_.clone.to_numpy().astype(str)          # "NA" for unlabelled
    is_labelled = bins_.labelled.to_numpy().astype(bool)
    ci = {c: i for i, c in enumerate(clones)}
    A = np.zeros((n, len(clones)), dtype=np.float32)
    A[np.arange(n), [ci[c] for c in cell_clone]] = 1.0

    # ---- 7. guard rails --------------------------------------------------------------
    out = dict(X_sc=X_sc, X_st=X_st, C_tree=C_tree, C_space=C_space, xy=xy.astype(np.float32),
               A=A, clones=clones, cell_clone=cell_clone, bin_clone=bin_clone,
               is_labelled=is_labelled)
    for k, v in out.items():
        if np.issubdtype(np.asarray(v).dtype, np.floating):
            assert np.isfinite(v).all(), f"non-finite value in {k}"
    assert (C_tree == C_tree.T).all(), "C_tree not symmetric"
    assert (np.diag(C_tree) == 0).all(), "C_tree diagonal not zero"
    assert (C_space == C_space.T).all() and (np.diag(C_space) == 0).all(), "C_space malformed"
    assert set(cell_clone) <= set(clones), "cell clone outside `clones`"
    assert set(bin_clone) <= set(clones) | {"NA"}, "bin clone outside `clones` + NA"
    assert (is_labelled == (bin_clone != "NA")).all(), "is_labelled disagrees with bin_clone"
    assert A.sum() == n and (A.sum(1) == 1).all(), "A is not one-hot"
    assert (A.argmax(1) == np.array([ci[c] for c in cell_clone])).all(), "A rows mislabelled"
    # Guard the SAMPLING, not a hard-coded stage: the realised labelled fraction must be
    # consistent with binomial draws from this stage's own population fraction. A fixed
    # [0.15, 0.30] band is an LE assumption and rejects a perfectly good EE sample (population
    # 35.3%, realised 35.4%).
    frac = float(is_labelled.mean())
    if stratified:
        print(f"[prep] sampling check skipped: stratified draw, labelled fraction {frac:.4f} "
              f"is enriched over the population {frac_pop:.4f} by design")
    else:
        sd = float(np.sqrt(max(frac_pop * (1 - frac_pop), 1e-12) / len(bins_)))
        lo, hi = frac_pop - 5 * sd, frac_pop + 5 * sd
        assert lo <= frac <= hi, (f"labelled fraction {frac:.4f} outside [{lo:.4f}, {hi:.4f}] "
                                  f"= population {frac_pop:.4f} +- 5 binomial sd; not uniform")
        print(f"[prep] sampling check: realised {frac:.4f} vs population {frac_pop:.4f} "
              f"({abs(frac - frac_pop) / sd:.1f} sd) -- uniform")

    np.savez_compressed(HERE / f"etracer_{STAGE}_{out_suffix}.npz", **out)

    # ---- 8. summary ------------------------------------------------------------------
    lb = pd.Series(bin_clone[is_labelled]).value_counts()
    cc = pd.Series(cell_clone).value_counts()
    print("\n" + "=" * 70)
    print(f"etracer_{STAGE}_{out_suffix}.npz")
    print("=" * 70)
    print(f"cells                {n}")
    print(f"bins                 {len(bins_)}   (of {len(bins_all)} in the LE tissue)")
    print(f"labelled bins        {int(is_labelled.sum())}  = {100*frac:.2f}% "
          f"(population {100*frac_pop:.2f}%, brief {100*BASE_LABELLED:.1f}%)")
    print(f"clones               {len(clones)}")
    print(f"HVGs                 {X_sc.shape[1]}  (from {len(genes)} shared genes)")
    print(f"X_sc {X_sc.shape}  X_st {X_st.shape}  C_tree {C_tree.shape}  C_space {C_space.shape}")
    print(f"\n{'clone':>7}{'cells':>8}{'lab.bins':>10}   {'clone':>7}{'cells':>8}{'lab.bins':>10}   "
          f"{'clone':>7}{'cells':>8}{'lab.bins':>10}")
    rows = [(c, int(cc.get(c, 0)), int(lb.get(c, 0))) for c in clones]
    third = -(-len(rows) // 3)
    cols = [rows[i*third:(i+1)*third] for i in range(3)]
    for r in range(third):
        line = ""
        for cl in cols:
            line += (f"{cl[r][0]:>7}{cl[r][1]:>8}{cl[r][2]:>10}   " if r < len(cl) else " " * 28)
        print(line.rstrip())
    print(f"{'TOTAL':>7}{n:>8}{int(is_labelled.sum()):>10}")
    print(f"\nunlabelled bins      {int((~is_labelled).sum())}  (bin_clone == 'NA')")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-bins", type=int, default=5000)
    p.add_argument("--n-cells", type=int, default=3000)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="LE", choices=["BL", "EE", "LE"])
    p.add_argument("--restrict-clones", default=None,
                   help="file with one clone id per line; others become NA")
    p.add_argument("--max-bins-per-clone", type=int, default=None)
    p.add_argument("--out-suffix", default="full")
    p.add_argument("--stratified", action="store_true",
                   help="keep all labelled bins + unlabelled filler, instead of a uniform draw")
    p.add_argument("--force", action="store_true", help="ignore the extracted-matrix cache")
    p.add_argument("--no-match-composition", dest="match_composition",
                   action="store_false", help="use water-filling caps instead")
    main(**{k.replace("-", "_"): v for k, v in vars(p.parse_args()).items()})
