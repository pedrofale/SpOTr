"""Build SpOTr inputs for the PEtracer (koblan25 tumor) sections, one .npz per section.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
PEtracer reads the lineage barcode IN SITU: every traced cell has BOTH a tree position and a
measured (x, y). It is therefore not natively an unmatched sc-to-spatial mapping problem. We
CONSTRUCT one, exactly as the simulator validation did: the traced cells of a single section are
split DISJOINTLY, ~50/50 stratified by clade. One half is treated as the dissociated scRNA-seq
side (it keeps the tree, and its true coordinates are held out as the answer key, `xy_sc`); the
other half is binned onto a square lattice into pseudo-spots. Because the split is disjoint, no
cell ever appears in its own spot -- that is asserted, not assumed.

This construction MUST be labelled as a construction in any writeup. It is used because it is
the only real dataset where both SpOTr claims can be tested at once: it has per-cell ground truth
AND an in-situ lineage anchor. The premise is section-replicated (coherence 0.241 / 0.247 /
0.359, Mantel +0.500 / +0.530 / +0.328, all p < 0.002, signs agreeing) -- see
petracer_premise_persection.py. Compare eTRACER late escape, coherence 0.309, which delivered
+0.108 clone accuracy.

Sections are handled ONE AT A TIME. obsm["spatial"] tiles the four samples into disjoint x
ranges, so a cross-section euclidean distance measures tile layout, not tissue; every coordinate
here is taken from a single sample and re-origined to that section's own bounding box.

ARRAY CONTRACT (same as experiments/etracer/prep_le_full.py)
------------------------------------------------------------
  X_sc  [n, g]      log-normalised dissociated cells
  X_st  [m, g]      log-normalised pseudo-spots, SAME gene space (the full 175-gene MERFISH panel)
  C_tree  [n, n]    patristic distance over the dissociated cells, from obst["tree"]
  C_space [m, m]    euclidean on spot coordinates, divided by its max
  xy    [m, 2]      spot lattice centres
  xy_sc [n, 2]      TRUE coordinates of the dissociated cells (held-out answer key)
  cell_clade [n], spot_clade [m], clades [K], A [n, K] one-hot, is_labelled [m]
  anchor_frac_grid  candidate anchor coverages -- the mask itself is drawn at RUN time

is_labelled is all-True here: every pseudo-spot is built from traced cells, so every spot has a
clade. That is unrealistically generous, so the npz also carries `anchor_frac_grid`, the
coverages a runner should subsample the anchor set down to. The realised mask is deliberately NOT
baked in, so a runner can average over draws rather than inherit one arbitrary one.

TREE METRIC: obst["tree"] edges carry no length attribute, but every node carries `time`, so
branch length = time(child) - time(parent) and the distances below are true patristic distances.
The script detects this and refuses to pass off a hop-count fallback silently -- see
`tree_branch_lengths`.

Usage:  python prep_petracer.py                       # all three traced sections
        python prep_petracer.py --sections M3-4       # just one
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
SECTIONS = ["M3-4", "M3-1", "M3-2"]          # largest first, as briefed
ANCHOR_FRAC_GRID = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 1.00], dtype=np.float32)


# --------------------------------------------------------------------- tree branch lengths
def tree_branch_lengths(g):
    """Return (blen dict keyed by (u, v), metric_name, evidence string).

    Priority: an explicit edge length attribute > node `time` differences > hop counts.
    The metric name is printed in the summary. A hop-count tree is a strictly weaker metric --
    it throws away every rate difference between branches -- so it must never be assumed
    silently; if it is what we fall back to, the summary says so in capitals.
    """
    edge_len_keys = ("length", "weight", "dist", "branch_length", "brlen")
    for k in edge_len_keys:
        vals = nx.get_edge_attributes(g, k)
        if len(vals) == g.number_of_edges():
            v = np.asarray(list(vals.values()), float)
            return ({e: float(x) for e, x in vals.items()}, f"edge attribute '{k}'",
                    f"all {g.number_of_edges()} edges carry '{k}'; "
                    f"range {v.min():.4g}..{v.max():.4g}, {len(np.unique(v))} distinct")

    tm = nx.get_node_attributes(g, "time")
    if len(tm) == g.number_of_nodes():
        bl = {(u, v): float(tm[v]) - float(tm[u]) for u, v in g.edges()}
        a = np.asarray(list(bl.values()))
        if (a >= -1e-9).all():
            a = np.maximum(a, 0.0)
            dep = nx.get_node_attributes(g, "depth")
            r = (np.corrcoef([tm[n] for n in g], [dep[n] for n in g])[0, 1]
                 if len(dep) == g.number_of_nodes() else float("nan"))
            return ({e: max(x, 0.0) for e, x in bl.items()}, "node 'time' differences",
                    f"all {g.number_of_nodes()} nodes carry 'time'; branch lengths "
                    f"{a.min():.4g}..{a.max():.4g}, mean {a.mean():.4g}, "
                    f"{len(np.unique(np.round(a, 9)))} distinct over {len(a)} edges, "
                    f"{int((a == 0).sum())} zero-length, {int((a < 0).sum())} negative; "
                    f"corr(node time, node depth) = {r:.3f} -- times are NOT a relabelled hop count")

    return ({e: 1.0 for e in g.edges()}, "HOP COUNTS (no branch lengths found)",
            "no edge length attribute and no usable node 'time': every branch forced to 1.0")


def patristic(g, root, leaves, blen):
    """Patristic distance among `leaves` (in the given order) over the whole tree `g`.

    One postorder pass. Each node accumulates (leaf slot, distance from that node); when a node
    is reached, every pair of leaves whose LCA is that node gets filled by an outer sum, so the
    total work is the n^2/2 entries themselves and nothing more.
    """
    idx = {l: i for i, l in enumerate(leaves)}
    assert len(idx) == len(leaves), "duplicate leaf name"
    have = set(g.nodes)
    missing = [l for l in leaves if l not in have]
    assert not missing, f"leaves absent from the tree would get all-zero rows: {missing[:5]}"
    n = len(leaves)
    D = np.zeros((n, n), dtype=np.float64)
    acc = {}                                    # node -> (int idx array, float dist array)
    for node in nx.dfs_postorder_nodes(g, root):
        kids = list(g.successors(node))
        if not kids:
            acc[node] = ((np.array([idx[node]], np.int64), np.zeros(1))
                         if node in idx else (np.zeros(0, np.int64), np.zeros(0)))
            continue
        parts = []
        for c in kids:
            ci, cd = acc.pop(c)
            if len(ci):
                parts.append((ci, cd + blen[(node, c)]))
        for a in range(len(parts)):
            ia, da = parts[a]
            for b in range(a + 1, len(parts)):
                ib, db = parts[b]
                blk = da[:, None] + db[None, :]
                D[np.ix_(ia, ib)] = blk
                D[np.ix_(ib, ia)] = blk.T
        acc[node] = ((np.concatenate([p[0] for p in parts]),
                      np.concatenate([p[1] for p in parts])) if parts
                     else (np.zeros(0, np.int64), np.zeros(0)))
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------- pseudo-spot lattice
def bin_cells(P, pitch):
    """Square lattice. Returns (spot id per cell, unique keys, counts per occupied spot)."""
    ij = np.floor((P - P.min(0)) / pitch).astype(np.int64)
    key = ij[:, 0] * (ij[:, 1].max() + 2) + ij[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return inv, ij, cnt


def choose_pitch(P, lo=3.0, hi=6.0, target=4.5, n_try=600):
    """Smallest-deviation pitch whose realised mean cells per OCCUPIED spot lands in [lo, hi].

    Occupied-spot occupancy, not naive density, is what matters: traced cells sit in clumps, so a
    pitch chosen from area/count alone overshoots.
    """
    span = float(np.ptp(P, axis=0).max())
    best = None
    for pitch in np.geomspace(span / 400.0, span / 4.0, n_try):
        _, _, cnt = bin_cells(P, pitch)
        mu = float(cnt.mean())
        if lo <= mu <= hi:
            s = abs(mu - target)
            if best is None or s < best[0]:
                best = (s, float(pitch), mu, len(cnt))
    assert best is not None, f"no pitch gives mean cells/spot in [{lo}, {hi}]"
    return best[1], best[2], best[3]


# --------------------------------------------------------------------- main
def build(t, section, seed=0, min_cells_per_spot=2, lo=3.0, hi=6.0, target=4.5, out_dir=HERE):
    rng = np.random.default_rng(seed)
    g = t.obst["tree"]
    roots = [n for n in g if g.in_degree(n) == 0]
    assert len(roots) == 1, f"expected one root, got {len(roots)}"
    root = roots[0]
    blen, metric, evidence = tree_branch_lengths(g)

    # ---- 1. the traced cells of THIS section -----------------------------------------
    o = t.obs
    m = (o["sample"].astype(str) == section).to_numpy() & o["clade"].notna().to_numpy()
    names = np.asarray(o.index[m], dtype=object)
    leafset = {n for n in g if g.out_degree(n) == 0}
    assert set(names) <= leafset, "a claded cell is not a leaf of obst['tree']"

    Xall = t.X[m]                                     # sparse INTEGER counts
    tot = np.asarray(Xall.sum(1)).ravel()
    if (tot == 0).any():
        keep = tot > 0
        print(f"[prep] {section}: dropped {int((~keep).sum())} cells with zero panel counts")
        names, Xall = names[keep], Xall[keep]
        m_idx = np.where(m)[0][keep]
    else:
        m_idx = np.where(m)[0]
    clade = o["clade"].to_numpy().astype(str)[m_idx]
    P = np.asarray(t.obsm["spatial"])[m_idx].astype(float)
    P = P - P.min(0)                                  # section-local frame; obsm tiles samples
    clades = np.array(sorted(np.unique(clade), key=lambda c: int(c)))
    N = len(names)

    # ---- 2. DISJOINT 50/50 split, stratified by clade ---------------------------------
    is_sc = np.zeros(N, bool)
    flip = 0
    for c in clades:
        ii = np.where(clade == c)[0]
        ii = rng.permutation(ii)
        k = len(ii) // 2 + (len(ii) % 2 and flip)     # alternate who gets the odd cell
        flip ^= (len(ii) % 2 == 1)
        is_sc[ii[:k]] = True
    sc_i, sp_i = np.where(is_sc)[0], np.where(~is_sc)[0]
    assert not (set(sc_i) & set(sp_i)) and len(sc_i) + len(sp_i) == N

    # ---- 3. pseudo-spots on a square lattice ------------------------------------------
    Psp = P[sp_i]
    pitch, mu_all, n_all = choose_pitch(Psp, lo, hi, target)
    inv, ij, cnt = bin_cells(Psp, pitch)
    ok_spot = cnt >= min_cells_per_spot
    keep_cell = ok_spot[inv]
    dropped_cells, dropped_spots = int((~keep_cell).sum()), int((~ok_spot).sum())
    # renumber the surviving spots
    remap = -np.ones(len(cnt), np.int64)
    remap[np.where(ok_spot)[0]] = np.arange(int(ok_spot.sum()))
    inv2, sp_i2 = remap[inv[keep_cell]], sp_i[keep_cell]
    M = int(ok_spot.sum())
    members = [np.where(inv2 == s)[0] for s in range(M)]
    n_per = np.array([len(v) for v in members])
    assert (n_per >= min_cells_per_spot).all()

    # spot coordinate = its lattice cell CENTRE (a spot is a fixed capture location, not a
    # centroid of whatever happened to land in it)
    origin = Psp.min(0)
    ijk = ij[keep_cell][[v[0] for v in members]]
    xy = origin + (ijk + 0.5) * pitch

    # spot clade = the dominant clade among its member cells; ties -> lowest clade id
    cl_sp = clade[sp_i2]
    ci = {c: i for i, c in enumerate(clades)}
    cnt_kc = np.zeros((M, len(clades)), np.int64)
    np.add.at(cnt_kc, (inv2, np.array([ci[c] for c in cl_sp])), 1)
    spot_clade = clades[cnt_kc.argmax(1)]
    purity = cnt_kc.max(1) / cnt_kc.sum(1)

    # ---- 4. expression: same gene space, log-normalised -------------------------------
    genes = np.asarray(t.var_names, dtype=object)
    Xsc_raw = Xall[sc_i]
    rows = np.zeros((M, Xall.shape[1]), np.float64)
    Xsp_raw = Xall[sp_i2]
    for s, v in enumerate(members):
        rows[s] = np.asarray(Xsp_raw[v].sum(0)).ravel()
    asc, ast = sc.AnnData(Xsc_raw.astype(np.float32)), sc.AnnData(rows.astype(np.float32))
    asc.var_names = ast.var_names = list(genes)
    for a in (asc, ast):
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    dense = lambda X: np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)
    X_sc, X_st = dense(asc.X), dense(ast.X)

    # ---- 5. structural costs ----------------------------------------------------------
    C_tree = patristic(g, root, list(names[sc_i]), blen).astype(np.float32)
    C_tree = 0.5 * (C_tree + C_tree.T)
    np.fill_diagonal(C_tree, 0.0)
    C_space = squareform(pdist(xy)).astype(np.float32)
    C_space /= max(float(C_space.max()), 1e-12)

    cell_clade = clade[sc_i]
    n = len(sc_i)
    A = np.zeros((n, len(clades)), np.float32)
    A[np.arange(n), [ci[c] for c in cell_clade]] = 1.0
    is_labelled = np.ones(M, bool)

    out = dict(
        X_sc=X_sc, X_st=X_st, C_tree=C_tree, C_space=C_space,
        xy=xy.astype(np.float32), xy_sc=P[sc_i].astype(np.float32),
        cell_clade=cell_clade.astype(str), spot_clade=spot_clade.astype(str),
        clades=clades.astype(str), A=A, is_labelled=is_labelled,
        anchor_frac_grid=ANCHOR_FRAC_GRID,
        # provenance / evaluation extras
        cell_names=names[sc_i].astype(str), spot_cell_names=names[sp_i2].astype(str),
        spot_of_cell=inv2.astype(np.int64), spot_n_cells=n_per.astype(np.int64),
        spot_purity=purity.astype(np.float32), genes=genes.astype(str),
        section=np.array(section), seed=np.array(seed), pitch=np.array(pitch, np.float32),
        tree_metric=np.array(metric), tree_metric_evidence=np.array(evidence),
    )

    # ---- 6. guard rails ---------------------------------------------------------------
    for k, v in out.items():
        v = np.asarray(v)
        if np.issubdtype(v.dtype, np.floating):
            assert np.isfinite(v).all(), f"non-finite value in {k}"
    assert (C_tree == C_tree.T).all(), "C_tree not symmetric"
    assert (np.diag(C_tree) == 0).all(), "C_tree diagonal not zero"
    assert (C_tree[~np.eye(n, dtype=bool)] > 0).all(), "zero off-diagonal patristic distance"
    assert (C_space == C_space.T).all() and (np.diag(C_space) == 0).all(), "C_space malformed"
    assert set(cell_clade) <= set(clades), "cell clade outside `clades`"
    assert set(spot_clade) <= set(clades), "spot clade outside `clades`"
    assert A.sum() == n and (A.sum(1) == 1).all(), "A is not one-hot"
    assert (A.argmax(1) == np.array([ci[c] for c in cell_clade])).all(), "A rows mislabelled"
    assert X_sc.shape == (n, len(genes)) and X_st.shape == (M, len(genes)), "gene space mismatch"
    assert not (set(out["cell_names"].tolist()) & set(out["spot_cell_names"].tolist())), \
        "a dissociated cell also sits in a spot -- the split is not disjoint"
    assert lo <= n_per.mean() <= hi, f"realised cells/spot {n_per.mean():.2f} outside [{lo}, {hi}]"

    path = out_dir / f"petracer_{section}.npz"
    np.savez_compressed(path, **out)

    # ---- 7. summary -------------------------------------------------------------------
    cc = pd.Series(cell_clade).value_counts()
    sc_ = pd.Series(spot_clade).value_counts()
    W = 78
    print("\n" + "=" * W)
    print(f"petracer_{section}.npz")
    print("=" * W)
    print(f"traced cells in section      {N}   (clade non-null, leaves of obst['tree'])")
    print(f"  -> dissociated (sc side)   {n}")
    print(f"  -> spot side               {len(sp_i)}  "
          f"({dropped_cells} dropped in spots of < {min_cells_per_spot} cells)")
    print(f"pseudo-spots                 {M}   (lattice pitch {pitch:.1f} um; "
          f"{dropped_spots} under-occupied spots dropped)")
    print(f"cells per spot               mean {n_per.mean():.2f}  median {int(np.median(n_per))}  "
          f"min {n_per.min()}  max {n_per.max()}  (target {lo:.0f}-{hi:.0f})")
    print(f"spot clade purity            mean {purity.mean():.3f}  "
          f"frac pure spots {float((purity == 1).mean()):.3f}")
    print(f"clades                       {len(clades)}   {', '.join(clades)}")
    print(f"genes                        {len(genes)} (full MERFISH panel, no HVG selection)")
    print(f"X_sc {X_sc.shape}  X_st {X_st.shape}  C_tree {C_tree.shape}  C_space {C_space.shape}")
    print(f"tree metric                  {metric}")
    print(f"    evidence: {evidence}")
    print(f"C_tree patristic             min>0 {C_tree[C_tree > 0].min():.4g}  "
          f"max {C_tree.max():.4g}  mean {C_tree[~np.eye(n, dtype=bool)].mean():.4g}  (RAW, unnormalised)")
    print(f"C_space                      normalised by its max "
          f"({float(squareform(pdist(xy)).max()):.1f} um)")
    print(f"is_labelled                  {int(is_labelled.sum())}/{M} = 100% -- every pseudo-spot "
          f"is built from traced cells")
    print(f"anchor_frac_grid             {', '.join(f'{f:.2f}' for f in ANCHOR_FRAC_GRID)}  "
          f"(mask drawn at RUN time, not baked in)")
    print(f"\n{'clade':>7}{'sc cells':>10}{'spots':>8}   {'clade':>7}{'sc cells':>10}{'spots':>8}   "
          f"{'clade':>7}{'sc cells':>10}{'spots':>8}")
    rows_ = [(c, int(cc.get(c, 0)), int(sc_.get(c, 0))) for c in clades]
    third = -(-len(rows_) // 3)
    cols = [rows_[i * third:(i + 1) * third] for i in range(3)]
    for r in range(third):
        line = ""
        for cl in cols:
            line += (f"{cl[r][0]:>7}{cl[r][1]:>10}{cl[r][2]:>8}   " if r < len(cl) else " " * 28)
        print(line.rstrip())
    print(f"{'TOTAL':>7}{n:>10}{M:>8}")
    print("=" * W)
    return out


def verify_patristic(t, seed=0, k=300):
    """Independent check of `patristic` against networkx shortest paths on the weighted graph."""
    g = t.obst["tree"]
    blen, metric, _ = tree_branch_lengths(g)
    o = t.obs
    m = (o["sample"].astype(str) == "M3-1").to_numpy() & o["clade"].notna().to_numpy()
    names = list(np.asarray(o.index[m], dtype=object)[:120])
    root = [n for n in g if g.in_degree(n) == 0][0]
    D = patristic(g, root, names, blen)
    u = nx.Graph()
    u.add_weighted_edges_from([(a, b, blen[(a, b)]) for a, b in g.edges()])
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(k):
        i, j = rng.integers(0, len(names), 2)
        if i == j:
            continue
        err = max(err, abs(D[i, j] - nx.dijkstra_path_length(u, names[i], names[j])))
    print(f"[check] patristic vs networkx dijkstra on {k} random pairs: max abs error {err:.3e} "
          f"(metric: {metric})")
    assert err < 1e-8, "patristic disagrees with networkx"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sections", nargs="*", default=SECTIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-cells-per-spot", type=int, default=2)
    p.add_argument("--lo", type=float, default=3.0)
    p.add_argument("--hi", type=float, default=6.0)
    p.add_argument("--target", type=float, default=4.5)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()

    import pycea
    t = pycea.datasets.koblan25("tumor")
    g = t.obst["tree"]
    _, metric, evidence = tree_branch_lengths(g)
    print(f"[prep] obst['tree']: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
          f"{sum(1 for n in g if g.out_degree(n) == 0)} leaves")
    print(f"[prep] TREE METRIC = {metric}")
    print(f"[prep]   {evidence}")
    if "HOP" in metric:
        print("[prep]   *** WARNING: hop counts are a WEAKER metric than branch lengths and this "
              "must not be assumed silently. ***")
    if not a.no_verify:
        verify_patristic(t, a.seed)
    for s in a.sections:
        build(t, s, seed=a.seed, min_cells_per_spot=a.min_cells_per_spot,
              lo=a.lo, hi=a.hi, target=a.target)


if __name__ == "__main__":
    main()
