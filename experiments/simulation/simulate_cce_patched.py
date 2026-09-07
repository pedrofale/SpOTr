"""Controlled testbed: every quantity that drives the effect is REALIZED to target, per seed.

The confusable testbed produced gains of +0.098, +0.040, -0.021, +0.086 across four seeds
whose *nominal* parameters were identical (12 clades, 6 dispersed, ~108 cells each). The
design parameters were verified on seed 0 and assumed to hold; they did not. Three things
were free to vary:

  the tree      BirthDeathFitnessSimulator draws a new topology each seed -- K came out 8-12,
                min clade size 18-108, and half the seeds died of lineage extinction
  the layout    clade territories came from a 2D Brownian walk, so some draws separated them
                cleanly and others overlapped
  pair sep      how far apart a confusable pair's territories land sets both how much damage
                expression does and how much the tree can repair. Completely uncontrolled,
                and it is the dominant term.

Here the causal order is inverted. Instead of drawing a tree and hoping the induced layout is
well conditioned, we draw a clade-level tree, embed ITS distances in 2D by MDS, and place
territories there. Layout and tree agree by construction, and the embedding quality is
measured rather than assumed. Then:

  * K and clade sizes are exact (no partitioning of whatever topology turned up)
  * territory scale is normalised so mean nearest-neighbour distance = 1, making
    compact_sd and pair separation comparable across seeds
  * confusable pairs are chosen at a TARGET spatial separation
  * expression confusability is CALIBRATED by bisection until measured within-pair
    discrimination hits its target, instead of being set nominally through a Dirichlet draw
  * branch lengths are randomised so the tree is not degenerate-ultrametric (which annihilates
    the GW term); |L|/|F| is reported per seed

Every realized property is printed and range-checked, so a seed that lands out of band is
visible rather than silently contributing variance.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent



def neighbor_joining(D):
    """Fit an ADDITIVE tree to D and return its patristic leaf-leaf distances.

    UPGMA gives an ULTRAMETRIC, in which every pair separated at the root sits at the same
    distance -- that saturation capped tree-vs-layout correlation at ~0.65 no matter how clean
    the layout was. Real phylogenies are not clock-like; branch lengths vary and path distances
    are additive, which fits a planar layout far better. NJ is the standard estimator for that
    and is still a genuine tree metric, not a relabelled Euclidean one.
    """
    n = D.shape[0]
    cur = D.astype(float).copy()
    ids = list(range(n))
    par, blen, nxt = {}, {}, n
    while len(ids) > 2:
        m = len(ids)
        tot = cur.sum(1)
        Q = (m - 2) * cur - tot[:, None] - tot[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = (i, j) if i < j else (j, i)
        di = 0.5 * cur[i, j] + (tot[i] - tot[j]) / (2.0 * (m - 2))
        u = nxt; nxt += 1
        par[ids[i]], blen[ids[i]] = u, max(di, 1e-9)
        par[ids[j]], blen[ids[j]] = u, max(cur[i, j] - di, 1e-9)
        new = 0.5 * (cur[i, :] + cur[j, :] - cur[i, j])
        keep = [k for k in range(m) if k not in (i, j)]
        nd = np.zeros((len(keep) + 1, len(keep) + 1))
        nd[:-1, :-1] = cur[np.ix_(keep, keep)]
        nd[-1, :-1] = nd[:-1, -1] = new[keep]
        cur, ids = nd, [ids[k] for k in keep] + [u]
    par[ids[0]], blen[ids[0]] = ids[1], max(cur[0, 1], 1e-9)

    def chain(x):
        d, out = 0.0, {x: 0.0}
        while x in par:
            d += blen[x]; x = par[x]; out[x] = d
        return out

    ch = [chain(i) for i in range(n)]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            common = set(ch[i]) & set(ch[j])
            P[i, j] = P[j, i] = min(ch[i][a] + ch[j][a] for a in common)
    return P


def poisson_points(K, rng, min_sep=1.0, tries=8000):
    """K points in the plane with a guaranteed minimum separation.

    Drawing the layout FIRST and deriving the tree from it (rather than embedding a random
    tree by MDS) is what makes tree-vs-layout agreement a controlled quantity instead of a
    lottery -- MDS of a random ultrametric onto 2D topped out at r = 0.83.
    """
    side = np.sqrt(K) * min_sep * 1.35
    pts = []
    for _ in range(tries):
        if len(pts) == K:
            break
        p = rng.uniform(0, side, size=2)
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    while len(pts) < K:                       # relax if the disk packing stalls
        min_sep *= 0.9
        p = rng.uniform(0, side, size=2)
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    return np.array(pts)


def tree_from_layout(cen, rng, target_r=0.85, tol=0.02, iters=14):
    """Additive tree fitted to the territory layout, degraded to a TARGET fidelity.

    NJ on a planar layout fits almost perfectly, which would make the tree more informative
    than any real one. Branch-length noise is bisected until the realized tree-vs-layout
    correlation hits target_r, so fidelity becomes a controlled dial rather than whatever the
    estimator happened to achieve.
    """
    d = squareform(pdist(cen))
    base = neighbor_joining(d)
    flat = squareform(d, checks=False)
    lo, hi, best = 0.0, 3.0, None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rr = np.random.default_rng(int(mid * 1e9) % (1 << 31))
        P = base * np.exp(rr.normal(0.0, mid, size=base.shape))
        P = np.triu(P, 1); P = P + P.T                       # keep it symmetric
        got = float(np.corrcoef(squareform(P, checks=False), flat)[0, 1])
        if best is None or abs(got - target_r) < abs(best[1] - target_r):
            best = (P, got)
        if abs(got - target_r) <= tol:
            break
        if got > target_r:
            lo = mid
        else:
            hi = mid
    return best[0]


def random_ultrametric(K, rng, height=1.0, jitter=0.45):
    """Random binary topology on K leaves with randomised merge heights -> KxK tree distance.

    Merge heights are drawn rather than evenly spaced, so the resulting metric is far from the
    balanced-ultrametric case whose GW linearisation is additive and double-centers to zero.
    """
    groups = [[i] for i in range(K)]
    D = np.zeros((K, K))
    h = 0.0
    while len(groups) > 1:
        i, j = rng.choice(len(groups), size=2, replace=False)
        if i > j:
            i, j = j, i
        h += height * np.exp(rng.normal(0.0, jitter))
        for x in groups[i]:
            for y in groups[j]:
                D[x, y] = D[y, x] = h
        groups[i] = groups[i] + groups[j]
        groups.pop(j)
    return D / D.max()


def calibrate_confusability(rng, K, n_per, n_prog, n_genes, pairs, target, prog_strength,
                            expr_noise, tol=0.015, iters=12):
    """Bisect the pair-mixing coefficient until MEASURED within-pair discrimination hits target.

    Setting the mixing nominally (as before) leaves the realized ambiguity at the mercy of the
    Dirichlet draw, which is one of the things that varied between seeds.
    """
    progs = np.zeros((n_prog, n_genes))
    for p, gi in enumerate(np.array_split(rng.permutation(n_genes), n_prog)):
        progs[p, gi] = 1.0
    W0 = rng.dirichlet(np.ones(n_prog) * 0.6, size=K)
    base_seed = int(rng.integers(1 << 30))

    def build(c):
        W = W0.copy()
        for i, j in pairs:
            m = 0.5 * (W[i] + W[j])
            W[i] = c * m + (1 - c) * W[i]
            W[j] = c * m + (1 - c) * W[j]
        r2 = np.random.default_rng(base_seed)
        E = r2.normal(0.0, expr_noise, size=(K * n_per, n_genes))
        for k in range(K):
            rows = slice(k * n_per, (k + 1) * n_per)
            E[rows] += prog_strength * (r2.dirichlet(W[k] * 25.0, size=n_per) @ progs)
        return E, W

    def measure(E):
        lab = np.repeat(np.arange(K), n_per)
        accs = []
        for i, j in pairs:
            sel = np.isin(lab, [i, j])
            accs.append(cross_val_score(
                LogisticRegression(max_iter=500, class_weight="balanced"),
                E[sel], (lab[sel] == i).astype(int), cv=3,
                scoring="balanced_accuracy").mean())
        return float(np.mean(accs))

    lo, hi = 0.0, 1.0
    best = None
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        E, W = build(c)
        m = measure(E)
        if best is None or abs(m - target) < abs(best[2] - target):
            best = (E, W, m, c)
        if abs(m - target) <= tol:
            break
        if m > target:                      # too easy to tell apart -> mix them more
            lo = c
        else:
            hi = c
    return best


def main(K=12, n_per=160, n_prog=6, n_genes=240, seed=0,
         cnv_kappa_w=20.0, cnv_amp_w=0.25, lam_w=20.0, lam_scale=1.0,
         disp_signal=0.0, n_bins=150, cnv_kappa=4.0, cnv_amp=0.55, cnv_noise=0.25, cnv_spot_noise=0.15,
         target_r=0.85, target_confuse=0.58, pair_sep=2.0, compact_sd=0.20, within_tree=0.12,
         prog_strength=2.0, expr_noise=1.0, cells_per_spot=3.0,
         batch_sd=0.3, spot_noise=0.3, out="sim_ctrl.npz"):
    rng = np.random.default_rng(seed)
    n = K * n_per
    lab = np.repeat(np.arange(K), n_per)

    # ---- layout first, then the tree FROM the layout ----
    # resample the layout until the achievable pair separation matches target. With K points
    # at unit nearest-neighbour spacing the attainable distances are effectively discrete, so
    # a single draw overshoots or undershoots; rejecting bad draws is what makes realized
    # separation constant across seeds instead of varying 1.9-2.4.
    def draw():
        c = poisson_points(K, rng, min_sep=1.0)
        c = c / np.sort(squareform(pdist(c)) + np.eye(K) * 1e9, axis=1)[:, 0].mean()
        d = squareform(pdist(c))
        cand = sorted(((abs(d[i, j] - pair_sep), i, j)
                       for i in range(K) for j in range(i + 1, K)), key=lambda t: t[0])
        used, pr = set(), []
        for _, i, j in cand:
            if i not in used and j not in used:
                used |= {i, j}; pr.append((i, j))
        return c, d, pr, float(np.mean([d[i, j] for i, j in pr]))

    best_draw = None
    for _ in range(200):
        c, d, pr, sep = draw()
        if best_draw is None or abs(sep - pair_sep) < abs(best_draw[3] - pair_sep):
            best_draw = (c, d, pr, sep)
        if abs(sep - pair_sep) / pair_sep <= 0.04:
            break
    cen, d_emb, pairs_pre, _ = best_draw
    D_clade = tree_from_layout(cen, rng, target_r=target_r)
    D_clade = D_clade / D_clade.max()
    stress = float(np.corrcoef(squareform(D_clade, checks=False),
                               squareform(d_emb, checks=False))[0, 1])

    # ---- confusable pairs at a TARGET spatial separation ----
    # greedily match each unpaired clade to the partner whose territory distance is closest to
    # pair_sep, so "confusing a pair" costs the same amount in every seed
    pairs = pairs_pre                      # already matched during layout selection
    realized_sep = float(np.mean([d_emb[i, j] for i, j in pairs]))

    # ---- expression calibrated to realized confusability ----
    E, W, got_confuse, coef = calibrate_confusability(
        rng, K, n_per, n_prog, n_genes, pairs, target_confuse, prog_strength, expr_noise)

    # ---- which clades are dispersed: balanced over the pairing ----
    # exactly one member of each pair, so every pair contains one compact and one dispersed
    # clade and the compact/dispersed contrast is never confounded with pair identity
    is_disp = np.zeros(K, dtype=bool)
    for i, j in pairs:
        is_disp[i if rng.random() < 0.5 else j] = True

    if disp_signal > 0:
        drng = np.random.default_rng(60000 + seed)
        progs_l = np.zeros((n_prog, n_genes))
        for pp, gi in enumerate(np.array_split(drng.permutation(n_genes), n_prog)):
            progs_l[pp, gi] = 1.0
        for k in range(K):
            if is_disp[k]:
                rows = slice(k * n_per, (k + 1) * n_per)
                extra = drng.dirichlet(np.ones(n_prog) * 3.0, size=n_per) @ progs_l
                E[rows] += disp_signal * prog_strength * (extra - extra.mean(0))

    # PATCH: the within-clade sub-trees are drawn on a SEPARATE rng, BEFORE the layout, so
    # that the main `rng` stream (layout centres, pairs, calibration, is_disp) is unchanged
    # relative to simulate_controlled.py.  Same distribution as the shipped `sub`.
    wrng = np.random.default_rng(50000 + seed)
    subs = {k: random_ultrametric(int((lab == k).sum()), wrng) * within_tree for k in range(K)}

    lo_b, hi_b = cen.min(0) - 1.0, cen.max(0) + 1.0
    xy = np.zeros((n, 2))
    for k in range(K):
        rows = np.where(lab == k)[0]
        if is_disp[k]:
            xy[rows] = rng.uniform(lo_b, hi_b, size=(len(rows), 2))
        else:
            # PATCH (ii): heritable within-clade layout for COMPACT clades only.  Same number
            # of normals consumed as `rng.normal(0, compact_sd, (len(rows), 2))`, so the rng
            # stream stays aligned; unit-marginal-variance GP keeps the compact spread in band.
            Sw = np.exp(-(lam_w * lam_scale) * subs[k])
            Lw = np.linalg.cholesky(Sw + 1e-6 * np.eye(len(rows)))
            xy[rows] = cen[k] + compact_sd * (Lw @ rng.normal(size=(len(rows), 2)))

    # ---- cell-level tree distances: between-clade from D_clade, within-clade fine structure ----
    C_tree = D_clade[np.ix_(lab, lab)].astype(float)
    for k in range(K):
        rows = np.where(lab == k)[0]
        C_tree[np.ix_(rows, rows)] = subs[k]
    np.fill_diagonal(C_tree, 0.0)

    # ---- disjoint populations ----
    is_sc = np.zeros(n, dtype=bool)
    for k in range(K):
        rows = np.where(lab == k)[0]
        is_sc[rng.choice(rows, size=n_per // 2, replace=False)] = True
    sc_i, sp_i = np.where(is_sc)[0], np.where(~is_sc)[0]

    blo, bhi = xy[sp_i].min(0), xy[sp_i].max(0)
    nb = max(2, int(np.sqrt(len(sp_i) / cells_per_spot)))
    bx = np.clip(((xy[sp_i] - blo) / np.maximum(bhi - blo, 1e-9) * nb).astype(int), 0, nb - 1)
    bid = bx[:, 0] * nb + bx[:, 1]
    ub, inv = np.unique(bid, return_inverse=True)
    m = len(ub)
    Mx = np.zeros((m, len(sp_i))); Mx[inv, np.arange(len(sp_i))] = 1.0
    cnt = Mx.sum(1, keepdims=True)
    X_st = (Mx @ E[sp_i]) / cnt
    xy_st = (Mx @ xy[sp_i]) / cnt
    X_st = X_st * np.exp(rng.normal(0, batch_sd, n_genes))[None, :] + \
        rng.normal(0, spot_noise, size=X_st.shape)
    st_clade = np.array([np.bincount(lab[sp_i][inv == k], minlength=K).argmax() for k in range(m)])
    OH = np.zeros((len(sp_i), K)); OH[np.arange(len(sp_i)), lab[sp_i]] = 1.0
    spot_comp = (Mx @ OH) / cnt                       # [m, K] true clone proportions per spot

    # ---- CNV channel: an anchor modality that tracks the TREE, not expression ----
    # deconvolution_loss_ls is written for CNV profiles ("divide by 2.0 because the CNVs are
    # the copy number of the alleles"). Feeding it expression signatures makes it circular,
    # because the OT feature cost is built from expression too, so the loss is minimised by
    # exactly the coupling the linear term already produces. CNVs break that: they are
    # measured on both sides but are a different modality from the one driving the cost.
    #
    # Copy number is generated as a Gaussian process over clones with covariance
    # exp(-kappa * D_clade), so CNV similarity decays with tree distance -- clones that split
    # recently share segments. A separate RNG keeps every other draw in this simulation
    # bit-identical to the runs made before the channel existed.
    crng = np.random.default_rng(90000 + seed)
    Kc = np.exp(-cnv_kappa * D_clade)
    L = np.linalg.cholesky(Kc + 1e-6 * np.eye(K))
    Z = L @ crng.normal(size=(K, n_bins))
    cnv_clone = np.clip(2.0 + cnv_amp * Z, 0.0, 6.0)
    # PATCH (i): heritable WITHIN-clade anchor.  A GP over the same within-clade sub-tree
    # that is written into C_tree, on a separate rng so cnv_clone and the iid noise below are
    # bit-identical to simulate_controlled.py.
    wcrng = np.random.default_rng(91000 + seed)
    cell_cnv = cnv_clone[lab] + crng.normal(0, cnv_noise, size=(n, n_bins))
    for k in range(K):
        rows = np.where(lab == k)[0]
        Sw = np.exp(-(cnv_kappa_w * lam_scale) * subs[k])
        Lw = np.linalg.cholesky(Sw + 1e-6 * np.eye(len(rows)))
        cell_cnv[rows] += cnv_amp_w * (Lw @ wcrng.normal(size=(len(rows), n_bins)))
    cnv_spot = (Mx @ cell_cnv[sp_i]) / cnt + crng.normal(0, cnv_spot_noise, size=(m, n_bins))

    Ct = C_tree[np.ix_(sc_i, sc_i)]
    C_space = squareform(pdist(xy_st))
    pair_id = np.zeros(K, dtype=int)
    for pi, (i, j) in enumerate(pairs):
        pair_id[i] = pair_id[j] = pi

    # ---- realized properties, range-checked ----
    checks = [
        ("clades", K, None), ("cells/clade", n_per, None),
        ("tree-vs-layout r", stress, (target_r - 0.04, target_r + 0.04)),
        ("pair separation", realized_sep, (pair_sep * 0.93, pair_sep * 1.07)),
        ("within-pair discrim", got_confuse, (target_confuse - 0.04, target_confuse + 0.04)),
        ("compact spread", float(np.mean([xy[lab == k].std(0).mean()
                                          for k in np.where(~is_disp)[0]])), (0.1, 0.45)),
        ("dispersed spread", float(np.mean([xy[lab == k].std(0).mean()
                                            for k in np.where(is_disp)[0]])), (0.8, None)),
        ("spots", m, (150, 800)),
        # the anchor is only useful if it separates the clades expression CANNOT. pair members
        # are far apart in the tree by construction, so their CNV profiles should differ
        ("CNV separates pairs", float(np.mean([
            np.linalg.norm(cnv_clone[i] - cnv_clone[j]) for i, j in pairs])
            / max(np.mean([np.linalg.norm(cnv_clone[i] - cnv_clone[j])
                           for i in range(K) for j in range(i + 1, K)]), 1e-9)), (0.7, None)),
        # a degenerate ultrametric has near-identical row means and its GW linearisation
        # double-centers to zero -- alpha then does nothing regardless of design
        ("tree row-mean sd/mean", float(np.std(Ct.mean(1)) / np.mean(Ct.mean(1))), (0.02, None)),
        # PATCHED-SIM DIAGNOSTICS: how much within-clade tree->space / tree->anchor signal
        # was actually planted (Mantel r, compact clades only for space)
        ("within-clade tree~space r", float(np.mean([
            np.corrcoef(squareform(subs[k], checks=False),
                        pdist(xy[lab == k]))[0, 1]
            for k in np.where(~is_disp)[0]])), None),
        ("within-clade tree~anchor r", float(np.mean([
            np.corrcoef(squareform(subs[k], checks=False),
                        pdist(cell_cnv[lab == k]))[0, 1] for k in range(K)])), None),
    ]
    print(f"[ctrl] seed {seed} realized properties")
    bad = []
    for nm, v, rng_ok in checks:
        flag = ""
        if rng_ok is not None:
            lo_, hi_ = rng_ok
            if (lo_ is not None and v < lo_) or (hi_ is not None and v > hi_):
                flag = "   <-- OUT OF BAND"; bad.append(nm)
        print(f"   {nm:<24} {v:>8.3f}{flag}")
    print(f"[ctrl] {'ALL IN BAND' if not bad else 'OUT OF BAND: ' + ', '.join(bad)}")

    np.savez_compressed(
        HERE / out,
        X_sc=E[sc_i].astype(np.float32), X_st=X_st.astype(np.float32),
        C_tree=(Ct / Ct.max()).astype(np.float32),
        C_space=(C_space / max(C_space.max(), 1e-9)).astype(np.float32),
        xy_sc=xy[sc_i], xy_st=xy_st,
        cell_clade=lab[sc_i], spot_clade=st_clade,
        clades=np.arange(K), is_dispersed=is_disp, pair_id=pair_id,
        cnv_clone=cnv_clone.astype(np.float32), cnv_spot=cnv_spot.astype(np.float32),
        cell_cnv_sc=cell_cnv[sc_i].astype(np.float32),
        spot_comp=spot_comp.astype(np.float32),
        realized=np.array([stress, realized_sep, got_confuse], dtype=float),
        in_band=np.array([len(bad) == 0]))
    print(f"[ctrl] wrote {out}: {len(sc_i)} cells x {m} spots\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in [("K", 12), ("n-per", 160), ("n-prog", 6), ("n-genes", 240), ("seed", 0), ("n-bins", 150)]:
        ap.add_argument(f"--{k}", type=int, default=v, dest=k.replace("-", "_"))
    for k, v in [("disp-signal", 0.0), ("cnv-kappa", 4.0), ("cnv-amp", 0.55), ("cnv-noise", 0.25),
                 ("cnv-spot-noise", 0.15), ("target-r", 0.85), ("target-confuse", 0.58), ("pair-sep", 2.0), ("compact-sd", 0.20),
                 ("within-tree", 0.12), ("prog-strength", 2.0), ("expr-noise", 1.0),
                 ("cells-per-spot", 3.0), ("batch-sd", 0.3), ("spot-noise", 0.3),
                 ("cnv-kappa-w", 20.0), ("cnv-amp-w", 0.25), ("lam-w", 20.0), ("lam-scale", 1.0)]:
        ap.add_argument(f"--{k}", type=float, default=v, dest=k.replace("-", "_"))
    ap.add_argument("--out", default="sim_ctrl.npz")
    main(**vars(ap.parse_args()))
