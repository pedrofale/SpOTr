"""VDA -- cross-view variance decomposition, REPAIRED per the adversarial review.

CANDIDATE (as specified)
  Never score a coupling against a modality. Solve FOUR couplings from disjoint halves of the
  same evidence and score map-vs-map displacement:
      gA = solve(C_f^(1), D_full)   gB = solve(C_f^(2), D_full)     [gene split, tree shared]
      gC = solve(C_f^full, D^(1))   gD = solve(C_f^full, D^(2))     [tree split, genes shared]
  mu^V = rownorm(gamma^V) @ x_spot.   V_gene = E||mu^A-mu^B||^2,  V_tree = E||mu^C-mu^D||^2.
  S_raw = (V_gene + V_tree) / (2 tr Cov(x_spots));  alpha_hat = argmin S.
  V_gene(1)=0 and V_tree(0)=0 exactly (the switched-off channel makes the pair bit-identical),
  so S has an interior optimum by construction.

WHAT THE REVIEW SAID AND WHAT IS IMPLEMENTED HERE
  F1 (fatal): the interior minimum is manufactured. build_perclade_alpha_cost_v2 has no
      across-alpha equalizer, so median|C0| ~ sqrt((1-a)^2+a^2) dips ~29% at mid-alpha; with
      eps=0.02 that diffuses the plan, mu shrinks to the spot centroid and ||mu^A-mu^B||^2
      collapses MECHANICALLY.
      FIX (1) implemented, both halves of it:
        (a) the q29_scalefixed one-liner -- every cost is renormalised to the median |C| it
            had at alpha=0 IN ITS OWN VIEW, so eps means the same thing at every alpha;
        (b) S is replaced by a scale-free split-half reliability
                S_gene = E||mu^A-mu^B||^2 / (Var_i(mu^A) + Var_i(mu^B))
            (1 - ICC), invariant to isotropic shrinkage of the mu cloud. S = S_gene + S_tree.
        Both S_raw (the spec) and S_icc (the repair) are reported, from the same solves.
        TEST A diagnostics (median|C0|, mean row entropy, spread Sp of the mu cloud) are
        recorded at every alpha so the artefact is visible either way.
  F3 (partial fix): the edge split has no estimand and no character matrix exists on this
      simulator, so the review's preferred target-site bootstrap cannot be run. The review's
      fallback demand IS run: sweep the split fraction (50/50 vs 75/25 genes) and check
      whether argmin S moves -- if it does, the criterion reports the split design.
  OBSTACLE 2 (mandatory): the whole criterion is re-run with a LEAF-SHUFFLED tree
      (C_tree -> P C_tree P^T, and the same P applied to both half-tree metrics, which is
      exactly the real tree with the leaves relabelled).

TREE SPLIT ON THIS SIMULATOR
  simulate_controlled.py builds C_tree blockwise: cross-clade entries are an NJ tree on the K
  clade centroids, within-clade blocks are random ultrametrics * 0.12. There is no tree object
  stored, so the edge set is reconstructed exactly from that construction: NJ on the KxK clade
  matrix gives the deep edges, UPGMA on each (ultrametric, hence exactly recoverable)
  within-clade block gives the shallow ones. A patristic metric over an edge SUBSET is then
  D_ij = s_i + s_j - 2 sum_e len_e M_ie M_je with M the leaf-below-edge indicator, which is
  the "other edges contracted" metric the candidate asks for. Edges are split 50/50 stratified
  by subtree size (the depth proxy), each half rescaled to median 1, each view given its own
  sF_ref/sL_ref from make_cladexfgw_scales as the spec requires.

Solver: the canonical q10/q29 harness (balanced sinkhorn_unrolled_safe, eps=0.02, T=50, J=3,
build_perclade_alpha_cost_v2(equalize='measured')) rather than the spec's semi-relaxed
tau=0.01, so the accuracy numbers are directly comparable with the twelve criteria already on
record. Row-normalising gamma makes mu well defined under either solver.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp
from scipy.cluster.hierarchy import linkage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]


# ---------------------------------------------------------------- tree edge decomposition
def _nj_tree(D):
    """Neighbour joining -> (parent dict, branch-length dict, root id). Leaves are 0..K-1."""
    n = D.shape[0]
    cur = D.astype(float).copy(); ids = list(range(n))
    par, blen, nxt = {}, {}, n
    while len(ids) > 2:
        m = len(ids); tot = cur.sum(1)
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
    return par, blen, ids[1]


def tree_edges(C_tree, lab, K):
    """Return (memberships [n_edges x n_cells] bool, lengths [n_edges], subtree size).

    Deep edges: NJ on the KxK cross-clade matrix.  Shallow edges: UPGMA on each within-clade
    block (they are ultrametric by construction, so UPGMA is exact).
    """
    n = len(lab)
    idx = [np.where(lab == k)[0] for k in range(K)]
    Dk = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            if k != l:
                Dk[k, l] = C_tree[idx[k][0], idx[l][0]]
    par, blen, _ = _nj_tree(Dk)
    below = {k: set([k]) for k in range(K)}          # clade-ids below each node
    for node in sorted(par, key=lambda v: v):
        p = par[node]
        below.setdefault(p, set()).update(below.get(node, {node} if node < K else set()))
    # propagate upward properly (nodes created in increasing order, children always < parent)
    below = {}
    for node in range(K):
        below[node] = {node}
    for node in sorted(par):
        p = par[node]
        below.setdefault(p, set()).update(below.get(node, set()))
    mem, ln, sz = [], [], []
    for node, p in par.items():
        cells = np.zeros(n, bool)
        for k in below.get(node, {node} if node < K else set()):
            cells[idx[k]] = True
        if cells.sum() == 0 or cells.sum() == n:
            continue
        mem.append(cells); ln.append(blen[node]); sz.append(int(cells.sum()))
    # within-clade edges
    for k in range(K):
        ii = idx[k]; nk = len(ii)
        if nk < 3:
            continue
        sub = C_tree[np.ix_(ii, ii)]
        cond = sub[np.triu_indices(nk, 1)]
        Z = linkage(cond, method="average")
        h = np.zeros(nk + nk - 1)                     # node heights
        members = [np.array([t]) for t in range(nk)]
        for t in range(nk - 1):
            c1, c2 = int(Z[t, 0]), int(Z[t, 1])
            h[nk + t] = Z[t, 2] / 2.0
            members.append(np.concatenate([members[c1], members[c2]]))
        for t in range(nk - 1):
            for c in (int(Z[t, 0]), int(Z[t, 1])):
                L = h[nk + t] - h[c]
                if L <= 0:
                    continue
                cells = np.zeros(n, bool); cells[ii[members[c]]] = True
                mem.append(cells); ln.append(float(L)); sz.append(int(cells.sum()))
    return np.asarray(mem), np.asarray(ln, float), np.asarray(sz)


def patristic_from_edges(mem, ln, keep):
    M = mem[keep].astype(np.float32)                  # [E' x n]
    w = ln[keep].astype(np.float32)
    s = (w[:, None] * M).sum(0)
    G = (M * w[:, None]).T @ M                        # [n x n]
    D = s[:, None] + s[None, :] - 2.0 * G
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def split_edges(sz, rng, frac=0.5):
    """50/50 edge split stratified by subtree size (the depth proxy)."""
    order = np.argsort(sz)
    keep1 = np.zeros(len(sz), bool)
    for s0 in range(0, len(order), 10):               # strata of 10 in the size ordering
        blk = order[s0:s0 + 10]
        pick = rng.choice(blk, size=max(1, int(round(frac * len(blk)))), replace=False)
        keep1[pick] = True
    return keep1, ~keep1


def split_genes(X, rng, frac=0.5):
    """Gene split stratified by mean-expression decile."""
    m = X.mean(0)
    order = np.argsort(m)
    g1 = np.zeros(X.shape[1], bool)
    for s0 in range(0, len(order), 10):
        blk = order[s0:s0 + 10]
        pick = rng.choice(blk, size=max(1, int(round(frac * len(blk)))), replace=False)
        g1[pick] = True
    return g1, ~g1


# ---------------------------------------------------------------- one view
class View:
    def __init__(self, X_sc, X_st, C_tree, C_space, omega, K):
        self.Cf, self.CT, self.CS, self.a, self.b = S.prepare_ot_inputs(
            ad.AnnData(np.asarray(X_sc, np.float32)), ad.AnnData(np.asarray(X_st, np.float32)),
            np.asarray(C_tree, np.float32), np.asarray(C_space, np.float32))
        self.g0 = self.a[:, None] * self.b[None, :]
        self.sF, self.sL = S.make_cladexfgw_scales(self.Cf, self.CT, self.CS,
                                                   self.a, self.b, self.g0, eps=EPS)
        self.omega, self.K = omega, K
        self.REF = float(jnp.median(jnp.abs(self._C0(0.0, self.g0))))

    def _C0(self, av, g):
        return build_perclade_alpha_cost_v2(jnp.full((self.K,), av, dtype=jnp.float32),
                                            self.Cf, self.CT, self.CS, self.a, self.b, g,
                                            self.omega, self.sF, self.sL, equalize="measured")

    def solve(self, av, fix_scale=True):
        g = self.g0; medC0 = None
        for _ in range(J):
            C = self._C0(av, g)
            if medC0 is None:
                medC0 = float(jnp.median(jnp.abs(C)))
            if fix_scale:
                C = C * (self.REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))
            g, _ = S.sinkhorn_unrolled_safe(C, self.a, self.b, EPS, T)
        return np.asarray(g), medC0


def mu_of(g, x):
    p = g / np.maximum(g.sum(1, keepdims=True), 1e-30)
    return p @ x, p


def disagreement(gU, gV, x):
    muU, _ = mu_of(gU, x); muV, _ = mu_of(gV, x)
    d2 = float(((muU - muV) ** 2).sum(1).mean())
    var = float(((muU - muU.mean(0)) ** 2).sum(1).mean() + ((muV - muV.mean(0)) ** 2).sum(1).mean())
    return d2, var


# ---------------------------------------------------------------- main
def run(seeds, R, frac_gene, tag, shuffle_tree, fix_scale, rows):
    for seed in seeds:
        f = HERE / f"sim_ctrl_s{seed}.npz"
        d = np.load(f, allow_pickle=True)
        cidx, spot_ci = d["cell_clade"], d["spot_clade"]
        K = len(d["clades"]); n = len(cidx)
        A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
        omega = jnp.asarray(A)
        X_sc, X_st = np.asarray(d["X_sc"]), np.asarray(d["X_st"])
        C_tree = np.asarray(d["C_tree"], np.float64)
        C_space = np.asarray(d["C_space"], np.float64)
        x = np.asarray(d["xy_st"], np.float64)
        trC = float(((x - x.mean(0)) ** 2).sum(1).mean())

        mem, ln, sz = tree_edges(C_tree, cidx, K)
        D_all = patristic_from_edges(mem, ln, np.ones(len(ln), bool))
        fid = float(np.corrcoef(C_tree[np.triu_indices(n, 1)], D_all[np.triu_indices(n, 1)])[0, 1])

        perm = np.arange(n)
        if shuffle_tree:
            perm = np.random.default_rng(1000 + seed).permutation(n)
        CT_use = C_tree[np.ix_(perm, perm)]

        full = View(X_sc, X_st, CT_use, C_space, omega, K)
        base = {}
        for av in GRID:
            g, medC0 = full.solve(av, fix_scale)
            mu, p = mu_of(g, x)
            ent = float((-(p * np.log(np.maximum(p, 1e-30))).sum(1)).mean())
            Sp = float(((mu - mu.mean(0)) ** 2).sum(1).mean())
            base[av] = dict(acc=float((spot_ci[g.argmax(1)] == cidx).mean()),
                            medC0=medC0, ent=ent, Sp=Sp)

        acc_r = {av: [] for av in GRID}
        agg = {av: dict(dg=0.0, vg=0.0, dt=0.0, vt=0.0) for av in GRID}
        for r in range(R):
            rng = np.random.default_rng(7 * seed + 101 * r + 13)
            g1, g2 = split_genes(X_sc, rng, frac_gene)
            e1, e2 = split_edges(sz, rng)
            D1 = patristic_from_edges(mem, ln, e1); D2 = patristic_from_edges(mem, ln, e2)
            md = lambda D: np.median(D[np.triu_indices(n, 1)]) + 1e-12
            D1 = (D1 / md(D1))[np.ix_(perm, perm)]
            D2 = (D2 / md(D2))[np.ix_(perm, perm)]
            vA = View(X_sc[:, g1], X_st[:, g1], CT_use, C_space, omega, K)
            vB = View(X_sc[:, g2], X_st[:, g2], CT_use, C_space, omega, K)
            vC = View(X_sc, X_st, D1, C_space, omega, K)
            vD = View(X_sc, X_st, D2, C_space, omega, K)
            for av in GRID:
                gA, _ = vA.solve(av, fix_scale); gB, _ = vB.solve(av, fix_scale)
                gC, _ = vC.solve(av, fix_scale); gD, _ = vD.solve(av, fix_scale)
                dg, vg = disagreement(gA, gB, x)
                dt, vt = disagreement(gC, gD, x)
                agg[av]["dg"] += dg / R; agg[av]["vg"] += vg / R
                agg[av]["dt"] += dt / R; agg[av]["vt"] += vt / R

        for av in GRID:
            q = agg[av]
            rows.append(dict(tag=tag, seed=seed, alpha=av, R=R, frac_gene=frac_gene,
                             shuffled=shuffle_tree, fix_scale=fix_scale, edge_fid=fid,
                             V_gene=q["dg"], V_tree=q["dt"],
                             S_raw=(q["dg"] + q["dt"]) / (2 * trC),
                             S_icc=q["dg"] / max(q["vg"], 1e-30) + q["dt"] / max(q["vt"], 1e-30),
                             Sg_icc=q["dg"] / max(q["vg"], 1e-30),
                             St_icc=q["dt"] / max(q["vt"], 1e-30),
                             **base[av]))
        print(f"  [{tag}] seed {seed} done  edge-fidelity r={fid:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=99)
    ap.add_argument("--stages", default="main,shuffled")
    a = ap.parse_args()
    seeds = [s for s in range(14)
             if (HERE / f"sim_ctrl_s{s}.npz").exists()
             and bool(np.load(HERE / f"sim_ctrl_s{s}.npz", allow_pickle=True)["in_band"][0])]
    seeds = seeds[:a.seeds]
    print("in-band seeds:", seeds, flush=True)
    rows = []
    st = a.stages.split(",")
    t0 = time.time()
    if "main" in st:
        run(seeds, a.R, 0.5, "main", False, True, rows)
        print(f"  main {time.time()-t0:.0f}s", flush=True)
    if "shuffled" in st:
        run(seeds, a.R, 0.5, "shuffled", True, True, rows)
        print(f"  +shuffled {time.time()-t0:.0f}s", flush=True)
    if "frac" in st:
        run(seeds, a.R, 0.75, "frac75", False, True, rows)
        print(f"  +frac75 {time.time()-t0:.0f}s", flush=True)
    if "noscalefix" in st:
        run(seeds, a.R, 0.5, "noscalefix", False, False, rows)
        print(f"  +noscalefix {time.time()-t0:.0f}s", flush=True)
    if "wide" in st:
        global GRID
        GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 0.95]
        run(seeds, 1, 0.5, "wide", False, True, rows)
        print(f"  +wide {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "cand_vda-two-channel-variance-decomposition.csv",
                              index=False)
    print("wrote csv", flush=True)


if __name__ == "__main__":
    main()
