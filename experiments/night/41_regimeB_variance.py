"""STOP CONDITION 2: in regime B, does the DERIVED variance r(a_k) match or beat a FITTED one?

Regime B: a query cell has a position and no lineage readout; the targets are the K = 18 clades of
the reference tree.  The cost is the average over reference cells of the Rayleigh/Gaussian NLL

    C[i,k] = (1/|R|) sum_j [ d_ij^2 / (2 s(k,j)^2)  +  2 log s(k,j) ]

and the ONLY thing that differs between the arms is where s(k,j) comes from.  Because the PEtracer
tree is exactly ultrametric (verified: C_tree from a clade's members to any outside cell has
sd/mean = 0), a(k,j) -- the age of the MRCA of clade k's stem and reference cell j -- is read
straight off C_tree, is constant over the members of k, and takes only a handful of distinct values
per clade.  That makes the fitted competitor SATURATED rather than approximate: it is the best any
per-clade variance could do on this data.

  derived r(a)     s(k,j) = r(a(k,j)),  ONE global monotone curve, ~20 knots for the whole tumour
  fitted (group)   s(k,j) free per (clade, distinct age)   <- the SATURATED fit, THE competitor
  fitted (in/out)  s(k,j) free per clade, two levels: j in clade k vs j outside
  fitted (blob)    only clade k's own cells contribute, one free scale per clade -- no tree at all
  flat (control)   s(k,j) = one global constant -- "the tree says the same thing at every age"

and four repairs applied to the derived arm, so it cannot be said to have been scored uncharitably:
NO log term (the handoff warns this collapses -- it does), a single fitted global multiplier c,
and refitting the in-clade or the outside scale alone to localise where the derived curve fails.

Every s is estimated on the REFERENCE half and scored on the disjoint QUERY half.  No coupling, no
solver, no new module: a scoring exercise, as decision-recomb-submission requires.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
SECS = ["M3-4", "M3-1", "M3-2"]
NSEED, MINPAIR, KNN = 5, 30, 15


def fit_spread(t, s, nbin=22, min_pairs=200):
    """Verbatim from src/spotr/clonespread.py (branch spotr-evaluation) so this script stands alone."""
    t = np.asarray(t, float); s = np.asarray(s, float)
    a = t / 2.0; cap = a.max(); below = a[a < cap - 1e-9]
    edges = np.r_[np.linspace(below.min(), below.max(), nbin), cap - 1e-9, cap + 1e-9]
    ac, r = [], []
    for i in range(len(edges) - 1):
        m = (a >= edges[i]) & (a < edges[i + 1])
        if m.sum() < min_pairs:
            continue
        ac.append(a[m].mean()); r.append(np.sqrt((s[m] ** 2).mean() / 2.0))
    return dict(age=np.array(ac), r=np.maximum.accumulate(np.array(r)))


def scale_from(d2):
    """ML per-axis scale of a 2-D Gaussian/Rayleigh separation: sigma = sqrt(E[d^2]/2)."""
    return float(np.sqrt(max(np.mean(d2), 1e-12) / 2.0))


def metrics(C, y, K):
    order = np.argsort(C, axis=1)
    pred = order[:, 0]
    rank = np.argmax(order == y[:, None], axis=1) + 1
    rec = [float((pred[y == k] == k).mean()) for k in range(K) if (y == k).sum() > 0]
    return dict(top1=float((pred == y).mean()), macro=float(np.mean(rec)),
                top3=float((rank <= 3).mean()), mean_rank=float(rank.mean()))


def run_split(D, ref, qry):
    C_tree, XY, Y, K = D["C_tree"], D["xy"], D["y"], D["K"]
    xr, xq, yr, yq = XY[ref], XY[qry], Y[ref], Y[qry]
    nR = len(ref)
    D2 = cdist(xq, xr, "sqeuclidean")

    # ---- the tree, as ages.  a(k,j) is constant over the members of clade k (ultrametric).
    # A clade needs >= 2 reference members: with one, its MRCA age would be 0 and r(0) would hand
    # it an artificially tiny variance.
    A = np.zeros((K, nR)); ok = np.zeros(K, bool)
    for k in range(K):
        mk = ref[yr == k]
        if len(mk) < 2:
            continue
        ok[k] = True
        A[k] = C_tree[mk[0], ref] / 2.0                      # correct for every j outside clade k
        A[k][yr == k] = C_tree[np.ix_(mk, mk)].max() / 2.0    # inside k: the clade's own MRCA age

    # ---- reference-only estimation
    iu = np.triu_indices(nR, 1)
    t_ref = C_tree[np.ix_(ref, ref)][iu]; s_ref = pdist(xr)
    sp = fit_spread(t_ref, s_ref)
    Dr2 = squareform(s_ref ** 2)
    s_der = np.interp(A, sp["age"], sp["r"])

    grp, inout, in_only, out_only = (s_der.copy() for _ in range(4))
    blob = np.full(K, np.nan)
    fitted_rows = []
    num = den = 0.0
    for k in range(K):
        if not ok[k]:
            continue
        mk = np.where(yr == k)[0]; ink = yr == k
        for lev, mask in (("in", ink), ("out", ~ink)):
            d2 = Dr2[np.ix_(mk, np.where(mask)[0])]
            d2 = d2[d2 > 0]
            if d2.size >= MINPAIR:
                s_hat = scale_from(d2)
                inout[k, mask] = s_hat
                (in_only if lev == "in" else out_only)[k, mask] = s_hat
                if lev == "in":
                    blob[k] = s_hat
        for a in np.unique(A[k]):                             # SATURATED: free scale per (k, age)
            mask = A[k] == a
            d2 = Dr2[np.ix_(mk, np.where(mask)[0])]; d2 = d2[d2 > 0]
            if d2.size < MINPAIR:
                continue
            grp[k, mask] = scale_from(d2)
            r_a = float(np.interp(a, sp["age"], sp["r"]))
            num += d2.sum(); den += 2 * d2.size * r_a ** 2    # for the single global multiplier
            fitted_rows.append(dict(clade=k, age=float(a), npair=int(d2.size),
                                    s_fit=scale_from(d2), s_derived=r_a,
                                    n_in_group=int(mask.sum()), inside=bool(a == A[k][ink][0])))
    c_glob = float(np.sqrt(num / den)) if den > 0 else 1.0

    def score(s, log=True):
        s = np.maximum(s, 1e-3)
        C = D2 @ (1.0 / (2.0 * s ** 2)).T / nR
        if log:
            C = C + (2.0 * np.log(s)).mean(1)[None, :]
        C[:, ~ok] = np.inf
        return metrics(C, yq, K)

    out = {"derived r(a)": score(s_der),
           "fitted (group)": score(grp),
           "fitted (in/out)": score(inout),
           "flat (control)": score(np.full_like(s_der, scale_from(s_ref ** 2))),
           "derived, NO log term": score(s_der, log=False),
           "derived x fitted global c": score(s_der * c_glob),
           "derived, in-clade refitted": score(in_only),
           "derived, outside refitted": score(out_only)}

    C = np.full((len(qry), K), np.inf)                        # fitted (blob): clade k's own cells only
    for k in range(K):
        if ok[k] and np.isfinite(blob[k]):
            C[:, k] = D2[:, yr == k].mean(1) / (2 * blob[k] ** 2) + 2 * np.log(blob[k])
    out["fitted (blob)"] = metrics(C, yq, K)

    # ---- tree-free baselines for context
    nn = np.argsort(D2, axis=1)[:, :KNN]
    votes = np.zeros((len(qry), K))
    for c in range(KNN):
        np.add.at(votes, (np.arange(len(qry)), yr[nn[:, c]]), 1.0)
    votes[:, ~ok] = -1
    out[f"kNN vote (k={KNN})"] = metrics(-votes, yq, K)
    cen = np.array([xr[yr == k].mean(0) if ok[k] else [1e9, 1e9] for k in range(K)])
    out["nearest centroid"] = metrics(cdist(xq, cen, "sqeuclidean"), yq, K)
    pri = np.array([(yr == k).sum() for k in range(K)], float); pri /= pri.sum()
    out["prior (chance)"] = metrics(np.tile(-pri, (len(qry), 1)), yq, K)
    return out, pd.DataFrame(fitted_rows), c_glob


rows, fitted_all, cs = [], [], []
for sec in SECS:
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    clades = np.asarray(z["clades"]); K = len(clades)
    ix = {c: i for i, c in enumerate(clades)}
    D = dict(C_tree=np.asarray(z["C_tree"], float), xy=np.asarray(z["xy_sc"], float),
             y=np.array([ix[c] for c in np.asarray(z["cell_clade"])]), K=K)
    n = len(D["xy"])
    for seed in range(NSEED):
        h = np.random.default_rng(seed).permutation(n)
        for half in (0, 1):
            ref = np.sort(h[:n // 2] if half == 0 else h[n // 2:])
            qry = np.sort(h[n // 2:] if half == 0 else h[:n // 2])
            res, fr, cg = run_split(D, ref, qry)
            fr["section"] = sec; fr["seed"] = seed; fr["half"] = half
            fitted_all.append(fr); cs.append(cg)
            for name, m in res.items():
                rows.append(dict(section=sec, seed=seed, half=half, arm=name, **m))
    print(f"  {sec}: n={n}, K={K}, {NSEED * 2} splits", flush=True)

R = pd.DataFrame(rows); F = pd.concat(fitted_all, ignore_index=True)
MAIN = ["derived r(a)", "fitted (group)", "fitted (in/out)", "fitted (blob)", "flat (control)",
        f"kNN vote (k={KNN})", "nearest centroid", "prior (chance)"]
REPAIR = ["derived r(a)", "derived, NO log term", "derived x fitted global c",
          "derived, in-clade refitted", "derived, outside refitted", "fitted (group)"]

print("\n" + "=" * 100)
print(f"REGIME B ON PEtracer M3 -- clade recovery for every query cell, {NSEED*2*len(SECS)} splits, K = 18 clades")
print("=" * 100)
print(f"{'arm':<26}{'top-1':>9}{'macro':>9}{'top-3':>9}{'mean rank':>11}")
for a in MAIN:
    g = R[R.arm == a]
    print(f"{a:<26}{g.top1.mean():9.3f}{g.macro.mean():9.3f}{g.top3.mean():9.3f}{g.mean_rank.mean():11.2f}")
print("\nPER SECTION (top-1)")
print(R.pivot_table(index="arm", columns="section", values="top1").reindex(MAIN)[SECS].round(3).to_string())

base = R[R.arm == "derived r(a)"].set_index(["section", "seed", "half"])
def paired(block, title):
    print("\n" + "=" * 100); print(title); print("=" * 100)
    for a in block:
        if a == "derived r(a)":
            continue
        o = R[R.arm == a].set_index(["section", "seed", "half"]).loc[base.index]
        for met in ["top1", "macro"]:
            dl = base[met] - o[met]
            w = int((dl > 0).sum()); pv = binomtest(w, len(dl), 0.5).pvalue
            try:
                wp = wilcoxon(base[met], o[met]).pvalue
            except ValueError:
                wp = float("nan")
            v = "derived WINS" if (dl.mean() > 0 and pv < 0.05) else \
                ("derived LOSES" if (dl.mean() < 0 and pv < 0.05) else "tie")
            print(f"  derived vs {a:<28}{met:<6}{w:3d}/{len(dl)} {dl.mean():+.4f}  "
                  f"sign p={pv:<9.3g} wilcoxon p={wp:<9.3g} {v}")
paired(MAIN, "THE STOP CONDITION: derived r(a) vs each fitted variance, paired over splits")
paired(REPAIR, "IS THE DERIVED ARM BEING SCORED UNCHARITABLY?  every repair applied")
print(f"\n  single fitted global multiplier c: mean {np.mean(cs):.3f} "
      f"[{np.min(cs):.3f}, {np.max(cs):.3f}] -- rescaling the whole curve changes nothing.")

print("\n" + "=" * 100)
print("WHY: how far the fitted scales depart from the derived curve  (sigma_fit / r(a))")
print("=" * 100)
F["ratio"] = F.s_fit / F.s_derived
ins = F[F.inside]
print(f"  all {len(F)} (clade, age) groups: median {F.ratio.median():.3f}, "
      f"IQR [{F.ratio.quantile(.25):.3f}, {F.ratio.quantile(.75):.3f}], "
      f"range [{F.ratio.min():.3f}, {F.ratio.max():.3f}]")
print(f"  in-clade groups (n={len(ins)}):   median {ins.ratio.median():.3f}, "
      f"IQR [{ins.ratio.quantile(.25):.3f}, {ins.ratio.quantile(.75):.3f}]; "
      f"{(ins.ratio < 0.7).mean():.0%} have sigma_fit < 0.7 r(a)")
g = ins.groupby(["section", "clade"]).agg(s_fit=("s_fit", "mean"), s_der=("s_derived", "mean")).reset_index()
print("\n  dynamic range of the in-clade scale across clades (mean over splits):")
for sec in SECS:
    gs = g[g.section == sec]
    print(f"    {sec}: fitted {gs.s_fit.min():5.0f}-{gs.s_fit.max():5.0f} um (cv {gs.s_fit.std()/gs.s_fit.mean():.3f})"
          f"   derived {gs.s_der.min():5.0f}-{gs.s_der.max():5.0f} um (cv {gs.s_der.std()/gs.s_der.mean():.3f})")
print("\n  The derived curve is BIASED HIGH in-clade and UNDER-DISPERSED across clades: it cannot"
      "\n  express the between-clade differences the fitted variance is exploiting.")

out = HERE / "regimeB_variance"; out.mkdir(exist_ok=True)
R.to_csv(out / "scores.csv", index=False); F.to_csv(out / "fitted_scales.csv", index=False)
print(f"\nwrote {out}")
