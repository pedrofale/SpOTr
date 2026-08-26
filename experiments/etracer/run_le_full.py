"""SpOTr on eTRACER late-escape at full scale, with the barcode as a masked loss anchor.

Setup. 35,835 LE bins, of which 21.1% carry a clone barcode; 46,171 dissociated cells with
clone labels and per-clone ML trees. The barcoded bins are split into TRAIN and TEST. Only
TRAIN labels are ever shown to the model -- as a loss (to learn per-clade alpha) or as a
feature (the competing route). TEST labels are the answer key and are never used for fitting.

Five configurations:
  alpha = 0            expression only, no tree
  best global alpha    swept and selected on TRAIN bins only
  learned per-clade    alpha_k fitted against the barcode loss masked to TRAIN bins
  shuffled per-clade   the same alpha values permuted across clones (control)
  feature route        TRAIN clone labels appended to the spot features instead

The scientific centrepiece is not the mapping score but whether the learned alpha_k tracks the
INDEPENDENTLY measured per-clone tree-space concordance from etracer_premise.csv, which was
computed from the spatial trees and never enters this fit. The previous attempt, using an
expression loss, gave Spearman -0.103 (p = 0.64) -- no signal, as the circularity predicts.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")
_ORIG = S.deconvolution_loss_ls

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
rng = np.random.default_rng(0)

d = np.load(HERE / "etracer_LE_full.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clone = np.asarray(d["cell_clone"]).astype(str)
bin_clone = np.asarray(d["bin_clone"]).astype(str)
clones = np.asarray(d["clones"]).astype(str)
A = np.asarray(d["A"], np.float32)
is_lab = np.asarray(d["is_labelled"], bool)
K, m = len(clones), X_st.shape[0]
cidx = np.array([list(clones).index(c) for c in cell_clone])
print(f"[run] {X_sc.shape[0]} cells x {m} bins, {K} clones, "
      f"{is_lab.sum()} labelled bins ({100*is_lab.mean():.1f}%)")

# ---- train / test split of the labelled bins, stratified by clone ----
lab_idx = np.where(is_lab)[0]
train = np.zeros(m, bool)
for c in clones:
    ii = lab_idx[bin_clone[lab_idx] == c]
    if len(ii) == 0:
        continue
    train[rng.choice(ii, size=max(1, int(round(0.6 * len(ii)))), replace=False)] = True
test = is_lab & ~train
print(f"[run] train {train.sum()} labelled bins, test {test.sum()}")

bidx = np.array([list(clones).index(c) if c in list(clones) else -1 for c in bin_clone])
Y_oh = np.zeros((m, K), np.float32)
Y_oh[np.arange(m)[bidx >= 0], bidx[bidx >= 0]] = 1.0

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
L = np.asarray(S.compute_Lgw(C_T, C_S, a, b, g0))
print(f"[run] |L|/|F| = {np.abs(L).mean()/np.abs(np.asarray(Cf)).mean():.3e}")

mask = jnp.asarray(train.astype(np.float32))[:, None]
Yj = jnp.asarray(Y_oh)
sig_I = jnp.asarray(2.0 * np.eye(K, dtype=np.float32))


def barcode_loss(Y, gam, ca, cc, _m=mask, _Y=Yj):
    p = jnp.einsum('ji,ik->jk', gam.T, ca)
    p = p / jnp.maximum(jnp.sum(p, axis=1, keepdims=True), 1e-12)
    return jnp.sum(((_Y - p) ** 2) * _m) / jnp.maximum(jnp.sum(_m) * p.shape[1], 1.0)


def solve(ak):
    g = g0
    for _ in range(J):
        C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf, C_T, C_S,
                                         a, b, g, omega, sF, sL, equalize="measured")
        g, _u = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
    return np.asarray(g)


def score(g, sel):
    """Score EVERY cell by restricting its coupling row to the held-out bins.

    Scoring only the cells whose global argmax happens to fall on a test bin is not a fair
    comparison: the scored subset differs between configurations (318 / 315 / 289 cells in the
    first run) and is itself chosen by the mapping being scored. Restricting the row first
    scores all cells under every configuration on the same footing.
    """
    if not np.isfinite(g).all():
        return dict(acc=np.nan, ari=np.nan, n=0)
    cols = np.where(sel)[0]
    sub = g[:, cols]
    j = cols[sub.argmax(1)]
    pred = bin_clone[j]
    return dict(acc=float((pred == cell_clone).mean()),
                ari=float(adjusted_rand_score(cell_clone, pred)),
                n=int(len(pred)))


def train_loss(g):
    return float(barcode_loss(None, jnp.asarray(g), jnp.asarray(A), None))


results = {}
sw = {v: solve(np.full(K, v)) for v in GRID}
tl = {v: train_loss(g) for v, g in sw.items()}
bv = min(tl, key=lambda k: tl[k])              # selected on TRAIN only
print(f"\n[run] global alpha selected on train barcode loss: {bv}")
print("      " + "  ".join(f"{v}:{tl[v]:.5f}" for v in GRID))
results["alpha=0"] = score(sw[0.0], test)
results[f"best global (a={bv})"] = score(sw[bv], test)

S.deconvolution_loss_ls = barcode_loss
try:
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    res = S.learn_alpha_gamma_perclade_alpha_data(
        Cf, Yj, C_T, C_S, a, b, jnp.asarray(A), sig_I, omega, eps=EPS, T_sinkhorn=50, J_alt=3,
        K_outer=400, lr=0.05, alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
finally:
    S.deconvolution_loss_ls = _ORIG
alpha_k = np.asarray(res["alpha"])
loss_hist = np.asarray(res["loss_hist"])
print(f"[run] learned alpha: loss {loss_hist[0]:.5f} -> {loss_hist[-1]:.5f}, "
      f"range {alpha_k.min():.3f}-{alpha_k.max():.3f}")
results["learned per-clade"] = score(solve(alpha_k), test)
sh = [score(solve(rng.permutation(alpha_k)), test) for _ in range(10)]
sh_acc = np.array([x["acc"] for x in sh])
results["control: shuffled per-clade"] = dict(
    acc=float(np.nanmean(sh_acc)), ari=float(np.nanmean([x["ari"] for x in sh])), n=sh[0]["n"])
results["control: uniform at mean alpha"] = score(solve(np.full(K, float(alpha_k.mean()))), test)

# ---- feature route: TRAIN clone labels appended to spot features ----
z = lambda M: (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)
FEAT_W = 3.0    # weight of the clone-label block relative to a standardised gene
# Every bin needs an entry, but marking non-train bins with the train MEAN makes them
# identifiable as "not train" and pushes cells onto train bins -- which the test-bin metric
# then punishes. Use zeros, which is genuinely uninformative rather than anti-informative.
feat_st = np.zeros((m, K), np.float32)
feat_st[train] = Y_oh[train]
Cf2, CT2, CS2, a2, b2 = S.prepare_ot_inputs(
    ad.AnnData(np.hstack([z(X_sc), z(A)]).astype(np.float32)),
    ad.AnnData(np.hstack([z(X_st), feat_st * FEAT_W]).astype(np.float32)),
    np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
g = a2[:, None] * b2[None, :]
sF2, sL2 = S.make_cladexfgw_scales(Cf2, CT2, CS2, a2, b2, g, eps=EPS)
for _ in range(J):
    C = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cf2, CT2, CS2, a2, b2, g,
                                     omega, sF2, sL2, equalize="measured")
    g, _u = S.sinkhorn_unrolled_safe(C, a2, b2, EPS, T)
results["feature route (train labels)"] = score(np.asarray(g), test)

z_sh = (results["learned per-clade"]["acc"] - np.nanmean(sh_acc)) / max(np.nanstd(sh_acc), 1e-9)
print(f"\n{'configuration':<32}{'accuracy':>10}{'ARI':>9}{'n cells':>9}")
print("-" * 60)
for k, v in results.items():
    print(f"{k:<32}{v['acc']:>10.4f}{v['ari']:>9.4f}{v.get('n', 0):>9}")
print(f"\nlearned vs shuffled: {results['learned per-clade']['acc'] - np.nanmean(sh_acc):+.4f} "
      f"(shuffled {np.nanmean(sh_acc):.4f} +- {np.nanstd(sh_acc):.4f}, z = {z_sh:+.1f})")
print(f"learned vs alpha=0 : {results['learned per-clade']['acc'] - results['alpha=0']['acc']:+.4f}")

# ---- the centrepiece: does learned alpha track the independent premise measurement? ----
P = pd.read_csv(HERE / "etracer_premise.csv")
P = P[P.stage == "LE"].copy()
P["cid"] = P.clone.str.split("-").str[0]
P = P.groupby("cid").agg(r=("r", "mean"), pval=("p", "min"), n_leaves=("n_leaves", "max")).reset_index()
AK = pd.DataFrame(dict(cid=clones, alpha=alpha_k)).merge(P, on="cid", how="inner")
print(f"\n{len(AK)} clones with both a learned alpha and an independent concordance measurement")
print(AK.sort_values("alpha", ascending=False)
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if len(AK) >= 6:
    rs, ps = stats.spearmanr(AK.alpha, AK.r)
    rp, pp = stats.pearsonr(AK.alpha, AK.r)
    print(f"\n  Spearman learned alpha vs premise r = {rs:+.3f} (p = {ps:.4f})")
    print(f"  Pearson                              = {rp:+.3f} (p = {pp:.4f})")
    null = [stats.spearmanr(AK.alpha, rng.permutation(AK.r.values))[0] for _ in range(10000)]
    print(f"  permutation p (one-sided, positive)  = {np.mean(np.array(null) >= rs):.4f}")
    print(f"\n  baseline, previous expression-loss run: Spearman -0.103 (p = 0.64)")
    print("  -> " + ("alpha TRACKS the independent concordance measurement"
                     if ps < 0.05 and rs > 0 else "no relationship detected"))
AK.to_csv(HERE / "le_full_alpha_vs_premise.csv", index=False)
pd.DataFrame([dict(config=k, **v) for k, v in results.items()]).to_csv(
    HERE / "le_full_results.csv", index=False)
