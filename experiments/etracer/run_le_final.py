"""SpOTr on eTRACER late-escape, with the barcode anchor split three ways.

check_loss_split.py established that the barcode loss OVERFITS the bins it is fitted on: the
fit-split loss is monotone increasing in alpha and selects alpha = 0, while the same loss
evaluated on HELD-OUT labelled bins has an interior minimum at 0.075 -- which is also the alpha
the original eTRACER sweep found best. So alpha must be selected against held-out anchor bins.

Labelled bins are therefore split three ways:
  FIT   40%  the loss that alpha_k is fitted against
  VAL   30%  selects along the optimisation trajectory (early stopping) and the global alpha
  TEST  30%  never touched until scoring

Every cell is scored by restricting its coupling row to TEST bins, so all configurations are
compared on the same cells.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy import stats
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
STAGE = sys.argv[1] if len(sys.argv) > 1 else "LE"
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
S.build_perclade_alpha_cost = functools.partial(build_perclade_alpha_cost_v2, equalize="measured")
_ORIG = S.deconvolution_loss_ls

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.45]
rng = np.random.default_rng(0)

d = np.load(HERE / f"etracer_{STAGE}_full.npz", allow_pickle=True)
print(f"[run] STAGE = {STAGE}")
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clone = np.asarray(d["cell_clone"]).astype(str)
bin_clone = np.asarray(d["bin_clone"]).astype(str)
clones = np.asarray(d["clones"]).astype(str)
A = np.asarray(d["A"], np.float32); is_lab = np.asarray(d["is_labelled"], bool)
K, m = len(clones), X_st.shape[0]

lab_idx = np.where(is_lab)[0]
split = np.full(m, "none", dtype=object)
for c in clones:
    ii = rng.permutation(lab_idx[bin_clone[lab_idx] == c])
    n1, n2 = int(round(0.4 * len(ii))), int(round(0.7 * len(ii)))
    split[ii[:n1]] = "fit"; split[ii[n1:n2]] = "val"; split[ii[n2:]] = "test"
fit_m, val_m, test_m = (split == "fit"), (split == "val"), (split == "test")
print(f"[run] {len(cell_clone)} cells x {m} bins, {K} clones | labelled {is_lab.sum()} "
      f"-> fit {fit_m.sum()} / val {val_m.sum()} / test {test_m.sum()}")

bidx = np.array([list(clones).index(c) if c in list(clones) else -1 for c in bin_clone])
Y = np.zeros((m, K), np.float32); Y[np.arange(m)[bidx >= 0], bidx[bidx >= 0]] = 1.0
Yj = jnp.asarray(Y)
Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
fit_mask = jnp.asarray(fit_m.astype(np.float32))[:, None]


def fit_loss(Yv, gam, ca, cc, _m=fit_mask, _Y=Yj):
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


def loss_on(g, sel):
    p = np.asarray(g).T @ A
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return float((((Y - p) ** 2)[sel]).sum() / (sel.sum() * K))


cols_test = np.where(test_m)[0]
def score(g):
    if not np.isfinite(g).all():
        return dict(acc=np.nan, ari=np.nan)
    j = cols_test[np.asarray(g)[:, cols_test].argmax(1)]
    return dict(acc=float((bin_clone[j] == cell_clone).mean()),
                ari=float(adjusted_rand_score(cell_clone, bin_clone[j])))


sw = {v: solve(np.full(K, v)) for v in GRID}
vl = {v: loss_on(g, val_m) for v, g in sw.items()}
bv = min(vl, key=lambda k: vl[k])
print(f"[run] global alpha selected on VAL loss: {bv}")
print("      " + "  ".join(f"{v}:{vl[v]:.5f}" for v in GRID))

S.deconvolution_loss_ls = fit_loss
try:
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    res = S.learn_alpha_gamma_perclade_alpha_data(
        Cf, Yj, C_T, C_S, a, b, jnp.asarray(A), jnp.asarray(2.0 * np.eye(K, dtype=np.float32)),
        omega, eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=400, lr=0.05,
        alpha_init=jnp.full((K,), 0.10), gamma0=gi, use_loss_ls=True)
finally:
    S.deconvolution_loss_ls = _ORIG

hist = np.asarray(res["alpha_hist"])            # [K, n_recorded]
print(f"[run] alpha trajectory recorded at {hist.shape[1]} points")
val_curve = [loss_on(solve(hist[:, t]), val_m) for t in range(hist.shape[1])]
best_t = int(np.argmin(val_curve))
alpha_k = hist[:, best_t]
print(f"[run] early stopping on VAL: point {best_t}/{hist.shape[1]-1}, "
      f"val loss {val_curve[0]:.5f} -> {val_curve[best_t]:.5f} | "
      f"alpha mean {alpha_k.mean():.3f}, range {alpha_k.min():.3f}-{alpha_k.max():.3f}")

results = {"alpha = 0": score(sw[0.0]), f"best global (VAL-selected, a={bv})": score(sw[bv]),
           "learned per-clade (VAL early-stop)": score(solve(alpha_k)),
           "control: uniform at mean alpha": score(solve(np.full(K, float(alpha_k.mean()))))}
sh = [score(solve(rng.permutation(alpha_k))) for _ in range(10)]
sa = np.array([x["acc"] for x in sh])
results["control: shuffled per-clade"] = dict(acc=float(np.nanmean(sa)),
                                              ari=float(np.nanmean([x["ari"] for x in sh])))
print(f"\n{'configuration':<40}{'accuracy':>10}{'ARI':>9}")
print("-" * 59)
for k, v in results.items():
    print(f"{k:<40}{v['acc']:>10.4f}{v['ari']:>9.4f}")
lp = results["learned per-clade (VAL early-stop)"]["acc"]
um = results["control: uniform at mean alpha"]["acc"]
print(f"\nlearned vs uniform-at-same-mean : {lp - um:+.4f}   <- isolates the PER-CLADE part")
print(f"learned vs shuffled             : {lp - np.nanmean(sa):+.4f} "
      f"(shuffled {np.nanmean(sa):.4f} +- {np.nanstd(sa):.4f}, z = {(lp-np.nanmean(sa))/max(np.nanstd(sa),1e-9):+.1f})")
print(f"learned vs alpha=0              : {lp - results['alpha = 0']['acc']:+.4f}")

P = pd.read_csv(HERE / "etracer_premise.csv"); P = P[P.stage == STAGE].copy()
P["cid"] = P.clone.str.split("-").str[0]
P = P.groupby("cid").agg(r=("r", "mean"), pval=("p", "min")).reset_index()
AK = pd.DataFrame(dict(cid=clones, alpha=alpha_k)).merge(P, on="cid", how="inner")
rs, ps = stats.spearmanr(AK.alpha, AK.r)
null = [stats.spearmanr(AK.alpha, rng.permutation(AK.r.values))[0] for _ in range(10000)]
print(f"\nalpha vs independent premise r, n = {len(AK)}")
print(f"  Spearman {rs:+.3f} (p = {ps:.4f}), permutation p (one-sided) = "
      f"{np.mean(np.array(null) >= rs):.4f}")
print(f"  baseline, expression-loss run: -0.103 (p = 0.64)")
nb = pd.Series(bin_clone[is_lab]).value_counts()
AK["n_lab_bins"] = AK.cid.map(nb).fillna(0).astype(int)
print(f"\n  clones with >=10 labelled bins: {(AK.n_lab_bins >= 10).sum()}/{len(AK)}, "
      f">=30: {(AK.n_lab_bins >= 30).sum()}")
for thr in [1, 5, 10, 30]:
    sub = AK[AK.n_lab_bins >= thr]
    if len(sub) >= 6:
        rr, pp2 = stats.spearmanr(sub.alpha, sub.r)
        print(f"    restricted to >={thr:>2} bins (n={len(sub):>2}): Spearman {rr:+.3f} (p={pp2:.3f})")
print(f"\n  learned alpha spread: {alpha_k.min():.4f} - {alpha_k.max():.4f} (sd {alpha_k.std():.4f})")
AK.to_csv(HERE / f"{STAGE.lower()}_final_alpha_vs_premise.csv", index=False)
pd.DataFrame([dict(config=k, **v) for k, v in results.items()]).to_csv(
    HERE / f"{STAGE.lower()}_final_results.csv", index=False)
