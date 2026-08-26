"""Claim (b) on PEtracer: does per-clade alpha beat a single well-chosen alpha?

Claim (a) is established here: soft accuracy 0.1336 -> 0.3183 (+138% relative) at tau = 0.01,
against a random-spot floor of 0.0977. This asks the harder question.

The bar is NOT beating alpha = 0 -- that only reproduces claim (a). Per-clade alpha must beat
UNIFORM-AT-THE-SAME-MEAN, with shuffled per-clade alpha as the null. On eTRACER that margin was
+0.0036, i.e. nothing, but there per-clade alpha was never estimable: 10 of 45 clones had zero
labelled bins and learned alpha spanned only 0.091-0.109. Here all 18 clades are populated.

Anchor: the spot clade composition, split fit/val/test over SPOTS. Only fit spots enter the
loss, val selects along the optimisation trajectory, test is scored and never seen. Every cell
is scored by restricting its coupling row to test spots.
"""
import functools, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed
_ORIG = S.deconvolution_loss_ls

EPS, T, J = 0.02, 50, 3
rng = np.random.default_rng(0)
d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
cell_clade = np.asarray(d["cell_clade"]).astype(str)
clades = np.asarray(d["clades"]).astype(str)
A = np.asarray(d["A"], np.float32)
K, n, m = len(clades), len(cell_clade), X_st.shape[0]
cidx = np.array([list(clades).index(c) for c in cell_clade])

import pycea as py
t = py.datasets.koblan25("tumor")
cl_of = t.obs["clade"].astype(str).to_dict()
COMP = np.zeros((m, K))
for nm, sp in zip(np.asarray(d["spot_cell_names"]).astype(str), np.asarray(d["spot_of_cell"])):
    c = cl_of.get(nm)
    if c in list(clades):
        COMP[sp, list(clades).index(c)] += 1
COMP = COMP / np.maximum(COMP.sum(1, keepdims=True), 1e-12)

perm = rng.permutation(m)
n1, n2 = int(0.4 * m), int(0.7 * m)
fit_m = np.zeros(m, bool); fit_m[perm[:n1]] = True
val_m = np.zeros(m, bool); val_m[perm[n1:n2]] = True
test_m = np.zeros(m, bool); test_m[perm[n2:]] = True
print(f"[run] {SEC} tau={TAU}: {n} cells x {m} spots, {K} clades | "
      f"fit {fit_m.sum()} / val {val_m.sum()} / test {test_m.sum()} spots")

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         np.asarray(d["C_tree"]), np.asarray(d["C_space"]))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)
Cj = jnp.asarray(COMP.astype(np.float32))
fit_mask = jnp.asarray(fit_m.astype(np.float32))[:, None]


def comp_loss(Y, gam, ca, cc, _m=fit_mask, _C=Cj):
    p = jnp.einsum('ji,ik->jk', gam.T, ca)
    p = p / jnp.maximum(jnp.sum(p, axis=1, keepdims=True), 1e-12)
    return jnp.sum(((_C - p) ** 2) * _m) / jnp.maximum(jnp.sum(_m) * p.shape[1], 1.0)


def solve(ak):
    def build(g):
        return build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)), Cf, C_T,
                                            C_S, a, b, g, omega, sF, sL, equalize="measured")
    if TAU <= 0:
        g = g0
        for _ in range(J):
            g, _u = S.sinkhorn_unrolled_safe(build(g), a, b, EPS, T)
        return np.asarray(g)
    g, _uv = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
    return np.asarray(g)


cols = np.where(test_m)[0]
def score(g):
    if not np.isfinite(g).all():
        return np.nan
    j = cols[np.asarray(g)[:, cols].argmax(1)]
    return float(COMP[j, cidx].mean())


def val_loss(g):
    p = np.asarray(g).T @ A
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    return float((((COMP - p) ** 2)[val_m]).sum() / (val_m.sum() * K))


GRID = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
sw = {v: solve(np.full(K, v)) for v in GRID}
vl = {v: val_loss(g) for v, g in sw.items()}
bv = min(vl, key=lambda k: vl[k])
print(f"[run] global alpha by VAL loss: {bv}   " + " ".join(f"{v}:{vl[v]:.5f}" for v in GRID))
print(f"[run] test soft acc by alpha:   " + " ".join(f"{v}:{score(sw[v]):.4f}" for v in GRID))

S.deconvolution_loss_ls = comp_loss
try:
    gi, _ = S.sinkhorn_fgw(Cf, C_T, C_S, a, b, EPS, T_sinkhorn=50, J_alt=1, alpha=0.0)
    res = S.learn_alpha_gamma_perclade_alpha_data(
        Cf, Cj, C_T, C_S, a, b, jnp.asarray(A), jnp.asarray(2.0 * np.eye(K, dtype=np.float32)),
        omega, eps=EPS, T_sinkhorn=50, J_alt=3, K_outer=400, lr=0.05,
        alpha_init=jnp.full((K,), float(bv) if bv > 0 else 0.10), gamma0=gi, use_loss_ls=True)
finally:
    S.deconvolution_loss_ls = _ORIG

hist = np.asarray(res["alpha_hist"])
vc = [val_loss(solve(hist[:, i])) for i in range(hist.shape[1])]
bt = int(np.argmin(vc))
ak = hist[:, bt]
print(f"[run] early stop at {bt}/{hist.shape[1]-1}; alpha mean {ak.mean():.3f}, "
      f"range {ak.min():.3f}-{ak.max():.3f}, sd {ak.std():.3f}")

out = {"alpha = 0": score(sw[0.0]), f"best global (val, a={bv})": score(sw[bv]),
       "learned per-clade": score(solve(ak)),
       "uniform at same mean": score(solve(np.full(K, float(ak.mean()))))}
sh = [score(solve(rng.permutation(ak))) for _ in range(10)]
out["shuffled per-clade (10)"] = float(np.nanmean(sh))
print(f"\n{'configuration':<34}{'soft acc':>10}")
print("-" * 44)
for k, v in out.items():
    print(f"{k:<34}{v:>10.4f}")
lp, um = out["learned per-clade"], out["uniform at same mean"]
print("-" * 44)
print(f"\nlearned vs uniform-at-same-mean : {lp - um:+.4f}   <- THE BAR for claim (b)")
print(f"learned vs shuffled             : {lp - np.nanmean(sh):+.4f} "
      f"(shuffled {np.nanmean(sh):.4f} +- {np.nanstd(sh):.4f}, "
      f"z = {(lp - np.nanmean(sh)) / max(np.nanstd(sh), 1e-9):+.1f})")
print(f"learned vs alpha=0              : {lp - out['alpha = 0']:+.4f}")
pd.DataFrame([dict(config=k, soft=v) for k, v in out.items()]).to_csv(
    HERE / f"petracer_{SEC}_perclade.csv", index=False)
pd.DataFrame(dict(clade=clades, alpha=ak)).to_csv(
    HERE / f"petracer_{SEC}_alpha.csv", index=False)
