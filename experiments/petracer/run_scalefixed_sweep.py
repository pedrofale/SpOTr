"""Is the monotonicity result an artefact of the cost SCALE drifting with alpha?

build_fgw_cost (global path) ends with s = Z/den, which holds median|C| ~ constant across
alpha. build_perclade_alpha_cost_v2 does NOT: its s_k equalizes across CLADES only and
collapses to exactly 1 for a uniform alpha vector. So in the per-clade path the ratio
median|C|/eps -- the real transport temperature -- drifts with alpha (measured: 1.78x on
M3-1, 1.38x on M3-4).

Every recorded monotonicity result swept alpha at fixed eps. If the cost scale drifts, those
sweeps varied the temperature too, and "the loss is monotone in alpha" may be partly "the
loss is monotone in temperature".

Minimal intervention: same cost geometry, renormalised to a fixed median magnitude at every
alpha. If the expression loss is STILL monotone, the identifiability result is robust and
this is settled. If it stops being monotone, there is a route.
"""
import sys, numpy as np, jax, jax.numpy as jnp, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from petracer_common import Sec, T, J, TAU, GRID_G
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

SEC = sys.argv[1] if len(sys.argv) > 1 else "M3-4"
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
s = Sec(SEC)
Cf, C_T, C_S, a, b, omega, sF, sL = s._p
X_sc = np.asarray(s.d["X_sc"], np.float64); X_st = np.asarray(s.d["X_st"], np.float64)
A_np = np.asarray(s.d["A"], np.float64)
g0 = a[:, None] * b[None, :]

# reference magnitude: median|C| at alpha=0
REF = float(jnp.median(jnp.abs(build_perclade_alpha_cost_v2(
    jnp.zeros(s.K, jnp.float32), Cf, C_T, C_S, a, b, g0, omega, sF, sL, equalize="measured"))))
print(f"[{SEC}] reference median|C| at alpha=0: {REF:.5f}   eps={EPS}\n")


def solve(alpha, fix_scale):
    ak = jnp.full(s.K, float(alpha), jnp.float32)
    def build(g):
        C = build_perclade_alpha_cost_v2(ak, Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                         equalize="measured")
        if fix_scale:
            m = jnp.maximum(jnp.median(jnp.abs(C)), 1e-9)
            C = C * jax.lax.stop_gradient(REF / m)
        return C
    g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
    return np.asarray(g)


def metrics(G):
    w = G.sum(0)
    pred = (G.T @ X_sc) / np.maximum(w[:, None], 1e-12)
    expr = float(np.mean((pred - X_st) ** 2))
    p = G.T @ A_np; p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    comp = float(np.mean((s.COMP - p) ** 2))
    mass = G.sum(1); mass = mass / mass.sum()
    P = G / np.maximum(G.sum(1, keepdims=True), 1e-30)
    ent = float((mass * -(P * np.log(P + 1e-30)).sum(1)).sum())
    return s.score(G), expr, comp, ent


rows = []
for v in GRID_G:
    af, ef, cf_, nf = metrics(solve(v, False))
    at, et, ct, nt = metrics(solve(v, True))
    rows.append(dict(alpha=v, acc_drift=af, expr_drift=ef, comp_drift=cf_, ent_drift=nf,
                     acc_fixed=at, expr_fixed=et, comp_fixed=ct, ent_fixed=nt))
    print(f"  alpha {v:<5} | DRIFTING scale: acc {af:.4f} expr {ef:.5f} comp {cf_:.6f} ent {nf:.3f}"
          f"  || FIXED scale: acc {at:.4f} expr {et:.5f} comp {ct:.6f} ent {nt:.3f}", flush=True)

t = pd.DataFrame(rows); t.to_csv(HERE / f"scalefixed_{SEC}.csv", index=False)
mono = lambda x: bool((np.diff(np.asarray(x)) > 0).all() or (np.diff(np.asarray(x)) < 0).all())
print("\n" + "=" * 78)
for tag in ["drift", "fixed"]:
    print(f"\n--- {tag.upper()} SCALE ---")
    print(f"  accuracy peak at alpha={t.loc[t[f'acc_{tag}'].idxmax(),'alpha']} "
          f"({t[f'acc_{tag}'].max():.4f}); alpha=0 gives {t[f'acc_{tag}'].iloc[0]:.4f}; "
          f"tree gain {t[f'acc_{tag}'].max()-t[f'acc_{tag}'].iloc[0]:+.4f}")
    print(f"  entropy range {t[f'ent_{tag}'].min():.3f}-{t[f'ent_{tag}'].max():.3f} "
          f"({t[f'ent_{tag}'].max()/max(t[f'ent_{tag}'].min(),1e-9):.2f}x)")
    for crit in ["expr", "comp"]:
        col = f"{crit}_{tag}"
        print(f"  {crit:<5} loss monotone: {str(mono(t[col])):<5}  argmin alpha="
              f"{t.loc[t[col].idxmin(),'alpha']}  (accuracy argmax "
              f"{t.loc[t[f'acc_{tag}'].idxmax(),'alpha']})  -> AGREE: "
              f"{t.loc[t[col].idxmin(),'alpha'] == t.loc[t[f'acc_{tag}'].idxmax(),'alpha']}")
