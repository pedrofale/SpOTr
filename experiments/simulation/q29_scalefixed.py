"""Q29: is the 11/11 "expression loss monotone, argmin pinned at alpha=0" result an artefact
of the COST SCALE drifting with alpha?

build_perclade_alpha_cost_v2's equalizer works ACROSS CLADES only -- for a uniform alpha
vector every m_k is equal, so s_k == 1 and the global magnitude of C is left free. Since
C0 = (1-a)*F_hat + a*L_hat with both normalised to median|.| ~ 1, median|C0| dips at mid-alpha.
Under a shared eps that changes the transport, so the alpha sweep varied two things.

On PEtracer M3-4 fixing the scale flipped the expression loss from monotone (argmin 0.0) to
non-monotone (argmin 0.2). This is the same test on the CONTROLLED SIMULATOR, where the
11/11 result actually lives, and where the accuracy optimum is 0.10-0.35.

Identical to q10_loss_alignment.py except for the one-line renormalisation.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists(): continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]): continue
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    K = len(d["clades"])
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    X_sc = d["X_sc"]
    sig = jnp.asarray(np.stack([X_sc[cidx == k].mean(0) for k in range(K)]).astype(np.float32) * 2.0)
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    Y = jnp.asarray(np.asarray(d["X_st"], np.float32))
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    def cost(av, g):
        return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32),
                                            Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                            equalize="measured")
    REF = float(jnp.median(jnp.abs(cost(0.0, g0))))

    for av in GRID:
        for fix in (False, True):
            g = g0
            for _ in range(J):
                C = cost(av, g)
                if fix:
                    C = C * (REF / jnp.maximum(jnp.median(jnp.abs(C)), 1e-9))
                g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
            gn = np.asarray(g)
            rows.append(dict(seed=seed, alpha=av, fixed=fix,
                             loss=float(S.deconvolution_loss_ls(Y, g, jnp.asarray(A), sig)),
                             acc=float((spot_ci[gn.argmax(1)] == cidx).mean()),
                             medC=float(jnp.median(jnp.abs(C)))))
    print(f"  seed {seed} done", flush=True)

r = pd.DataFrame(rows); r.to_csv(HERE / "q29_scalefixed.csv", index=False)
mono = lambda x: bool((np.diff(np.asarray(x)) > 0).all() or (np.diff(np.asarray(x)) < 0).all())
print("\n" + "=" * 96)
for fix in (False, True):
    tag = "FIXED SCALE" if fix else "DRIFTING SCALE (as shipped)"
    sub = r[r.fixed == fix]
    per = []
    for s_, g_ in sub.groupby("seed"):
        g_ = g_.set_index("alpha").sort_index()
        per.append(dict(seed=s_, argmin=float(g_.loss.idxmin()), argmax=float(g_.acc.idxmax()),
                        mono=mono(g_.loss.values), acc_sel=float(g_.acc[g_.loss.idxmin()]),
                        acc_a0=float(g_.acc[0.0]), acc_opt=float(g_.acc.max()),
                        medC_range=float(g_.medC.max() / g_.medC.min())))
    p = pd.DataFrame(per)
    print(f"\n--- {tag} ---   n={len(p)} seeds")
    print(f"  median|C| max/min across alpha : {p.medC_range.mean():.3f}x")
    print(f"  loss monotone in alpha         : {int(p.mono.sum())}/{len(p)} seeds")
    print(f"  argmin(loss) per seed          : {', '.join(f'{v:.2f}' for v in p.argmin)}")
    print(f"  argmax(acc)  per seed          : {', '.join(f'{v:.2f}' for v in p.argmax)}")
    print(f"  ARGMIN AGREEMENT               : {int((np.abs(p.argmin-p.argmax)<1e-9).sum())}/{len(p)}")
    print(f"  argmin inside accuracy plateau [0.10, 0.35]: "
          f"{int(((p.argmin>=0.10)&(p.argmin<=0.35)).sum())}/{len(p)}")
    print(f"  mean |argmin - argmax|         : {np.abs(p.argmin-p.argmax).mean():.3f}")
    print(f"  ACCURACY at selected alpha     : {p.acc_sel.mean():.4f}  "
          f"(alpha=0 {p.acc_a0.mean():.4f}, oracle {p.acc_opt.mean():.4f}, "
          f"fixed a=0.2 {sub[sub.alpha==0.20].groupby('seed').acc.mean().mean():.4f})")
    print(f"  beats alpha=0 in               : {int((p.acc_sel>p.acc_a0).sum())}/{len(p)} seeds")
