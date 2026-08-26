"""Q9: the confusable question again, on the variance-controlled testbed.

Every quantity that drives the effect is now realized to target per seed (tree-vs-layout r
0.82-0.85, pair separation 1.93-2.06, within-pair discrimination 0.57-0.58, exact K and clade
sizes). If the seed-to-seed swings of +0.098 / -0.021 were simulator variance, they should
collapse here. If they persist, the variance is intrinsic to the OT problem rather than the
generator -- which is itself worth knowing.

  Q9a  best global alpha vs alpha = 0
  Q9b  oracle per-clade alpha vs best global, against shuffled and inverted controls
  Q9c  oracle per-clade alpha vs no tree at all
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rows = []
for seed in range(14):
    f = HERE / f"sim_ctrl_s{seed}.npz"
    if not f.exists():
        p = subprocess.run([sys.executable, str(HERE / "simulate_controlled.py"),
                            "--seed", str(seed), "--out", f.name], cwd=HERE, capture_output=True)
        if p.returncode != 0 or not f.exists():
            print(f"seed {seed}: generation failed -- skipped"); continue
    d = np.load(f, allow_pickle=True)
    if not bool(d["in_band"][0]):
        print(f"seed {seed}: realized properties OUT OF BAND -- skipped"); continue
    rg = np.random.default_rng(seed)
    cidx, spot_ci = d["cell_clade"], d["spot_clade"]
    is_disp, pair_id = d["is_dispersed"], d["pair_id"]
    K = len(d["clades"])
    disp_cell = is_disp[cidx]
    A = np.zeros((len(cidx), K), np.float32); A[np.arange(len(cidx)), cidx] = 1.0
    Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"]),
                                             d["C_tree"], d["C_space"])
    omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
    sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

    def solve(ak):
        g = g0
        for _ in range(J):
            C = build_perclade_alpha_cost_v2(jnp.asarray(np.asarray(ak, np.float32)),
                                             Cf, C_T, C_S, a, b, g, omega, sF, sL,
                                             equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, EPS, T)
        return np.asarray(g)

    def M(g):
        if not np.isfinite(g).all():
            return dict(acc=np.nan, acc_c=np.nan, acc_d=np.nan)
        ok = spot_ci[g.argmax(1)] == cidx
        return dict(acc=ok.mean(), acc_c=ok[~disp_cell].mean(), acc_d=ok[disp_cell].mean())

    sw = {v: M(solve(np.full(K, v))) for v in GRID}
    bv = max(sw, key=lambda k: sw[k]["acc"])
    need = (~is_disp).astype(float)
    best, BV, cfg = -np.inf, None, None
    for lo, hi in [(0.0, 0.35), (0.0, 0.50), (0.0, 0.70), (0.05, 0.50), (0.05, 0.70), (0.10, 0.70)]:
        v = M(solve(lo + (hi - lo) * need))
        if v["acc"] > best:
            best, BV, cfg = v["acc"], v, (lo, hi)
    lo, hi = cfg
    SH = [M(solve(lo + (hi - lo) * rg.permutation(need)))["acc"] for _ in range(6)]
    IV = M(solve(lo + (hi - lo) * (1 - need)))["acc"]
    z = (BV["acc"] - np.nanmean(SH)) / max(np.nanstd(SH), 1e-9)
    rows.append(dict(seed=seed, alpha=bv, a0=sw[0.0]["acc"], ag=sw[bv]["acc"], ao=BV["acc"],
                     shuf=float(np.nanmean(SH)), inv=IV, z=z,
                     d0=sw[0.0]["acc_d"], dg=sw[bv]["acc_d"],
                     c0=sw[0.0]["acc_c"], cg=sw[bv]["acc_c"]))
    print(f"seed {seed:>2}: a0 {sw[0.0]['acc']:.4f} -> global {sw[bv]['acc']:.4f} (a={bv}) "
          f"-> oracle {BV['acc']:.4f} | shuf {np.nanmean(SH):.4f} inv {IV:.4f} z={z:+.1f}")

r = pd.DataFrame(rows)
r["q9a"] = r.ag - r.a0; r["q9b"] = r.ao - r.ag; r["q9c"] = r.ao - r.a0
print("\n" + "=" * 96)
print(r[["seed", "alpha", "a0", "ag", "ao", "q9a", "q9b", "q9c", "shuf", "inv", "z"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
n = len(r)
for k, nm in [("q9a", "Q9a global vs alpha=0 "), ("q9b", "Q9b oracle vs global  "),
              ("q9c", "Q9c oracle vs no tree ")]:
    v = r[k]
    print(f"\n{nm}: mean {v.mean():+.4f}  sd {v.std():.4f}  "
          f"positive in {int((v > 0.005).sum())}/{n}  range [{v.min():+.4f}, {v.max():+.4f}]")
print(f"\noracle vs shuffled z : {', '.join(f'{v:+.1f}' for v in r.z)}  "
      f"(positive in {int((r.z > 0).sum())}/{n})")
print(f"inverted worst       : {int((r.inv < r.shuf).sum())}/{n} seeds")
print(f"tree hurts dispersed : mean {(r.dg - r.d0).mean():+.4f}   "
      f"helps compact: mean {(r.cg - r.c0).mean():+.4f}")
r.to_csv(HERE / "replicate_controlled.csv", index=False)
