"""Does Q7 replicate across simulation seeds?

Every earlier positive on this project came from a single seed and inverted or vanished when
replicated, so a single-seed result is not evidence. Regenerates the confusable testbed under
several seeds and re-runs both questions: does a global alpha fix the mis-mapping (Q7a), and
does the oracle per-clade split beat the best global alpha against shuffled controls (Q7b).
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2

EPS, T, J = 0.02, 50, 3
GRID = [0.0, 0.10, 0.20, 0.35, 0.50, 0.70]
rows = []
for seed in range(16):
    f = HERE / ("sim_conf.npz" if seed == 0 else f"sim_conf_s{seed}.npz")
    if not f.exists():
        p = subprocess.run([sys.executable, str(HERE / "simulate_confusable.py"),
                            "--seed", str(seed), "--out", f.name], cwd=HERE, capture_output=True)
        if p.returncode != 0 or not f.exists():
            print(f"seed {seed}: simulation failed (lineage extinction) -- skipped")
            continue
    d = np.load(f, allow_pickle=True)
    rg = np.random.default_rng(seed)
    xy_sc, xy_st = d["xy_sc"], d["xy_st"]
    _D = cdist(xy_sc, xy_st) / np.linalg.norm(xy_st.max(0) - xy_st.min(0))
    cell_clade, spot_clade = d["cell_clade"], d["spot_clade"]
    clades, is_disp, pair_id = d["clades"], d["is_dispersed"], d["pair_id"]
    K = len(clades)
    cidx = np.array([list(clades).index(c) for c in cell_clade])
    spot_ci = np.array([list(clades).index(c) for c in spot_clade])
    disp_cell, has_pair = is_disp[cidx], pair_id[cidx] >= 0
    A = np.zeros((len(cell_clade), K), dtype=np.float32); A[np.arange(len(cell_clade)), cidx] = 1.0
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
        # `pair` is UNCONDITIONAL accuracy on cells whose clade has a confusable partner.
        # The earlier conditional version ("given you landed in the right pair, was it the
        # right member?") conditions on the very assignment it scores, so its denominator
        # moves between settings and the numbers are not comparable.
        if not np.isfinite(g).all():
            return dict(pair=np.nan, acc=np.nan, acc_c=np.nan, acc_d=np.nan)
        j = g.argmax(1); ok = spot_ci[j] == cidx
        return dict(pair=ok[has_pair].mean(), acc=ok.mean(),
                    acc_c=ok[~disp_cell].mean(), acc_d=ok[disp_cell].mean())

    sw = {v: M(solve(np.full(K, v))) for v in GRID}
    bv = max(sw, key=lambda k: sw[k]["pair"])
    need = (~is_disp).astype(float)
    best, BV, cfg = -np.inf, None, None
    # lo == hi would make the per-clade vector CONSTANT, so shuffling and inverting it are
    # no-ops and the controls become vacuous. Every candidate must have a real spread.
    for lo, hi in [(0.0, 0.35), (0.0, 0.50), (0.0, 0.70), (0.05, 0.50), (0.05, 0.70), (0.10, 0.70)]:
        v = M(solve(lo + (hi - lo) * need))
        if v["pair"] > best:
            best, BV, cfg = v["pair"], v, (lo, hi)
    lo, hi = cfg
    SH = [M(solve(lo + (hi - lo) * rg.permutation(need)))["pair"] for _ in range(6)]
    IV = M(solve(lo + (hi - lo) * (1 - need)))["pair"]
    z = (BV["pair"] - np.nanmean(SH)) / max(np.nanstd(SH), 1e-9)
    rows.append(dict(seed=seed, alpha=bv, pair0=sw[0.0]["pair"], pair_g=sw[bv]["pair"],
                     pair_o=BV["pair"], shuf=float(np.nanmean(SH)), inv=IV, z=z,
                     dc0=sw[0.0]["acc_d"], dcg=sw[bv]["acc_d"]))
    print(f"seed {seed}: pair {sw[0.0]['pair']:.4f} (a=0) -> {sw[bv]['pair']:.4f} (a={bv}) "
          f"-> oracle {BV['pair']:.4f} | shuf {np.nanmean(SH):.4f} inv {IV:.4f} z={z:+.1f}")

r = pd.DataFrame(rows)
r["q7a"] = r.pair_g - r.pair0
r["q7b"] = r.pair_o - r.pair_g
print("\n" + "=" * 92)
print(r[["seed", "alpha", "pair0", "pair_g", "pair_o", "q7a", "q7b", "shuf", "inv", "z"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nQ7a  global alpha vs alpha=0 : mean {r.q7a.mean():+.4f}  "
      f"({', '.join(f'{v:+.4f}' for v in r.q7a)})  positive in {int((r.q7a>0.005).sum())}/{len(r)}")
print(f"Q7b  oracle vs best global   : mean {r.q7b.mean():+.4f}  "
      f"({', '.join(f'{v:+.4f}' for v in r.q7b)})  positive in {int((r.q7b>0.005).sum())}/{len(r)}")
print(f"     oracle vs shuffled z    : {', '.join(f'{v:+.1f}' for v in r.z)}")
print(f"     inverted always worst   : {bool((r.inv < r.shuf).all())}")
print(f"     tree HURTS dispersed    : mean {(r.dcg-r.dc0).mean():+.4f} "
      f"({', '.join(f'{v:+.4f}' for v in (r.dcg-r.dc0))})")
r.to_csv(HERE / "replicate_confusable.csv", index=False)
