"""Q22: how strong must the biology-dispersion link be for a transferable rule to work?

Q21 failed, but the simulator assigns dispersion at RANDOM, independent of every clade
property, so only mapping-derived features could ever predict it -- and at ~22% mapping
accuracy those are swamped by error. That is an adversarial worst case.

In real tissue, dispersion plausibly tracks observable clade biology: plastic or migratory
clones are both more transcriptionally heterogeneous and more spatially scattered. Here that
link is introduced explicitly, with a dial, as extra program-mixture breadth in dispersed
clades -- observable from scRNA-seq alone, no spatial information involved. Sweeping the dial
converts "does the transfer idea work" into "how strong does the link have to be", which is
something a real dataset can be measured against.
"""
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

FEATS = ["size", "expr_sep", "tree_ratio", "disp", "bimod", "entropy", "eff_spots", "het"]
SIGNALS = [0.0, 0.15, 0.3, 0.6]
out = []
for sig_str in SIGNALS:
    rows = []
    for seed in range(8):
        tag = f"sim_bio{sig_str}_s{seed}.npz".replace(".", "p", 1)
        f = HERE / tag
        if not f.exists():
            p = subprocess.run([sys.executable, str(HERE / "simulate_controlled.py"),
                                "--seed", str(seed), "--disp-signal", str(sig_str),
                                "--out", f.name], cwd=HERE, capture_output=True)
            if p.returncode != 0 or not f.exists():
                continue
        d = np.load(f, allow_pickle=True)
        if not bool(d["in_band"][0]):
            continue
        cidx, is_disp, X_sc = d["cell_clade"], d["is_dispersed"], d["X_sc"]
        K = len(d["clades"]); n = len(cidx)
        A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0
        Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(d["X_st"]),
                                                 d["C_tree"], d["C_space"])
        g0 = a[:, None] * b[None, :]
        sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=0.02)
        from spotr.perclade_v2 import build_perclade_alpha_cost_v2
        g = g0
        for _ in range(3):
            C = build_perclade_alpha_cost_v2(jnp.zeros((K,), jnp.float32), Cf, C_T, C_S,
                                             a, b, g, jnp.asarray(A), sF, sL, equalize="measured")
            g, _ = S.sinkhorn_unrolled_safe(C, a, b, 0.02, 50)
        gn = np.asarray(g)
        w = gn / np.maximum(gn.sum(1, keepdims=True), 1e-30)
        pred = w @ d["xy_st"]
        diam = np.linalg.norm(d["xy_st"].max(0) - d["xy_st"].min(0))
        Ct = np.asarray(d["C_tree"], float)
        ent = -(w * np.log(np.maximum(w, 1e-30))).sum(1)
        for k in range(K):
            sel = cidx == k
            P = pred[sel]
            km = KMeans(n_clusters=2, n_init=6, random_state=0).fit(P)
            within = np.mean([np.linalg.norm(P[km.labels_ == c] - km.cluster_centers_[c], axis=1).mean()
                              for c in (0, 1) if (km.labels_ == c).sum() > 1])
            tot = max(np.linalg.norm(P - P.mean(0), axis=1).mean(), 1e-12)
            mass = w[sel].sum(0); mass = mass / max(mass.sum(), 1e-30)
            rows.append(dict(seed=seed, dispersed=int(is_disp[k]), size=sel.sum(),
                             expr_sep=cross_val_score(
                                 LogisticRegression(max_iter=500, class_weight="balanced"),
                                 X_sc, sel.astype(int), cv=3, scoring="balanced_accuracy").mean(),
                             tree_ratio=Ct[np.ix_(sel, sel)].mean() / max(Ct[np.ix_(sel, ~sel)].mean(), 1e-12),
                             disp=tot / diam, bimod=within / tot, entropy=ent[sel].mean(),
                             eff_spots=float(np.exp(-(mass * np.log(np.maximum(mass, 1e-30))).sum())),
                             het=float(np.asarray(X_sc)[sel].std(0).mean())))
    F = pd.DataFrame(rows)
    if not len(F):
        continue
    aucs = []
    for s in sorted(F.seed.unique()):
        tr, te = F[F.seed != s], F[F.seed == s]
        if te.dispersed.nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[FEATS])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            sc.transform(tr[FEATS]), tr.dispersed)
        aucs.append(roc_auc_score(te.dispersed, clf.predict_proba(sc.transform(te[FEATS]))[:, 1]))
    # how visible is the link, measured directly?
    link = roc_auc_score(F.dispersed, F.het)
    out.append(dict(signal=sig_str, n_clades=len(F), loso_auc=np.mean(aucs),
                    loso_sd=np.std(aucs), link_auc=link))
    print(f"disp_signal {sig_str:>4}: heterogeneity->dispersed AUC {link:.3f}   "
          f"leave-one-seed-out AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")

O = pd.DataFrame(out)
print("\n" + "=" * 70)
print(O.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
ok = O[O.loso_auc > 0.70]
print(f"\ntransferable rule (LOSO AUC > 0.70) reached at disp_signal >= "
      + (f"{ok.signal.min()}, where the raw link is AUC {ok.link_auc.min():.2f}"
         if len(ok) else "none of the tested levels"))
O.to_csv(HERE / "q22_bio_link.csv", index=False)
