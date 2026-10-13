"""Per-clade mixing lengths against features of the clones (2026-10-13).

The learned L_tau of 70_spotr.py (24 PEtracer splits: 3 sections x 8 seeds), averaged over a section's 8 splits,
against clone features fixed BEFORE looking at any result.  Primary (Pedro's hypothesis, 2026-10-13: mixing length
as migratory potential):
  0 EMT score              mean over the clade's cells of z(mesenchymal) - z(epithelial), the panel's canonical
                           markers: Zeb1, Snai1, Twist1, Vcan against Cdh1, Epcam, Krt19, Cldn4 (gene lists written
                           from standard EMT markers before the panel was inspected; these are the ones it carries)
Secondary:
  1 true spatial spread    the clade's mixing length on the cells' true positions (what L_tau should estimate)
  2 fitness                mean of PEtracer's Local Branching Index over the clade's cells
  3 boundary distance      mean distance of the clade's cells to the tumour boundary (migratory potential)
  4 expression plasticity  entropy of the clade's cells over the expression (Leiden) clusters
  5 clade size             number of traced cells (control: L is defined beyond clone packing)
  6 clade age              age of the clade's common ancestor, days
Two learned estimates (shared start 'per-clade', anchor-moment start 'per-clade-mom') and the true spread itself.
Clades with at least 30 cells in the mapped half.  Association: Spearman on ranks taken within each section and
pooled; p from 100,000 permutations of the L values within section; 95% interval from a stratified bootstrap.  All rows reported; 'significant' is p < 0.05
for the primary EMT test and Bonferroni over the six secondary features within a column.
Usage: python 76_perclade_features.py      (writes spotr_out/perclade_features.{csv,json})
"""
import json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


m70 = mod("m70", "70_spotr.py"); m55 = mod("m55", "55_decomposition_check.py")
import treedata as td
TD = td.read_h5td("/home/pedro/.treedata/datasets/koblan25_tumor.h5td"); OBS = TD.obs
MES = ["Vim", "Cdh2", "Fn1", "Zeb1", "Zeb2", "Snai1", "Snai2", "Twist1", "Twist2", "Mmp2", "Mmp9", "Mmp3", "Sparc", "Col1a1",
       "Col1a2", "Col3a1", "Col5a1", "Acta2", "Tgfb1", "Tgfbi", "Itgav", "Itga5", "Itgb1", "Fbn1", "Postn", "Tnc", "Serpine1",
       "Lox", "Loxl2", "Cald1", "Tagln", "Thy1", "Pdgfrb", "S100a4", "Fap", "Vcan", "Spp1", "Mmp14", "Cdh11", "Prrx1", "Tpm2",
       "Lgals1", "Cd44"]
EPI = ["Cdh1", "Epcam", "Krt8", "Krt18", "Krt19", "Krt7", "Krt5", "Krt14", "Cldn3", "Cldn4", "Cldn7", "Ocln", "Tjp1", "Esrp1",
       "Esrp2", "Grhl2", "Ovol1", "Ovol2", "Elf3", "Spint1", "Spint2", "Muc1", "Dsp", "Jup", "Cdh3", "Krt17", "Epha2"]
GENES = [str(g) for g in TD.var_names]
MI = [GENES.index(g) for g in MES if g in GENES]; EI = [GENES.index(g) for g in EPI if g in GENES]
print("EMT score: mesenchymal", [GENES[i] for i in MI], "epithelial", [GENES[i] for i in EI])


def emt_scores(cells):
    """per cell: library-normalised log expression, z-scored over the given cells; mean z(mes) - mean z(epi)"""
    X = TD[np.asarray(cells)].X
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, float)
    X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1.0) * 1e4)
    Zs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    return pd.DataFrame({"emt": Zs[:, MI].mean(1) - Zs[:, EI].mean(1), "mes": Zs[:, MI].mean(1), "epi": Zs[:, EI].mean(1)},
                        index=np.asarray(cells))
RES = sum([json.load(open(HERE / "spotr_out" / f)) for f in ["real.json", "real_more.json", "real_mom.json", "real_s47.json"]], [])
SECS, MIN_CELLS, NPERM = ["M3-4", "M3-1", "M3-2"], 30, 100000


def entropy(labels):
    p = pd.Series(labels).value_counts(normalize=True).values
    return float(-(p * np.log(p)).sum())


rows = []
for sec in SECS:
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    pool = np.asarray(z["cell_names"]); spot_cells = np.asarray(z["spot_cell_names"])
    clade_all = OBS["clade"].astype(str)
    names = sorted(set(clade_all.loc[pool]) | set(clade_all.loc[spot_cells]))
    # the whole dissociated pool (both halves) for the true spread and the clade's age
    AGE = np.asarray(z["C_tree"], float) / 2.0
    cl_pool = np.array([names.index(c) for c in clade_all.loc[pool]])
    S = m55.load(sec)
    Xfull = dict(xy_true=np.asarray(z["xy_sc"], float), NC=m55.clone_size(AGE).astype(float), rho=len(pool) / S["area"],
                 cl=cl_pool)
    meas = m70.measured_L(Xfull, cl_pool, m70.LAWS[sec][1])
    traced = OBS.loc[np.r_[pool, spot_cells]]                  # every traced cell of the section used in the study
    emt = emt_scores(traced.index)
    for k, name in enumerate(names):
        rec = {}
        for arm in ["per-clade", "per-clade-mom"]:
            rs = [r for r in RES if r["arm"] == arm and r["sec"] == sec]
            rec[arm] = np.array([np.log(r["L"][k]) for r in rs]); rec["size_half"] = np.mean([r["sizes"][k] for r in rs])
        cells = traced[traced["clade"].astype(str) == name]
        idx = np.where(cl_pool == k)[0]
        rows.append(dict(section=sec, clade=name, k=k, cells=int(len(cells)), size_half=float(rec["size_half"]),
                         logL_shared=float(rec["per-clade"].mean()), logL_shared_sd=float(rec["per-clade"].std()),
                         logL_anchor=float(rec["per-clade-mom"].mean()), logL_anchor_sd=float(rec["per-clade-mom"].std()),
                         n_seeds=int(len(rec["per-clade"])),
                         true_spread=float(meas.get(k, np.nan)),
                         emt=float(emt.loc[cells.index, "emt"].mean()), mes=float(emt.loc[cells.index, "mes"].mean()),
                         epi=float(emt.loc[cells.index, "epi"].mean()),
                         fitness=float(cells["fitness"].mean()),
                         boundary_dist=float(cells["tumor_boundary_dist"].mean()),
                         plasticity=entropy(cells["leiden_cluster"].astype(str)),
                         size=int(len(cells)),
                         age=float(AGE[np.ix_(idx, idx)].max()) if len(idx) > 1 else np.nan))
D = pd.DataFrame(rows)
D = D[(D.size_half >= MIN_CELLS) & D.true_spread.notna()].copy()
D["log_true_spread"] = np.log(D.true_spread)
# relative to the section: a section's level is set by its transferred law, not by its clades
for c in ["logL_shared", "logL_anchor", "log_true_spread"]:
    D[c + "_rel"] = D[c] - D.groupby("section")[c].transform("mean")
D.to_csv(HERE / "spotr_out" / "perclade_features.csv", index=False)


def within_rank(v, sec):
    """ranks taken within each section, scaled to [-0.5, 0.5], so sections pool without their levels"""
    out = np.empty(len(v))
    for s in np.unique(sec):
        i = np.where(sec == s)[0]
        r = pd.Series(v[i]).rank().values
        out[i] = (r - 1) / max(len(i) - 1, 1) - 0.5
    return out


def assoc(x, y, sec, rng, B=NPERM, NBOOT=4000):
    """pooled within-section rank correlation; permutation p (L shuffled within section, vectorised) and a
    stratified-bootstrap 95% interval (clades resampled with replacement within section, ranks recomputed)"""
    rx, ry = within_rank(x, sec), within_rank(y, sec)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    groups = [np.where(sec == s)[0] for s in np.unique(sec)]
    perm = np.empty((B, len(rx)), dtype=int)
    for i in groups:
        perm[:, i] = i[np.argsort(rng.random((B, len(i))), axis=1)]
    Rp = rx[perm]; Rp = Rp - Rp.mean(1, keepdims=True); yc = ry - ry.mean()
    null = (Rp @ yc) / (np.sqrt((Rp ** 2).sum(1)) * np.sqrt((yc ** 2).sum()))
    p = float((np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (B + 1))
    boots = []
    for _ in range(NBOOT):
        idx = np.concatenate([rng.choice(i, len(i), replace=True) for i in groups])
        bx, by = within_rank(x[idx], sec[idx]), within_rank(y[idx], sec[idx])
        if bx.std() > 0 and by.std() > 0:
            boots.append(np.corrcoef(bx, by)[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, p, float(lo), float(hi)


FEATS = [("emt", "EMT score (primary)"), ("true_spread", "true spatial spread"), ("fitness", "fitness (LBI)"), ("boundary_dist", "distance to tumour boundary"),
         ("plasticity", "expression plasticity"), ("size", "clade size"), ("age", "clade age")]
ESTS = [("logL_shared", "learned L, shared start"), ("logL_anchor", "learned L, anchor start"), ("log_true_spread", "true spread")]
rng = np.random.default_rng(0)
out = []
sec = D.section.values
for e, elab in ESTS:
    for f, flab in FEATS:
        if e == "log_true_spread" and f == "true_spread":
            continue
        ok = D[f].notna().values
        rho, p, lo, hi = assoc(D[e].values[ok], D[f].values[ok], sec[ok], rng)
        primary = f == "emt"
        out.append(dict(estimate=elab, feature=flab, rho=rho, p=p, lo=lo, hi=hi, n=int(ok.sum()),
                        significant=bool(p < 0.05) if primary else bool(p < 0.05 / 6)))
for e, elab in ESTS:
    for f, flab in [("mes", "  mesenchymal half (sensitivity)"), ("epi", "  epithelial half (sensitivity)")]:
        rho, p, lo, hi = assoc(D[e].values, D[f].values, sec, rng)
        out.append(dict(estimate=elab, feature=flab, rho=rho, p=p, lo=lo, hi=hi, n=int(len(D)), significant=None))
R = pd.DataFrame(out)
print(f"{len(D)} clade-sections ({D.groupby('section').size().to_dict()})")
print(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
json.dump(dict(n_clades=int(len(D)), per_section=D.groupby("section").size().to_dict(), tests=out),
          open(HERE / "spotr_out" / "perclade_features.json", "w"), indent=1)
