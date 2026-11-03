"""The frozen method on HTAN (Mo et al. 2024, WUSTL): cell2location + the GW-regularised traced map with clone labels (324_'s general runner,
within_free and its no-tree arm), and cell2location alone (328_'s transcription), scored as decision-htan-application-goals asks.  2026-10-30.
Pedro, 2026-10-29: "Let's just freeze and move on with testing on the other missing data sets", "Let's skip B189 ... Sorry I meant HTAN",
"Go ahead" (step 1 of the plan: the two-clone problems prepared by the HTAN screen).
Sections: HT260C1 (the authors' c1 / c2 split; one section), HT268B1 U12 and K3 (B against A); two clones each, so the clone tree is clone
labels; the clones are copy-number clones, so there is no lineage tree.
Comparators already on disk (htan_screen/spotr_out/<section>/): exact OT, Tangram alone, DestVI alone.
cell2location alone: types = the non-tumour cell types (fewer than 30 cells pooled as 'other', as 324_), the untraced tumour cells and each
clone; 324_'s 500 most variable fitted genes; N = 10 cells per spot (324_'s abundance prior); a clone's abundance is w_sf's posterior mean.
Readouts (decision-htan-application-goals, goal 1): case_score.py's Spearman against the spots' copy-number reference (genes SpOTr never
sees): (i) B's share of the traced mass on the spots that hold traced mass, (ii) (B - A) per unit mass on all tumour spots; held-out genes
(324_'s readout).
FIXED BEFORE RUNNING:
  E175  score (ii) of SpOTr (clone labels) >= max(exact OT, Tangram alone, cell2location alone) in >= 2 of 3 sections
  E176  SpOTr's held-out gene Spearman >= Tangram alone's in >= 2 of 3 sections
  read  score (i); the no-tree arm against SpOTr (what the GW term does); U12, where B is 16% of the nuclei but ~3% of the tumour spots
Usage: python 335_htan_c2l.py    (writes spotr_out/spotr_c2l_htan/<section>.json)
"""
import os, json, time, importlib.util
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE / "spotr_out" / "spotr_c2l_htan"; DATA = HERE.parents[1] / "data" / "htan"; SCR = HERE.parent / "htan_screen" / "spotr_out"
SECTIONS = ["HT260C1-Th1K1Fc2U1Z1Bs1", "HT268B1-Th1H3Fc2U12Z1Bs1", "HT268B1-Th1K3Fc2U1Z1Bs1"]


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def c2l_alone(path, m324, m328):
    """cell2location alone on 324_'s genes and types, each clone a type; returns (spots x clones) masses and held-out scores"""
    from scipy.stats import rankdata
    z = dict(np.load(path, allow_pickle=True)); Xc, Y = z["Xc_fit"].astype(float), z["Ys_fit"].astype(float); ct = np.asarray(z["cell_type"]).astype(str)
    u, cnt = np.unique(ct, return_counts=True); ct = np.where(np.isin(ct, u[cnt < m324.MIN_TYPE]), "other", ct)
    inform = np.asarray(z["clone_informative"], bool); clone_all = z["clone"].astype(int); traced = inform[clone_all]; Kc = int(inform.sum())
    remap = -np.ones(len(inform), int); remap[inform] = np.arange(Kc)
    L = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1.0) * 1e4); gsel = np.argsort(-L.var(0))[: min(m324.G_MAX, Xc.shape[1])]
    tk = sorted({k for k in ct[traced]}); labs = np.where(np.isin(ct, tk), "untraced", ct); labs[traced] = [f"c{remap[c]}" for c in clone_all[traced]]
    ul = sorted(set(labs.tolist())); sig = np.stack([Xc[labs == v][:, gsel].mean(0) for v in ul]); w, info = m328.c2l_basic(Y[:, gsel], sig, 10.0, x64=True, init_ls=0.01)   # float32 diverged on Visium counts
    Pm = np.stack([w[:, ul.index(f"c{k}")] for k in range(Kc)], 1)
    lib = z["lib"].astype(float); Cev = z["Xc_ev"].astype(float) / lib[:, None]; Ev = z["Ys_ev"].astype(float) / np.maximum(z["spot_total"].astype(float), 1.0)[:, None]
    pred = w @ np.stack([Cev[labs == v].mean(0) for v in ul]); ra, rb = np.apply_along_axis(rankdata, 0, pred), np.apply_along_axis(rankdata, 0, Ev); ra -= ra.mean(0); rb -= rb.mean(0)
    r = (ra * rb).sum(0) / np.maximum(np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0)), 1e-300)
    ev = [str(g) for g in z["ev_genes"]]
    return Pm, dict(heldout_spearman=float(np.nanmean(r)), **{f"spearman_{g}": float(r[ev.index(g)]) for g in ["Ankrd1", "Meox1", "EGFR"] if g in ev}), dict(info=info, types=ul)


def ref_scores(sec, m):
    """case_score.py's two Spearman correlations against the section's copy-number reference"""
    import pandas as pd
    z = np.load(DATA / f"{sec}.npz", allow_pickle=True); sb = z["spot_barcodes"]
    ref = pd.read_csv(DATA / f"{sec}_reference.tsv", sep="\t").set_index("barcode")["ref_c2"]
    idx = pd.Series(np.arange(len(sb)), index=sb).reindex(ref.index).to_numpy(); r = ref.to_numpy()
    m = np.asarray(m, float)[idx]; t = m[:, 0] + m[:, 1]; ok = t > 1e-12 * t.max(); d = (m[:, 0] - m[:, 1]) / np.maximum(m.sum(1), 1e-300)
    return dict(rho_traced=float(pd.Series(m[ok, 0] / t[ok]).corr(pd.Series(r[ok]), method="spearman")), spots=int(ok.sum()),
                rho_all=float(pd.Series(d).corr(pd.Series(r), method="spearman")))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    m324, m328 = mod("m324", "324_c2l_general.py"), mod("m328", "328_c2l_trees.py"); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); e175, e176 = [], []
    for sec in SECTIONS:
        f = OUT / f"{sec}.json"; print(f"{sec}: start", flush=True)
        if not (f.exists() and json.load(open(f)).get("done")):
            m324.run(sec, DATA / f"{sec}.npz", f)
        out = json.load(open(f))
        if "c2l" not in out["masses"] or not np.isfinite(np.asarray(out["masses"]["c2l"], float)).all():
            Pm, h, rec = c2l_alone(DATA / f"{sec}.npz", m324, m328); out["masses"]["c2l"] = Pm.tolist(); out["arms"]["c2l"] = dict(scores=h, **rec)
        prev = json.load(open(SCR / sec / f"{sec}.json")); maps = {"SpOTr": out["masses"]["within_free"], "no tree": out["masses"]["notree"], "cell2location alone": out["masses"]["c2l"],
                                                               "exact OT": np.load(SCR / sec / "exact_ot.npz")["clone_mass"], "Tangram alone": prev["masses"]["Tangram alone"]}
        out["reference"] = {k: ref_scores(sec, v) for k, v in maps.items()}
        held = {"SpOTr": out["arms"]["within_free"]["scores"]["heldout_spearman"], "no tree": out["arms"]["notree"]["scores"]["heldout_spearman"],
                "cell2location alone": out["arms"]["c2l"]["scores"]["heldout_spearman"], "Tangram alone": prev["arms"]["Tangram alone"]["heldout_spearman"]}
        out["heldout"] = held; json.dump(out, open(f, "w"), default=float); rs = out["reference"]
        e175.append(rs["SpOTr"]["rho_all"] >= max(rs[k]["rho_all"] for k in ["exact OT", "Tangram alone", "cell2location alone"])); e176.append(held["SpOTr"] >= held["Tangram alone"])
        print(f"{sec}: reference Spearman (traced spots / all tumour spots): " + "  ".join(f"{k} {v['rho_traced']:+.2f}/{v['rho_all']:+.2f}" for k, v in rs.items()), flush=True)
        print(f"{sec}: held-out genes: " + "  ".join(f"{k} {v:.3f}" for k, v in held.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"E175: {sum(e175)}/{len(e175)}   E176: {sum(e176)}/{len(e176)}", flush=True)
