"""Tumour-restricted allele-only premise test + B1/D1 agreement."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors

NB = Path("/mnt/data/pedro/BCSA/BCSA3/numbat_out")
HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa3")
SPATIAL = {"B1": "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_B1/outs/spatial",
           "D1": "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_D1/outs/spatial"}
K, NPERM = 12, 999
rng = np.random.default_rng(1)

def load_coords(d):
    df = pd.read_csv(Path(d)/"tissue_positions_list.csv", header=None,
        names=["barcode","in_tissue","row","col","pxl_row","pxl_col"]).set_index("barcode")
    return df[["pxl_row","pxl_col"]].astype(float)

def tree_dist(csv):
    e = pd.read_csv(csv); nodes = sorted(set(e["from_clone"])|set(e["to_clone"]))
    idx = {c:i for i,c in enumerate(nodes)}
    D = np.full((len(nodes),len(nodes)), np.inf); np.fill_diagonal(D,0.)
    for _,r in e.iterrows():
        i,j = idx[r["from_clone"]], idx[r["to_clone"]]; D[i,j]=D[j,i]=float(r["n_events"])
    for k in range(len(nodes)): D = np.minimum(D, D[:,[k]]+D[[k],:])
    return np.array(nodes), D

def hamming(cp):
    gt = cp.dropna(subset=["GT_opt"]).groupby("clone_opt")["GT_opt"].agg(lambda s: s.value_counts().index[0])
    clones = sorted(gt.index); sets = {c:set(x for x in str(gt[c]).split(",") if x) for c in clones}
    allev = sorted(set().union(*sets.values()))
    M = np.array([[1 if e in sets[c] else 0 for e in allev] for c in clones])
    return np.array(clones), (M[:,None,:]!=M[None,:,:]).sum(-1).astype(float)

def coh_vec(labels, nbrs, clones):
    return np.array([ (labels[nbrs[labels==c]]==c).mean() - (labels==c).mean() if (labels==c).sum() else np.nan
                      for c in clones])

def mantel(Dt, Ds, nperm=NPERM):
    n = Dt.shape[0]; iu = np.triu_indices(n,1); a,b = Dt[iu], Ds[iu]
    r = np.corrcoef(a,b)[0,1]
    null = np.array([np.corrcoef(Dt[np.ix_(p,p)][iu], b)[0,1] for p in (rng.permutation(n) for _ in range(nperm))])
    return r, (np.sum(np.abs(null)>=abs(r))+1)/(nperm+1)

cp = pd.read_csv(NB/"clone_post_2.tsv", sep="\t")
nb = cp.merge(pd.read_csv(NB/"cell_metadata.csv"), on="cell")
dep = pd.read_csv("/tmp/claude-1005/-home-pedro-projects/fd7cdd18-6dfc-4868-9c2a-fc644da4f924/scratchpad/allele_depth.tsv", sep="\t",
                  header=None, names=["cell","allele_reads","n_segs"]).set_index("cell")
PY = [f"p_y_{i}" for i in range(1,13)]
nodes_t, Dt_full = tree_dist(HERE/"trees"/"bcsa3_B1D1.csv")
nodes_h, Dh_full = hamming(cp)

print("== allele evidence depth per barcode (segments x SNP-covered reads) ==")
for src,pre in [("SS3",None),("Visium_B1","B1"),("Visium_D1","D1")]:
    ids = nb.loc[nb.source==src,"cell"]
    d = dep.reindex(ids).dropna()
    print(f"  {src}: n={len(d)}  allele reads median {d.allele_reads.median():.0f} "
          f"mean {d.allele_reads.mean():.0f} q10 {d.allele_reads.quantile(.1):.0f} q90 {d.allele_reads.quantile(.9):.0f}; "
          f"segments with data median {d.n_segs.median():.0f}")

store = {}
for sec in ["B1","D1"]:
    v = nb[nb.source==f"Visium_{sec}"].copy()
    v["bc"] = v["cell"].str.replace(f"^{sec}_","",regex=True)
    coords = load_coords(SPATIAL[sec]); v = v[v.bc.isin(coords.index)].reset_index(drop=True)
    xy = coords.loc[v.bc].to_numpy()
    P = v[PY].to_numpy(float); P /= P.sum(1,keepdims=True)
    tum = P[:,1:].sum(1)
    print(f"\n{'='*72}\n{sec}  n_spots={len(v)}")

    for tag, mask, cols in [
        ("A. all spots, argmax over all 12 clones", np.ones(len(v),bool), list(range(12))),
        ("B. all spots, argmax over tumour clones 2-12 (renormalised)", np.ones(len(v),bool), list(range(1,12))),
        ("C. confident-tumour spots (tumour mass>0.5), argmax over 2-12", tum>0.5, list(range(1,12))),
    ]:
        sub = P[np.ix_(mask, cols)]
        lab = np.array([cols[i]+1 for i in sub.argmax(1)])
        sxy = xy[mask]
        conf = (sub/sub.sum(1,keepdims=True)).max(1)
        nnb = NearestNeighbors(n_neighbors=K+1).fit(sxy)
        nbrs = nnb.kneighbors(sxy, return_distance=False)[:,1:]
        clones = np.array(sorted(set(lab)))
        obs = coh_vec(lab, nbrs, clones)
        nullm = np.empty((NPERM,len(clones)))
        for i in range(NPERM):
            nullm[i] = coh_vec(lab[rng.permutation(len(lab))], nbrs, clones)
        cnt = pd.Series(lab).value_counts()
        df = pd.DataFrame(dict(clone=clones, n=[cnt[c] for c in clones],
                               share=[(lab==c).mean() for c in clones], coherence=obs,
                               coh_norm=[o/(1-(lab==c).mean()) for o,c in zip(obs,clones)],
                               null_sd=nullm.std(0),
                               p=[(np.sum(nullm[:,j]>=obs[j])+1)/(NPERM+1) for j in range(len(clones))]))
        big = df[df.n>=20]
        print(f"\n--- {sec} / {tag}   (n={mask.sum()}, mean argmax conf {conf.mean():.3f})")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"    mean coherence (clones>=20 spots, n={len(big)}) = {big.coherence.mean():.4f}"
              f" | normalised {big.coh_norm.mean():.4f} | sig p<0.05: {int((big.p<0.05).sum())}/{len(big)}")
        # Mantel on this labelling
        for dn, nd, DD in [("mut_graph", nodes_t, Dt_full), ("Hamming", nodes_h, Dh_full)]:
            pos = {c:i for i,c in enumerate(nd)}
            cl = [c for c in clones if cnt[c]>=20 and c in pos]
            if len(cl)<4: print(f"    Mantel[{dn}]: only {len(cl)} clones -- skipped"); continue
            cent = np.stack([sxy[lab==c].mean(0) for c in cl])
            Ds = np.linalg.norm(cent[:,None]-cent[None,:],axis=-1)
            Dtt = DD[np.ix_([pos[c] for c in cl],[pos[c] for c in cl])]
            r,p = mantel(Dtt,Ds)
            print(f"    Mantel[{dn}] {len(cl)} clones {list(map(int,cl))}: r={r:+.4f} p={p:.4f}")
            store[(sec,tag[0],dn)] = (r,p,list(map(int,cl)))
        store[(sec,tag[0],"coh")] = df

# ---- B1 vs D1 agreement on shared clones
print(f"\n{'='*72}\nB1 vs D1 AGREEMENT")
for variant in ["A","B","C"]:
    a,b = store[("B1",variant,"coh")], store[("D1",variant,"coh")]
    m = a.merge(b, on="clone", suffixes=("_B1","_D1"))
    m = m[(m.n_B1>=20)&(m.n_D1>=20)]
    if len(m)>=3:
        r = np.corrcoef(m.coherence_B1, m.coherence_D1)[0,1]
        print(f"  variant {variant}: {len(m)} shared clones (>=20 spots both); "
              f"per-clone coherence corr r={r:+.3f}; "
              f"same sign {int((np.sign(m.coherence_B1)==np.sign(m.coherence_D1)).sum())}/{len(m)}")
        print(m[["clone","n_B1","coherence_B1","p_B1","n_D1","coherence_D1","p_D1"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    for dn in ["mut_graph","Hamming"]:
        ka,kb = store.get(("B1",variant,dn)), store.get(("D1",variant,dn))
        if ka and kb:
            print(f"  variant {variant} Mantel[{dn}]: B1 r={ka[0]:+.4f} (p={ka[1]:.3f}) | "
                  f"D1 r={kb[0]:+.4f} (p={kb[1]:.3f}) | sign {'AGREE' if np.sign(ka[0])==np.sign(kb[0]) else 'FLIP'}")
