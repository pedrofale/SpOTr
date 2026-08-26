"""Robustness: ceilings, confidence stratification, soft posterior-weighted Mantel."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
NB=Path("/mnt/data/pedro/BCSA/BCSA3/numbat_out"); HERE=Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa3")
SP={"B1":"/mnt/data/pedro/BCSA/BCSA3/V19T26-031_B1/outs/spatial","D1":"/mnt/data/pedro/BCSA/BCSA3/V19T26-031_D1/outs/spatial"}
K,NPERM=12,999; rng=np.random.default_rng(2)
def coords(d):
    df=pd.read_csv(Path(d)/"tissue_positions_list.csv",header=None,names=["barcode","in_tissue","row","col","pxl_row","pxl_col"]).set_index("barcode")
    return df[["pxl_row","pxl_col"]].astype(float)
def tdist(csv):
    e=pd.read_csv(csv); nodes=sorted(set(e.from_clone)|set(e.to_clone)); idx={c:i for i,c in enumerate(nodes)}
    D=np.full((len(nodes),)*2,np.inf); np.fill_diagonal(D,0.)
    for _,r in e.iterrows():
        i,j=idx[r.from_clone],idx[r.to_clone]; D[i,j]=D[j,i]=float(r.n_events)
    for k in range(len(nodes)): D=np.minimum(D,D[:,[k]]+D[[k],:])
    return np.array(nodes),D
def mantel(Dt,Ds,nperm=NPERM):
    n=Dt.shape[0]; iu=np.triu_indices(n,1); b=Dt[iu],Ds[iu]
    r=np.corrcoef(Dt[iu],Ds[iu])[0,1]
    null=np.array([np.corrcoef(Dt[np.ix_(p,p)][iu],Ds[iu])[0,1] for p in (rng.permutation(n) for _ in range(nperm))])
    return r,(np.sum(np.abs(null)>=abs(r))+1)/(nperm+1)
def cohv(lab,nbrs,cl): return np.array([(lab[nbrs[lab==c]]==c).mean()-(lab==c).mean() for c in cl])
cp=pd.read_csv(NB/"clone_post_2.tsv",sep="\t"); nb=cp.merge(pd.read_csv(NB/"cell_metadata.csv"),on="cell")
PY=[f"p_y_{i}" for i in range(1,13)]; nodes,DT=tdist(HERE/"trees"/"bcsa3_B1D1.csv")
for sec in ["B1","D1"]:
    v=nb[nb.source==f"Visium_{sec}"].copy(); v["bc"]=v.cell.str.replace(f"^{sec}_","",regex=True)
    cd=coords(SP[sec]); v=v[v.bc.isin(cd.index)].reset_index(drop=True); xy=cd.loc[v.bc].to_numpy()
    P=v[PY].to_numpy(float); P/=P.sum(1,keepdims=True); conf=P.max(1); lab=P.argmax(1)+1
    print(f"\n=== {sec} ===")
    # ceiling for the excess-over-chance statistic given observed shares
    for tag,L in [("argmax over 12",lab),("argmax over 2-12",np.array([i+2 for i in P[:,1:].argmax(1)]))]:
        cl=[c for c in sorted(set(L)) if (L==c).sum()>=20]
        ceil=np.mean([1-(L==c).mean() for c in cl])
        print(f"  {tag}: mean ceiling of excess-over-chance across {len(cl)} clones = {ceil:.3f} "
              f"(so 0.3 is attainable in principle)")
    # confidence stratification: does restricting to confident calls recover territory?
    nnb=NearestNeighbors(n_neighbors=K+1).fit(xy); nbrs=nnb.kneighbors(xy,return_distance=False)[:,1:]
    print("  confidence stratification (argmax over 12, k-NN computed within each stratum):")
    for lo,hi in [(0,.5),(.5,.8),(.8,.95),(.95,1.01)]:
        m=(conf>=lo)&(conf<hi)
        if m.sum()<60: continue
        sxy=xy[m]; sl=lab[m]
        nn2=NearestNeighbors(n_neighbors=min(K,len(sl)-1)+1).fit(sxy); nb2=nn2.kneighbors(sxy,return_distance=False)[:,1:]
        cl=[c for c in sorted(set(sl)) if (sl==c).sum()>=20]
        if not cl: continue
        o=cohv(sl,nb2,cl)
        print(f"    maxp [{lo:.2f},{hi:.2f}): n={m.sum():5d}  clones>=20: {len(cl):2d}  mean coherence {o.mean():+.4f}")
    # SOFT Mantel: posterior-weighted clone centroids, no argmax
    W=P/P.sum(0,keepdims=True)
    cent=(W.T@xy)
    keep=[i for i in range(12) if P[:,i].sum()>=20]
    Ds=np.linalg.norm(cent[keep][:,None]-cent[keep][None,:],axis=-1)
    Dt=DT[np.ix_(keep,keep)]
    r,p=mantel(Dt,Ds); print(f"  SOFT posterior-weighted centroid Mantel ({len(keep)} clones): r={r:+.4f} p={p:.4f}")
    keep2=[i for i in keep if i!=0]
    Ds2=np.linalg.norm(cent[keep2][:,None]-cent[keep2][None,:],axis=-1); Dt2=DT[np.ix_(keep2,keep2)]
    r2,p2=mantel(Dt2,Ds2); print(f"  SOFT, tumour clones only ({len(keep2)}): r={r2:+.4f} p={p2:.4f}")
