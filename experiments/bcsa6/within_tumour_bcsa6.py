#!/usr/bin/env python
"""The decisive test: within malignant spots only, does sub-clone identity predict space?
Plus spot-level tree-vs-space Mantel (the exact SpOTr GW premise)."""
import json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

NUMBAT="/mnt/data/pedro/BCSA/BCSA6/numbat_out"
COORDS="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro/visium/BCSA6_TumA1_visium_CR/outs/spatial/tissue_positions.csv"
HERE="/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa6"
K=12; NPERM=999; RNG=np.random.default_rng(2); out={}

cp=pd.read_csv(f"{NUMBAT}/clone_post_2.tsv",sep="\t")
nC=sum(c.startswith("p_y_") for c in cp.columns); yc=[f"p_y_{i}" for i in range(1,nC+1)]
sp=cp[cp["cell"].str.startswith("TumA1_")].copy()
sp["barcode"]=sp["cell"].str.replace("TumA1_","",regex=False)
P=sp[yc].to_numpy(float); P=np.where(np.isfinite(P),P,0); P=P/np.maximum(P.sum(1,keepdims=True),1e-300)
sp["clone_y"]=np.argmax(P,1)+1; sp["pmax_y"]=P.max(1)
tp=pd.read_csv(COORDS); tp=tp[tp.in_tissue==1]
m=sp.merge(tp,on="barcode",how="inner").reset_index(drop=True)
Pm=pd.DataFrame(P,index=sp["barcode"].to_numpy()).loc[m["barcode"].to_numpy()].to_numpy()
XY=m[["pxl_row_in_fullres","pxl_col_in_fullres"]].to_numpy(float)

ad=pd.read_csv(f"{HERE}/allele_depth_TumA1.tsv",sep="\t")
ad["barcode"]=ad["cell"].str.replace("TumA1_","",regex=False)
m["depth_tot"]=m["barcode"].map(ad[ad.total>0].groupby("barcode")["total"].sum()).fillna(0)

# tree distances between clones
edges=pd.read_csv(f"{HERE}/bcsa6_clone_edges_full.csv")
nodes=pd.read_csv(f"{HERE}/bcsa6_clone_nodes.csv"); nodes["gt"]=nodes["gt"].fillna("")
INF=1e9; D=np.full((nC+1,nC+1),INF); np.fill_diagonal(D,0)
for r in edges.itertuples():
    D[int(r.from_clone),int(r.to_clone)]=D[int(r.to_clone),int(r.from_clone)]=float(r.n_events)
for k in range(1,nC+1): D=np.minimum(D,D[:,[k]]+D[[k],:])
depth_from_root=D[1,:].copy()               # events accumulated since normal root
nevents={int(r.clone):len(set(str(r.gt).split(","))-{"","nan"}) for r in nodes.itertuples()}

def run(mask,tag,minn=20):
    sub=np.where(mask)[0]; xy=XY[sub]; L=m["clone_y"].to_numpy()[sub]; N=len(L)
    kk=min(K,N-1); _,i2=cKDTree(xy).query(xy,k=kk+1); nb=i2[:,1:]
    def ex(l):
        mt=(l[nb]==l[:,None]).mean(1); r={}
        for c in np.unique(l):
            s=l==c; r[int(c)]=float(mt[s].mean()-s.sum()/N)
        return r
    obs=ex(L); cnt=pd.Series(L).value_counts(); keep=sorted([c for c in obs if cnt[c]>=minn])
    res=dict(tag=tag,n_spots=int(N),clones_ge20=keep)
    if keep:
        om=float(np.mean([obs[c] for c in keep])); null=np.empty(NPERM)
        percl={}
        pn={c:np.empty(NPERM) for c in keep}
        for b in range(NPERM):
            r=ex(RNG.permutation(L)); null[b]=np.mean([r[c] for c in keep if c in r])
            for c in keep: pn[c][b]=r.get(c,np.nan)
        for c in keep:
            v=pn[c][np.isfinite(pn[c])]
            percl[int(c)]=dict(n=int(cnt[c]),excess=round(obs[c],4),
                               p=float((1+np.sum(v>=obs[c]))/(len(v)+1)))
        res.update(mean_excess=om,p=float((1+np.sum(null>=om))/(NPERM+1)),
                   null_sd=float(null.std()),per_clone=percl)
        # clone-centroid Mantel inside this subset
        if len(keep)>=4:
            C=pd.DataFrame(xy,columns=["r","c"]).assign(cl=L).groupby("cl").mean().loc[keep].to_numpy()
            S=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1))
            T=D[np.ix_(keep,keep)]; iu=np.triu_indices(len(keep),1)
            rp=pearsonr(T[iu],S[iu])[0]; rs=spearmanr(T[iu],S[iu])[0]
            cp_=cs=0
            for _ in range(NPERM):
                o=RNG.permutation(len(keep)); Tp=T[np.ix_(o,o)][iu]
                if abs(pearsonr(Tp,S[iu])[0])>=abs(rp): cp_+=1
                if abs(spearmanr(Tp,S[iu])[0])>=abs(rs): cs+=1
            res["mantel_centroid"]=dict(pearson_r=float(rp),p=float((1+cp_)/(NPERM+1)),
                                        spearman_r=float(rs),p_spearman=float((1+cs)/(NPERM+1)))
    return res

lab=m["clone_y"].to_numpy()
out["ALL_spots"]=run(np.ones(len(m),bool),"all")
out["MALIGNANT_only"]=run(lab!=1,"clone_y != 1 (drop normal root)")
out["MALIGNANT_beyond_trunk"]=run(~np.isin(lab,[1,2]),"drop root + trunk clone 2")
out["MALIGNANT_only_pmax_gt_0.6"]=run((lab!=1)&(m.pmax_y.to_numpy()>0.6),"malignant, confident")

# --- spot-level tree x space Mantel: the literal SpOTr GW premise
def spot_mantel(mask,tag,nsub=1200):
    idx=np.where(mask)[0]
    if len(idx)>nsub: idx=RNG.choice(idx,nsub,replace=False)
    L=lab[idx]; xy=XY[idx]
    T=D[np.ix_(L,L)]; S=np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(-1))
    iu=np.triu_indices(len(idx),1); t,s=T[iu],S[iu]
    rp=pearsonr(t,s)[0]; rs=spearmanr(t,s)[0]
    c1=c2=0
    for _ in range(199):
        o=RNG.permutation(len(idx)); Tp=T[np.ix_(o,o)][iu]
        if abs(pearsonr(Tp,s)[0])>=abs(rp): c1+=1
        if abs(spearmanr(Tp,s)[0])>=abs(rs): c2+=1
    return dict(tag=tag,n_spots=int(len(idx)),pearson_r=float(rp),p=float((1+c1)/200),
                spearman_r=float(rs),p_spearman=float((1+c2)/200))
out["spot_level_mantel_all"]=spot_mantel(np.ones(len(m),bool),"all spots")
out["spot_level_mantel_malignant"]=spot_mantel(lab!=1,"malignant spots")

# --- continuous lineage depth vs space
_,i2=cKDTree(XY).query(XY,k=K+1); NB=i2[:,1:]
def moran(v,nb=NB,n=499):
    v=np.asarray(v,float); z=v-v.mean()
    if z.std()==0: return np.nan,np.nan
    I=(z[:,None]*z[nb]).sum()/(nb.shape[1]*(z**2).sum())
    nl=[]
    for _ in range(n):
        p=RNG.permutation(v); zz=p-p.mean()
        nl.append((zz[:,None]*zz[nb]).sum()/(nb.shape[1]*(zz**2).sum()))
    return float(I),float((1+np.sum(np.array(nl)>=I))/(n+1))
hard_depth=depth_from_root[lab]
soft_depth=Pm@depth_from_root[1:nC+1]
I,p=moran(hard_depth); out["moran_tree_depth_hard"]=dict(moran_I=round(I,4),p=p)
I,p=moran(soft_depth); out["moran_tree_depth_soft_posterior"]=dict(moran_I=round(I,4),p=p)
mal=lab!=1
_,i3=cKDTree(XY[mal]).query(XY[mal],k=K+1)
I,p=moran(hard_depth[mal],i3[:,1:]); out["moran_tree_depth_hard_malignant_only"]=dict(moran_I=round(I,4),p=p)
I,p=moran(m["depth_tot"].to_numpy(),NB); out["control_moran_allele_depth"]=dict(moran_I=round(I,4),p=p)

# --- why so many root calls?
out["clone1_vs_depth"]=dict(
    median_depth_clone1=float(m.loc[lab==1,"depth_tot"].median()),
    median_depth_other=float(m.loc[lab!=1,"depth_tot"].median()),
    spearman_depth_vs_pmax=float(spearmanr(m["depth_tot"],m["pmax_y"])[0]),
    spearman_depth_vs_treedepth=float(spearmanr(m["depth_tot"],hard_depth)[0]),
    frac_spots_called_root=float((lab==1).mean()))
q=pd.qcut(m["depth_tot"],5,labels=False,duplicates="drop")
out["by_depth_quintile"]={int(k):dict(n=int((q==k).sum()),
    frac_root=float((lab[q==k]==1).mean()),
    median_pmax=float(m.loc[q==k,"pmax_y"].median()),
    mean_tree_depth=float(hard_depth[q==k].mean())) for k in sorted(pd.unique(q.dropna()))}

with open(f"{HERE}/within_tumour_bcsa6.json","w") as f: json.dump(out,f,indent=2,default=float)
print(json.dumps(out,indent=2,default=float))
