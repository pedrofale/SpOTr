#!/usr/bin/env python
"""BCSA6 robustness: confidence subsets, soft autocorrelation, positive controls."""
import json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

NUMBAT="/mnt/data/pedro/BCSA/BCSA6/numbat_out"
COORDS="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro/visium/BCSA6_TumA1_visium_CR/outs/spatial/tissue_positions.csv"
HERE="/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa6"
K=12; NPERM=999; RNG=np.random.default_rng(1)
out={}

cp=pd.read_csv(f"{NUMBAT}/clone_post_2.tsv",sep="\t")
nC=sum(c.startswith("p_y_") for c in cp.columns)
yc=[f"p_y_{i}" for i in range(1,nC+1)]

# SS3 side, allele-only, for contrast
ss=cp[~cp["cell"].str.startswith("TumA1_")]
Ps=ss[yc].to_numpy(float); Ps=np.where(np.isfinite(Ps),Ps,0); Ps=Ps/np.maximum(Ps.sum(1,keepdims=True),1e-300)
out["ss3_n"]=int(len(ss))
out["ss3_alleleonly_clone_shares"]=pd.Series(np.argmax(Ps,1)+1).value_counts(normalize=True).round(4).to_dict()
out["ss3_median_pmax_y"]=float(np.median(Ps.max(1)))
out["spot_vs_ss3_clone1_share"]=None

sp=cp[cp["cell"].str.startswith("TumA1_")].copy()
sp["barcode"]=sp["cell"].str.replace("TumA1_","",regex=False)
P=sp[yc].to_numpy(float); P=np.where(np.isfinite(P),P,0); P=P/np.maximum(P.sum(1,keepdims=True),1e-300)
sp["clone_y"]=np.argmax(P,1)+1; sp["pmax_y"]=P.max(1)

tp=pd.read_csv(COORDS); tp=tp[tp.in_tissue==1]
m=sp.merge(tp,on="barcode",how="inner")
Pm=P[sp["barcode"].isin(set(m.barcode)).to_numpy()]
order=m["barcode"].tolist()
Pm=pd.DataFrame(P,index=sp["barcode"]).loc[order].to_numpy()

ad=pd.read_csv(f"{HERE}/allele_depth_TumA1.tsv",sep="\t")
ad["barcode"]=ad["cell"].str.replace("TumA1_","",regex=False)
dep=ad[ad.total>0].groupby("barcode")["total"].sum()
m["depth_tot"]=m["barcode"].map(dep).fillna(0).to_numpy()

xy=m[["pxl_row_in_fullres","pxl_col_in_fullres"]].to_numpy(float)
tree=cKDTree(xy); d,idx=tree.query(xy,k=K+1); nbr=idx[:,1:]

def excess(labels,sub=None):
    N=len(labels); nb=labels[nbr] if sub is None else None
    match=(labels[nbr]==labels[:,None]).mean(1)
    res={}
    for c in np.unique(labels):
        s=labels==c; res[int(c)]=float(match[s].mean()-s.sum()/N)
    return res

def perm_test(labels,minn=20):
    obs=excess(labels); cnt=pd.Series(labels).value_counts()
    keep=[c for c in obs if cnt[c]>=minn]
    if not keep: return None
    om=float(np.mean([obs[c] for c in keep]))
    null=np.empty(NPERM)
    for b in range(NPERM):
        r=excess(RNG.permutation(labels)); null[b]=np.mean([r[c] for c in keep if c in r])
    return dict(n_spots=int(len(labels)),n_clones_used=len(keep),mean_excess=om,
                p=float((1+np.sum(null>=om))/(NPERM+1)),
                null_mean=float(null.mean()),null_sd=float(null.std()),
                per_clone={int(c):round(obs[c],4) for c in keep})

# --- A. confidence subsets
lab=m["clone_y"].to_numpy()
out["all_spots"]=perm_test(lab)
# confidence subsets re-derive the kNN graph on the retained spots only:
def perm_test_sub(mask,minn=20):
    sub=np.where(mask)[0]; pos={g:i for i,g in enumerate(sub)}
    xy2=xy[sub]; t2=cKDTree(xy2); kk=min(K,len(sub)-1)
    _,i2=t2.query(xy2,k=kk+1); nb2=i2[:,1:]
    L=lab[sub]; N=len(L)
    def ex(l):
        mt=(l[nb2]==l[:,None]).mean(1); r={}
        for c in np.unique(l):
            s=l==c; r[int(c)]=float(mt[s].mean()-s.sum()/N)
        return r
    obs=ex(L); cnt=pd.Series(L).value_counts(); keep=[c for c in obs if cnt[c]>=minn]
    if not keep: return None
    om=float(np.mean([obs[c] for c in keep])); null=np.empty(NPERM)
    for b in range(NPERM):
        r=ex(RNG.permutation(L)); null[b]=np.mean([r[c] for c in keep if c in r])
    return dict(n_spots=N,n_clones_used=len(keep),mean_excess=om,
                p=float((1+np.sum(null>=om))/(NPERM+1)),
                per_clone={int(c):round(obs[c],4) for c in keep})
for t in (0.6,0.8,0.9):
    out[f"pmax_gt_{t}"]=perm_test_sub((m.pmax_y>t).to_numpy())

# --- B. malignant vs normal (coarsest possible lineage signal)
binlab=np.where(lab==1,0,1)
obs=excess(binlab); null=np.empty(NPERM)
om=float(np.mean([obs[0],obs[1]]))
for b in range(NPERM):
    r=excess(RNG.permutation(binlab)); null[b]=np.mean([r[0],r[1]])
out["binary_normal_vs_malignant"]=dict(mean_excess=om,per_class=obs,
    p=float((1+np.sum(null>=om))/(NPERM+1)),null_sd=float(null.std()))

# --- C. soft (posterior) spatial autocorrelation: Moran's I of each p_y column
def moran(v):
    v=np.asarray(v,float); z=v-v.mean()
    if z.std()==0: return np.nan
    return float((z[:,None]*z[nbr]).sum()/(K*(z**2).sum()))
def moran_p(v,n=NPERM):
    o=moran(v)
    if not np.isfinite(o): return o,np.nan
    nl=np.array([moran(RNG.permutation(v)) for _ in range(n)])
    return o,float((1+np.sum(nl>=o))/(n+1))
mo={}
for i in range(nC):
    if Pm[:,i].std()>0:
        I,p=moran_p(Pm[:,i],199)
        mo[f"p_y_{i+1}"]=dict(moran_I=round(I,4),p=p,mean_post=round(float(Pm[:,i].mean()),4))
out["moran_posterior_columns"]=mo

# --- D. positive controls: is the machinery able to see real spatial structure?
I,p=moran_p(m["depth_tot"].to_numpy(),199)
out["control_moran_allele_depth"]=dict(moran_I=round(I,4),p=p)
I,p=moran_p(m["pmax_y"].to_numpy(),199)
out["control_moran_pmax_y"]=dict(moran_I=round(I,4),p=p)
I,p=moran_p(m["array_row"].to_numpy(),199)
out["control_moran_array_row_should_be_high"]=dict(moran_I=round(I,4),p=p)
if "clone_opt" in m.columns:
    jl=m["clone_opt"].to_numpy()
    out["JOINT_call_for_contrast_only"]=perm_test(jl)

with open(f"{HERE}/robustness_bcsa6.json","w") as f: json.dump(out,f,indent=2,default=float)
print(json.dumps(out,indent=2,default=float))
