#!/usr/bin/env python
"""Is there ANY spatial structure in the raw allele imbalance itself, before clone calling?
If not, the premise fails at the data level rather than at the discretisation step."""
import json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

HERE="/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa6"
COORDS="/mnt/frisen_nas/shared/shared/20260219_BCSA6_for_pedro/visium/BCSA6_TumA1_visium_CR/outs/spatial/tissue_positions.csv"
K=12; RNG=np.random.default_rng(3); NP=499
ad=pd.read_csv(f"{HERE}/allele_depth_TumA1.tsv",sep="\t")
ad["barcode"]=ad["cell"].str.replace("TumA1_","",regex=False)
tp=pd.read_csv(COORDS); tp=tp[tp.in_tissue==1].set_index("barcode")
sc=pd.read_csv(f"{HERE}/spot_calls_TumA1.csv")
XY=sc[["pxl_row_in_fullres","pxl_col_in_fullres"]].to_numpy(float)
pos={b:i for i,b in enumerate(sc.barcode)}
_,i2=cKDTree(XY).query(XY,k=K+1); NB=i2[:,1:]

def moran(v,mask):
    idx=np.where(mask)[0]
    if len(idx)<100: return None
    xy=XY[idx]; _,ii=cKDTree(xy).query(xy,k=K+1); nb=ii[:,1:]
    z=v[idx]-v[idx].mean()
    if z.std()==0: return None
    I=(z[:,None]*z[nb]).sum()/(K*(z**2).sum())
    nl=[]
    for _ in range(NP):
        p=RNG.permutation(v[idx]); zz=p-p.mean()
        nl.append((zz[:,None]*zz[nb]).sum()/(K*(zz**2).sum()))
    return dict(n=int(len(idx)),moran_I=round(float(I),4),
                p=float((1+np.sum(np.array(nl)>=I))/(NP+1)))

res={}
segs=ad.groupby("seg").agg(n_spots=("barcode","nunique"),med_depth=("total","median"),
                           tot=("total","sum")).sort_values("tot",ascending=False)
for seg in segs.index[:15]:
    s=ad[(ad.seg==seg)&(ad.total>=5)]
    v=np.full(len(sc),np.nan); msk=np.zeros(len(sc),bool)
    for b,mi,to in zip(s.barcode,s.minor,s.total):
        i=pos.get(b)
        if i is not None: v[i]=mi/to; msk[i]=True
    v=np.nan_to_num(v)
    r=moran(v,msk)
    if r: res[seg]=dict(**r,med_depth=float(segs.loc[seg,"med_depth"]),
                        cnv_state=str(s.cnv_state.iloc[0]))
# genome-wide: total minor-allele fraction across all CNV segs per spot
g=ad[ad.total>=1].groupby("barcode").agg(mi=("minor","sum"),to=("total","sum"))
v=np.zeros(len(sc)); msk=np.zeros(len(sc),bool)
for b,mi,to in zip(g.index,g.mi,g.to):
    i=pos.get(b)
    if i is not None: v[i]=mi/to; msk[i]=True
res["GENOME_WIDE_minor_frac"]=moran(v,msk)
# control
res["CONTROL_total_allele_depth"]=moran(sc.depth_tot.to_numpy(float),np.ones(len(sc),bool))
res["CONTROL_pmax"]=moran(sc.pmax_y.to_numpy(float),np.ones(len(sc),bool))
json.dump(res,open(f"{HERE}/raw_allele_space.json","w"),indent=2,default=float)
print(json.dumps(res,indent=2,default=float))
