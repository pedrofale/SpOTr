#!/usr/bin/env python
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE="/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa6"
NUMBAT="/mnt/data/pedro/BCSA/BCSA6/numbat_out"
sc=pd.read_csv(f"{HERE}/spot_calls_TumA1.csv")
coh=pd.read_csv(f"{HERE}/coherence_TumA1.csv")
wt=json.load(open(f"{HERE}/within_tumour_bcsa6.json"))
res=json.load(open(f"{HERE}/results_bcsa6.json"))
ad=pd.read_csv(f"{HERE}/allele_depth_TumA1.tsv",sep="\t")
cp=pd.read_csv(f"{NUMBAT}/clone_post_2.tsv",sep="\t")
nC=19; yc=[f"p_y_{i}" for i in range(1,nC+1)]
ss=cp[~cp["cell"].str.startswith("TumA1_")][yc].to_numpy(float)
ss=ss/np.maximum(ss.sum(1,keepdims=True),1e-300)

fig,ax=plt.subplots(2,3,figsize=(17,10))
# 1 map
top=sc.clone_y.value_counts().index[:9]
cm=plt.get_cmap("tab10")
a=ax[0,0]
for i,c in enumerate(top):
    s=sc[sc.clone_y==c]
    a.scatter(s.pxl_col_in_fullres,-s.pxl_row_in_fullres,s=5,color=cm(i),
              label=f"clone {c} (n={len(s)})")
oth=sc[~sc.clone_y.isin(top)]
a.scatter(oth.pxl_col_in_fullres,-oth.pxl_row_in_fullres,s=5,color="0.8",label=f"other (n={len(oth)})")
a.set_title("BCSA6 TumA1 — allele-only clone call (argmax p_y)\n3427 spots, 19 clones")
a.legend(fontsize=6,markerscale=2,loc="upper right"); a.set_aspect("equal"); a.axis("off")

# 2 posterior confidence
a=ax[0,1]
a.hist(sc.pmax_y,bins=50,color="#4477aa",alpha=.85,density=True,label="Visium spots (n=3427)")
a.hist(ss.max(1),bins=50,color="#cc6677",alpha=.55,density=True,label="SS3 cells (n=1748)")
a.axvline(1/19,ls=":",c="k"); a.text(1/19,a.get_ylim()[1]*.9," chance 1/19",fontsize=7)
a.set_xlabel("max allele-only posterior  max_c p_y(c)"); a.set_ylabel("density")
a.set_title(f"Posterior confidence\nspot median {sc.pmax_y.median():.2f} vs SS3 median {np.median(ss.max(1)):.2f}")
a.legend(fontsize=8)

# 3 depth
a=ax[0,2]
a.hist(np.clip(ad.total,0,40),bins=40,color="#228833",alpha=.85)
a.axvline(np.median(ad.total),c="k",ls="--")
a.set_xlabel("allele reads per spot x segment (clipped at 40)")
a.set_title(f"Allele read depth\nmedian {np.median(ad.total):.0f} reads/spot/seg; "
            f"{res['frac_spot_seg_pairs_depth_lt5']*100:.0f}% below 5")

# 4 per-clone coherence
a=ax[1,0]
u=coh[coh.used].sort_values("n_spots",ascending=False)
x=np.arange(len(u))
a.bar(x,u.excess_coherence,color=["#cc3311" if p<.05 else "#4477aa" for p in u.p_perm])
a.errorbar(x,u.null_mean,yerr=1.96*u.null_sd,fmt="none",ecolor="0.4",capsize=3)
a.axhline(0,c="k",lw=.8)
a.axhline(0.309,c="#117733",ls="--",lw=1); a.text(len(x)-1,0.309,"eTRACER late escape 0.309",ha="right",va="bottom",fontsize=7,color="#117733")
a.axhline(0.205,c="#999933",ls="--",lw=1); a.text(len(x)-1,0.205,"eTRACER baseline 0.205",ha="right",va="bottom",fontsize=7,color="#999933")
a.set_xticks(x); a.set_xticklabels([f"{c}\nn={n}" for c,n in zip(u.clone,u.n_spots)],fontsize=7)
a.set_ylabel("excess coherence (k=12)")
a.set_title(f"Per-clone spatial coherence vs 999-perm null\nmean over 11 clones = "
            f"{res['coherence_mean_over_clones_ge20']:.4f} (p={res['coherence_p_global']:.3f})")

# 5 mantel scatter
a=ax[1,1]
D=pd.read_csv(f"{HERE}/clone_tree_dist.csv",index_col=0); D.columns=D.columns.astype(int)
cent=pd.read_csv(f"{HERE}/clone_centroids.csv",index_col=0)
keep=sorted(coh[coh.used].clone.tolist())
C=cent.loc[keep].to_numpy()
S=np.sqrt(((C[:,None,:]-C[None,:,:])**2).sum(-1))
T=D.loc[keep,keep].to_numpy(); iu=np.triu_indices(len(keep),1)
a.scatter(T[iu],S[iu],s=25,color="#332288")
mt=res["mantel"][0]
a.set_xlabel("clone-tree distance (CNV events on path)")
a.set_ylabel("distance between clone spatial centroids (px)")
a.set_title(f"Clone tree vs space (Mantel, 999 perm)\nPearson r={mt['pearson_r']:.3f} p={mt['p_pearson']:.3f}; "
            f"Spearman r={mt['spearman_r']:.3f} p={mt['p_spearman']:.3f}")

# 6 nested tests
a=ax[1,2]
labels=["all spots\n(11 clones)","malignant only\n(10 clones)","beyond trunk\n(9 clones)",
        "malignant &\npmax>0.6"]
keys=["ALL_spots","MALIGNANT_only","MALIGNANT_beyond_trunk","MALIGNANT_only_pmax_gt_0.6"]
vals=[wt[k]["mean_excess"] for k in keys]; ps=[wt[k]["p"] for k in keys]
sds=[wt[k]["null_sd"] for k in keys]
a.bar(np.arange(4),vals,color=["#cc3311" if p<.05 else "#4477aa" for p in ps])
a.errorbar(np.arange(4),[0]*4,yerr=[1.96*s for s in sds],fmt="none",ecolor="0.4",capsize=4)
for i,(v,p) in enumerate(zip(vals,ps)): a.text(i,v+0.001,f"p={p:.3f}",ha="center",fontsize=8)
a.axhline(0,c="k",lw=.8)
a.axhline(0.205,c="#999933",ls="--",lw=1)
a.text(3.4,0.205,"eTRACER baseline",ha="right",va="bottom",fontsize=7,color="#999933")
a.set_xticks(np.arange(4)); a.set_xticklabels(labels,fontsize=8)
a.set_ylabel("mean excess coherence"); a.set_ylim(-0.005,0.22)
a.set_title("Coherence survives nothing once the\nnormal/trunk contrast is removed")
plt.tight_layout(); plt.savefig(f"{HERE}/bcsa6_premise.png",dpi=140)
print("wrote",f"{HERE}/bcsa6_premise.png")
