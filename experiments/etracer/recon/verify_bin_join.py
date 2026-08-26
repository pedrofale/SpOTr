import pandas as pd, numpy as np, collections

ROOT="/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer"
EXP=f"{ROOT}/export/spatial_LE"
LIN=f"{ROOT}/Tumor_Immune_Evasion_eTRACER/LineageTracing/spatial/spatial_lineage_LE.csv"

bc = pd.read_csv(f"{EXP}/barcodes.tsv", header=None, names=["bc"], dtype=str)["bc"].tolist()
print("barcodes n:", len(bc), "unique:", len(set(bc)))

md = pd.read_csv(f"{EXP}/metadata.csv", dtype={0:str})
md = md.rename(columns={md.columns[0]:"bc"})
md["bc"]=md["bc"].astype(str)
print("metadata n:", len(md), "unique bc:", md["bc"].nunique())
print("metadata order == barcodes order:", list(md["bc"])==bc)

# test encoding bc = x*2**32 + y
md["bc_int"]=md["bc"].astype(np.int64)
md["x_dec"]=md["bc_int"]//(2**32)
md["y_dec"]=md["bc_int"]%(2**32)
ok = (md["x_dec"]==md["x"]).all() and (md["y_dec"]==md["y"]).all()
print("bc == x*2**32 + y for ALL rows:", ok)
if not ok:
    bad=md[(md.x_dec!=md.x)|(md.y_dec!=md.y)]
    print("n bad:", len(bad)); print(bad.head())

md["cellid"]=md["x"].astype(str)+"_"+md["y"].astype(str)
print("matrix cellid unique:", md["cellid"].nunique(), "of", len(md))

print("\n-- coordinate ranges (ALL 35835 matrix bins) --")
for c in ["x","y"]:
    v=md[c].values
    print(f"  {c}: min={v.min()} max={v.max()} n_unique={len(np.unique(v))}")
    d=np.diff(np.unique(v)); print(f"     step: min={d.min()} mode={collections.Counter(d).most_common(3)}")

lin = pd.read_csv(LIN, dtype={"CellID":str})
print("\n-- lineage file --")
print("rows:", len(lin), "unique CellID:", lin["CellID"].nunique(),
      "unique lineageGrp:", lin["lineageGrp"].nunique())
# does CellID == x_y from its own cols?
lin["cellid_from_xy"]=lin["x"].astype(str)+"_"+lin["y"].astype(str)
print("CellID == x_y (own cols) all rows:", (lin["CellID"]==lin["cellid_from_xy"]).all())
dups = lin["CellID"][lin["CellID"].duplicated()].tolist()
print("duplicate CellIDs:", len(dups), dups[:5])

mset=set(md["cellid"])
lin["in_matrix"]=lin["cellid_from_xy"].isin(mset)
print("\n-- JOIN lineage -> matrix --")
print("lineage rows resolving to a matrix bin:", int(lin["in_matrix"].sum()), "/", len(lin))
miss=lin[~lin["in_matrix"]]
print("lineage rows FAILING to join:", len(miss))
if len(miss): print(miss.head(20).to_string())

lset=set(lin["cellid_from_xy"])
md["has_lineage"]=md["cellid"].isin(lset)
print("\n-- matrix bins --")
print("total:", len(md))
print("WITH lineage:", int(md["has_lineage"].sum()), f"({100*md['has_lineage'].mean():.2f}%)")
print("WITHOUT lineage:", int((~md["has_lineage"]).sum()), f"({100*(~md['has_lineage']).mean():.2f}%)")

print("\n-- coord range: labelled vs unlabelled --")
for lab,sub in [("labelled",md[md.has_lineage]),("unlabelled",md[~md.has_lineage])]:
    print(f"  {lab:11s} n={len(sub):6d} x[{sub.x.min()},{sub.x.max()}] y[{sub.y.min()},{sub.y.max()}]")

print("\n-- counts sanity --")
print("nCount_RNA: labelled median", md[md.has_lineage].nCount_RNA.median(),
      "| unlabelled median", md[~md.has_lineage].nCount_RNA.median())
md[["bc","cellid","x","y","has_lineage"]].to_csv(f"{ROOT}/recon/LE_bin_index.csv", index=False)
print("\nwrote", f"{ROOT}/recon/LE_bin_index.csv")
