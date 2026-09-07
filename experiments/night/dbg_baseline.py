import os,sys
os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, ot, time
sys.path.insert(0,"../../src")
from spotr.genfgw import GenFGW, standardise
d=np.load("../petracer/petracerFAIR_M3-4.npz",allow_pickle=True)
M=ot.dist(np.asarray(d["X_sc"],float),np.asarray(d["X_st"],float)); M/=M.max()
CT=np.asarray(d["C_tree"],float); CS=np.asarray(d["C_space"],float)
xy_sc=np.asarray(d["xy_sc"],float); xy_st=np.asarray(d["xy"],float)
n,m=M.shape; p,q=np.ones(n)/n,np.ones(m)/m
C1=CT/CT.mean(); C2=CS/CS.mean()
def err(am): return float(np.median(np.linalg.norm(xy_st[am]-xy_sc,axis=1)))
t0=time.time()
Tp=np.asarray(ot.gromov.fused_gromov_wasserstein(M,C1,C2,p,q,loss_fun="square_loss",alpha=0.75,
     max_iter=200,tol_rel=1e-9,tol_abs=1e-9))
print(f"POT FGW mean-normalised alpha=0.75 : {err(Tp.argmax(1)):.1f} um   ({time.time()-t0:.0f}s)")
iu=np.triu_indices(n,1)
s1=C1[iu].std(); s2=CS[np.triu_indices(m,1)].std()/CS.mean()
P1=standardise(C1); P2=standardise(C2)
w_old=0.75/0.25
w_new=w_old*s1*s2
for dt,tag in [(np.float64,"f64"),(np.float32,"f32")]:
    t0=time.time()
    Tg,info=GenFGW(M,[P1],[P2],[1.0],w_new/(1+w_new),dtype=dt).solve(p,q,max_iter=200,tol=1e-9)
    Tg=np.asarray(Tg)
    print(f"GenFGW identity {tag}  w_new={w_new:.4f} alpha={w_new/(1+w_new):.4f} : "
          f"{err(Tg.argmax(1)):.1f} um  iters {info['n_iter']}  max|dT| {np.abs(Tg-Tp).max():.2e} ({time.time()-t0:.0f}s)")
