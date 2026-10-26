"""Uncertainty framework: separate EPISTEMIC (ensemble variance), ALEATORIC (residual-variance model),
and CONFORMAL calibration — and test whether calibration holds UNDER DISTRIBUTION SHIFT (new section).
Question: does the 90% interval still cover about 90% of spots in unseen tissue?
Model: the released estimator (reproduce/estimator.py);
rank targets and the penalty's inner CV use the training sections."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import make
warnings.filterwarnings('ignore'); rng=np.random.RandomState(0)
DATA=P.DESIUM; SP=P.WORK
secs=["BC_525","BC_823","LC_091","LC_170"]; NMET=200
def adj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
from pathlib import Path
from within_benchmark import read_section     # slim h5py reader: log1p expression (float32), MSI, m/z, coordinates
RAW={}; XS={}
for s in secs:
    names,x,raw,_,c=read_section(Path(f"{DATA}/{s}.h5ad")); XS[s]=(names,x); RAW[s]=(np.asarray(raw,np.float32),np.asarray(c,float))
genes=sorted(set.intersection(*[set(XS[s][0]) for s in secs]))
# one ion panel for every section (the stacked training matrix needs the same ion in each column): ions detected
# in >20% of spots in every section, ranked by mean within-section variance.
det_all=np.all([(RAW[s][0]>0).mean(0)>0.2 for s in secs],axis=0)
vmean=np.mean([np.log1p(RAW[s][0]).var(0) for s in secs],axis=0); vmean[~det_all]=-1
top=np.argsort(vmean)[-NMET:]
Xg={}; Y={}; co={}; As={}
for s in secs:
    names,x=XS[s]; pos={g:i for i,g in enumerate(names)}; Xg[s]=x[:,[pos[g] for g in genes]]
    Y[s]=np.log1p(RAW[s][0])[:,top]; co[s]=RAW[s][1]; As[s]=adj(co[s])
del XS
gv=np.vstack([Xg[s] for s in secs]).var(0); hv=np.argsort(gv)[-2000:]
def feat(pca,s): P=pca.transform(Xg[s][:,hv]).astype(np.float32); return np.hstack([P,As[s]@P,As[s]@(As[s]@P)])
held="LC_170"; tr=[s for s in secs if s!=held]
pca=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s][:,hv] for s in tr]))
Ftr={s:feat(pca,s) for s in secs}
# hold out a calibration section-split: use BC_823 spots as in-distribution test + calibration
Fall=np.vstack([Ftr[s] for s in tr]); Yall=np.vstack([Y[s] for s in tr]); Lall=np.concatenate([[s]*len(Y[s]) for s in tr])
idx=rng.permutation(len(Fall)); ncal=len(idx)//5; cal=idx[:ncal]; test_id=idx[ncal:2*ncal]; fit=idx[2*ncal:]  # fit/cal/test_id all disjoint
# ---- ensemble (epistemic) ----
K=10; preds_id=[]; preds_ood=[]
Fid=Fall[cal]; Food=Ftr[held]
for k in range(K):
    bs=rng.randint(0,len(fit),len(fit)); sub=fit[bs]
    reg=make().fit(Fall[sub],Yall[sub],Lall[sub],Lall[sub])
    preds_id.append(reg.predict(Fid)); preds_ood.append(reg.predict(Food))
preds_id=np.array(preds_id); preds_ood=np.array(preds_ood)
epi_id=preds_id.std(0).mean(); epi_ood=preds_ood.std(0).mean()
# ---- point model + conformal + aleatoric ----
reg=make().fit(Fall[fit],Yall[fit],Lall[fit],Lall[fit])
res_cal=np.abs(Yall[cal]-reg.predict(Fall[cal]))                       # calibration residuals
qhat=np.quantile(res_cal,0.9,axis=0)                                   # per-metabolite 90% conformal width
# in-distribution coverage on a held-out split DISJOINT from both fit and cal (no train overlap)
cov_id=float(np.mean(np.abs(Yall[test_id]-reg.predict(Fall[test_id]))<=qhat))
# OOD coverage (held section) with SAME intervals
cov_ood=float(np.mean(np.abs(Y[held]-reg.predict(Food))<=qhat))
# aleatoric: fit residual-variance predictor; check it tracks error
al=Ridge(alpha=50).fit(Fall[fit],np.log1p(np.abs(Yall[fit]-reg.predict(Fall[fit])).mean(1)))
al_pred_id=al.predict(Fid); al_true_id=np.abs(Yall[cal]-reg.predict(Fid)).mean(1)
al_r=spearmanr(al_pred_id,al_true_id)[0]
print("=== UNCERTAINTY DECOMPOSITION ===")
print(f"  epistemic (ensemble std): in-distribution={epi_id:.3f}  OOD(held section)={epi_ood:.3f}  ratio={epi_ood/epi_id:.2f}x")
print(f"  aleatoric model vs true error: Spearman={al_r:+.3f}")
print(f"  conformal 90% coverage: in-distribution={cov_id:.2f}  OOD(held section)={cov_ood:.2f}")
print(f"  -> calibration {'DEGRADES' if cov_ood<0.8 else 'holds'} under shift (nominal 0.90)")
json.dump(dict(epistemic_id=float(epi_id),epistemic_ood=float(epi_ood),epistemic_ratio=float(epi_ood/epi_id),
    aleatoric_spearman=float(al_r),coverage_id=cov_id,coverage_ood=cov_ood,held=held),open(f"{SP}/rigor_uncertainty.json","w"),indent=2)
print("saved rigor_uncertainty.json")
