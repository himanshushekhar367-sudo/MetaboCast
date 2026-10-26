"""Broaden benchmark beyond correlation methods (kNN, SpaGE-style cross-modal imputation, MLP);
leave-region-out generalization using pathology labels; accuracy vs metabolite abundance.
MetaboCast and the no-diffusion model use the released estimator (reproduce/estimator.py); the kNN,
SpaGE-style and MLP baselines are fitted to log intensities."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import anndata as ad, numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor; from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import fit_predict
warnings.filterwarnings('ignore')
DATA=P.DESIUM; SP=P.WORK
def inner(co_tr): return KMeans(5,n_init=3,random_state=0).fit_predict(co_tr)
def fp(F,Y,tr,te,g):
    out=np.zeros_like(Y); out[te]=fit_predict(F[tr],Y[tr],F[te],g); return out
secs=["BC_525","BC_823","LC_091"]; NMET=1000
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def medS(pred,Y,te): return float(np.nanmedian([spearmanr(pred[te,j],Y[te,j])[0] for j in range(Y.shape[1]) if Y[te,j].std()>1e-9]))
def load(s):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Y=np.log1p(Yr); co=np.asarray(a.obsm['spatial'],float)
    det=(Yr>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-NMET:]]
    reg=None
    for k in ['Type','region','pathology','annotation']:
        if k in a.obs: reg=a.obs[k].astype(str).values; break
    hv=np.argsort(X.var(0))[-3000:]; return X[:,hv],Y[:,top],Yr[:,top],co,reg
DAT={s:load(s) for s in secs}
methods=["MetaboCast","no diffusion","gene-kNN","SpaGE-style (PC-kNN)","MLP (diffusion features)"]
agg={m:[] for m in methods}
for s in secs:
    Xh,Yd,Yraw,co,_=DAT[s]; A=nadj(co); blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)
    for fb in range(3):
        tr=blk!=fb; te=blk==fb; fit=np.where(tr)[0]
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[fit]).transform(Xh).astype(np.float32)
        T=np.hstack([P,A@P,A@(A@P)])
        g=inner(co[tr])
        agg["MetaboCast"].append(medS(fp(T,Yd,tr,te,g),Yd,te))
        agg["no diffusion"].append(medS(fp(P,Yd,tr,te,g),Yd,te))
        agg["gene-kNN"].append(medS(KNeighborsRegressor(30).fit(Xh[tr],Yd[tr]).predict(Xh),Yd,te))
        agg["SpaGE-style (PC-kNN)"].append(medS(KNeighborsRegressor(30).fit(P[tr],Yd[tr]).predict(P),Yd,te))
        agg["MLP (diffusion features)"].append(medS(MLPRegressor(hidden_layer_sizes=(128,),alpha=1e-2,max_iter=120,early_stopping=True,random_state=0).fit(T[tr],Yd[tr]).predict(T),Yd,te))
    print(f"  {s} done",flush=True)
bench={m:[float(np.mean(v)),float(np.std(v))] for m,v in agg.items()}
print("\n=== EXTENDED BASELINES (mean median Spearman, top-1000, spatial 3-fold) ===")
for m in methods: print(f"  {m:28s}: {bench[m][0]:+.3f} ± {bench[m][1]:.3f}")

# ---- leave-region-out (pathology) ----
lro=[]
for s in secs:
    Xh,Yd,Yraw,co,reg=DAT[s]
    if reg is None: continue
    A=nadj(co); regs=[r for r in np.unique(reg) if (reg==r).sum()>60 and r not in ['nan','None','']]
    for held in regs:
        tr=np.where(reg!=held)[0]; te=np.where(reg==held)[0]
        if len(te)<40: continue
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32); T=np.hstack([P,A@P,A@(A@P)])
        lro.append(medS(fp(T,Yd,tr,te,inner(co[tr])),Yd,te))
lro_s=[float(np.mean(lro)),float(np.std(lro))] if lro else [None,None]
print(f"\nLeave-region-out (pathology domains): {lro_s[0]:+.3f} ± {lro_s[1]:.3f}  (n={len(lro)} region-holdouts)")

# ---- accuracy vs abundance ----
Xh,Yd,Yraw,co,_=DAT['BC_525']; A=nadj(co); blk=KMeans(5,n_init=3,random_state=0).fit_predict(co)
pred=np.zeros_like(Yd)
for fb in range(5):
    tr=blk!=fb; te=blk==fb; P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32); T=np.hstack([P,A@P,A@(A@P)])
    pred[te]=fit_predict(T[tr],Yd[tr],T[te],blk[tr])
rmet=np.array([spearmanr(pred[:,j],Yd[:,j])[0] if Yd[:,j].std()>1e-9 else np.nan for j in range(Yd.shape[1])])
abund=np.log1p(Yraw).mean(0); q=np.quantile(abund,[0,.2,.4,.6,.8,1.0]); ab=[]
for i in range(5):
    m=(abund>=q[i])&(abund<=q[i+1])&np.isfinite(rmet); ab.append([float((q[i]+q[i+1])/2),float(np.nanmedian(rmet[m]))])
print("\naccuracy vs abundance (quintile center, median r):",[(round(a,2),round(b,2)) for a,b in ab])
json.dump(dict(baselines=bench,leave_region_out=lro_s,abundance=ab),open(f"{SP}/rigor_baselines.json","w"),indent=2)
print("saved rigor_baselines.json")
