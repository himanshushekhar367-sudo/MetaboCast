"""(1) Diffusion-hop ablation: does the diffusion prior saturate at 2-3 hops? (2) Spatial-leakage
audit: random-spot vs spatial-block vs leave-region-out vs leave-one-sample-out. Random-spot splits
inflate accuracy via spatial autocorrelation; spatially blocked splits give the estimates reported in the manuscript.
Model: the released estimator (reproduce/estimator.py). Inner CV for the penalty uses five k-means blocks of the training spots
(sections for leave-one-sample-out), exactly as MetaboCast.fit does."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import anndata as ad, numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import fit_predict
warnings.filterwarnings('ignore')
DATA=P.DESIUM; SP=P.WORK
def inner(co_tr): return KMeans(5,n_init=3,random_state=0).fit_predict(co_tr)
secs=["BC_525","BC_823","LC_091","LC_170"]; NMET=200
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def load(s):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],np.float32)); co=np.asarray(a.obsm['spatial'],float)
    det=(Y>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-NMET:]]
    reg=None
    for key in ['region','pathology','annotation','tissue','domain']:
        if key in a.obs: reg=a.obs[key].astype(str).values; break
    hv=np.argsort(X.var(0))[-3000:]; return X[:,hv],Y[:,top],co,reg
def medS(pred,Y,te): return float(np.nanmedian([spearmanr(pred[te,j],Y[te,j])[0] for j in range(Y.shape[1]) if Y[te,j].std()>1e-9]))
DAT={s:load(s) for s in secs}
print("region labels present:",{s:(DAT[s][3] is not None) for s in secs})

# ---- (1) diffusion-hop ablation (spatial 3-fold) ----
hopres={h:[] for h in range(5)}
for s in secs:
    Xh,Yd,co,_=DAT[s]; A=nadj(co); blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)
    for fb in range(3):
        tr=blk!=fb; te=blk==fb; fit=np.where(tr)[0]
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[fit]).transform(Xh).astype(np.float32)
        bases=[P]; cur=P
        for h in range(5):
            feats=np.hstack(bases); pr=np.zeros_like(Yd); pr[te]=fit_predict(feats[tr],Yd[tr],feats[te],inner(co[tr]))
            hopres[h].append(medS(pr,Yd,te)); cur=A@cur; bases.append(cur)
hops={h:[float(np.mean(v)),float(np.std(v))] for h,v in hopres.items()}
print("\n=== DIFFUSION-HOP ABLATION (mean median Spearman) ===")
for h in range(5): print(f"  {h}-hop [P..A^{h}P]: {hops[h][0]:+.3f} ± {hops[h][1]:.3f}")

# ---- (2) split-hierarchy leakage audit ----
def feats_full(Xh,co,fit):
    P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[fit]).transform(Xh).astype(np.float32); A=nadj(co)
    return np.hstack([P,A@P,A@(A@P)])
rand=[]; block=[]
for s in secs:
    Xh,Yd,co,_=DAT[s]
    kf=KFold(5,shuffle=True,random_state=0)
    for tr,te in kf.split(Xh):
        T=feats_full(Xh,co,tr); pr=np.zeros_like(Yd); pr[te]=fit_predict(T[tr],Yd[tr],T[te],inner(co[tr])); rand.append(medS(pr,Yd,te))
    blk=KMeans(5,n_init=3,random_state=0).fit_predict(co)
    for fb in range(5):
        tr=np.where(blk!=fb)[0]; te=np.where(blk==fb)[0]
        T=feats_full(Xh,co,tr); pr=np.zeros_like(Yd); pr[te]=fit_predict(T[tr],Yd[tr],T[te],blk[tr]); block.append(medS(pr,Yd,te))
# LOSO across the 4 sections (shared genes), detection-matched-ish on these top panels
genes=sorted(set.intersection(*[set(ad.read_h5ad(f"{DATA}/{s}.h5ad").var_names) for s in secs]))
print("\ncomputing LOSO...",flush=True)
As={}; Xg={}; Yl={}; cog={}
for s in secs:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); v=a[:,genes].layers['log1p']
    Xg[s]=(np.asarray(v.todense()) if hasattr(v,'todense') else np.asarray(v)).astype(np.float32)
    Yl[s]=np.log1p(np.asarray(a.uns['msi'],np.float32)); cog[s]=np.asarray(a.obsm['spatial'],float); As[s]=nadj(cog[s])
gv=np.vstack([Xg[s] for s in secs]).var(0); hv=np.argsort(gv)[-2000:]
loso=[]
for held in secs:
    tr=[s for s in secs if s!=held]
    pca=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s][:,hv] for s in tr]))
    def F(s): P=pca.transform(Xg[s][:,hv]).astype(np.float32); return np.hstack([P,As[s]@P,As[s]@(As[s]@P)])
    lab=np.concatenate([[s]*len(Yl[s]) for s in tr])
    pr=fit_predict(np.vstack([F(s) for s in tr]),np.vstack([Yl[s] for s in tr]),F(held),lab,lab); Yh=Yl[held]
    loso.append(float(np.nanmedian([spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if Yh[:,j].std()>1e-9])))
splits=dict(random_spot=[float(np.mean(rand)),float(np.std(rand))],
            spatial_block=[float(np.mean(block)),float(np.std(block))],
            LOSO=[float(np.mean(loso)),float(np.std(loso))])
print("\n=== SPLIT-HIERARCHY LEAKAGE AUDIT (median Spearman) ===")
for k,v in splits.items(): print(f"  {k:14s}: {v[0]:+.3f} ± {v[1]:.3f}")
print(f"  leakage inflation (random - block) = {splits['random_spot'][0]-splits['spatial_block'][0]:+.3f}")
json.dump(dict(hops=hops,splits=splits,n_metab=NMET),open(f"{SP}/rigor_splits.json","w"),indent=2)
print("saved rigor_splits.json")
