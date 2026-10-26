"""Is accuracy limited by biology or by the learner? Compare learner families on identical diffusion
features, and sweep PCA dimensionality. If all families plateau, the transcriptome, not the learner, bounds
predictability. Representative sections, top-variance metabolites, spatial 3-fold CV.
The two MetaboCast rows are the released estimator (reproduce/estimator.py: rank targets, cross-validated
penalty, ridge with or without ExtraTrees); the other learner families are fitted to log intensities.
The PCA-dimension sweep uses the released estimator."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import anndata as ad, numpy as np, json, time, warnings
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import Estimator, fit_predict
warnings.filterwarnings('ignore')
DATA=P.DESIUM; SP=P.WORK
secs=["BC_525","LC_091"]; NMET=80; NF=3
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def load(s):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Y=np.log1p(np.asarray(a.uns['msi'],np.float32)); co=np.asarray(a.obsm['spatial'],float)
    det=(Y>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-NMET:]]; Yd=Y[:,top]
    hv=np.argsort(X.var(0))[-3000:]; return X[:,hv],Yd,co
def medS(pred,Y,te): return float(np.nanmedian([spearmanr(pred[te,j],Y[te,j])[0] for j in range(Y.shape[1]) if Y[te,j].std()>1e-9]))
def diffuse(Xh,co,fit,npc=100):
    P=PCA(npc,svd_solver='randomized',random_state=0).fit(Xh[fit]).transform(Xh).astype(np.float32)
    A=nadj(co); return np.hstack([P,A@P,A@(A@P)])

DAT={s:load(s) for s in secs}
learners={
 'MetaboCast':lambda: Estimator(),
 'MetaboCast (ridge only)':lambda: Estimator(estimator='ridge'),
 'Ridge (alpha 200)':lambda: Ridge(alpha=200),
 'ElasticNet':lambda: MultiOutputRegressor(ElasticNet(alpha=0.05,l1_ratio=0.5,max_iter=2000),n_jobs=2),
 'RandomForest':lambda: RandomForestRegressor(n_estimators=60,max_depth=12,n_jobs=2,random_state=0),
 'GradBoost(HGB)':lambda: MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120,max_depth=4,learning_rate=0.08),n_jobs=2),
 'MLP':lambda: MLPRegressor(hidden_layer_sizes=(128,),alpha=1e-2,max_iter=120,early_stopping=True,random_state=0)}
res={m:[] for m in learners}
for s in secs:
    Xh,Yd,co=DAT[s]; blk=KMeans(NF,n_init=3,random_state=0).fit_predict(co)
    for fb in range(NF):
        tr=blk!=fb; te=blk==fb; fit=np.where(tr)[0]
        T=diffuse(Xh,co,fit)
        for name,mk in learners.items():
            if name=='GradBoost(HGB)' and s!='BC_525': continue    # HGB heavy: one section
            t0=time.time(); mdl=mk()
            if isinstance(mdl,Estimator): mdl.fit(T[tr],Yd[tr],KMeans(5,n_init=3,random_state=0).fit_predict(co[tr]))
            else: mdl.fit(T[tr],Yd[tr])
            pr=mdl.predict(T)
            res[name].append(medS(pr,Yd,te)); print(f"  {s} fold{fb} {name:15s} medr={res[name][-1]:+.3f} ({time.time()-t0:.0f}s)",flush=True)
fam={m:[float(np.mean(v)),float(np.std(v))] for m,v in res.items() if v}
print("\n=== MODEL-FAMILY CEILING (mean median Spearman, top-%d metab, spatial CV) ==="%NMET)
for m in learners:
    if fam.get(m): print(f"  {m:15s}: {fam[m][0]:+.3f} ± {fam[m][1]:.3f}")

# ---- PCA dimensionality sweep (Ridge) ----
dim={}
for npc in [25,50,100,200]:
    vv=[]
    for s in secs:
        Xh,Yd,co=DAT[s]; blk=KMeans(NF,n_init=3,random_state=0).fit_predict(co)
        for fb in range(NF):
            tr=blk!=fb; te=blk==fb; T=diffuse(Xh,co,np.where(tr)[0],npc=npc)
            pr=np.zeros_like(Yd); pr[te]=fit_predict(T[tr],Yd[tr],T[te],KMeans(5,n_init=3,random_state=0).fit_predict(co[tr])); vv.append(medS(pr,Yd,te))
    dim[npc]=float(np.mean(vv)); print(f"PCA dim {npc:3d}: medr={dim[npc]:+.3f}",flush=True)
json.dump(dict(family=fam,pca_dim=dim,n_metab=NMET),open(f"{SP}/rigor_ceiling.json","w"),indent=2)
print("saved rigor_ceiling.json")
