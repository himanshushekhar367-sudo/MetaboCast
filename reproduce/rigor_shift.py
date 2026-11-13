"""(A) Why does cross-sample transfer fail? Relate LOSO accuracy to transcriptomic distribution shift
(embedding distance) across the 7 sections. (B) Epistemic-stratified calibration: does conformal
coverage degrade FIRST in the high-epistemic regime? (a compelling OOD-detection result).
Model: the released estimator (reproduce/estimator.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import numpy as np, json, warnings, gc
from sklearn.decomposition import PCA
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import make
warnings.filterwarnings('ignore'); rng=np.random.RandomState(0)
DATA=P.DESIUM; SP=P.WORK
secs=["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]
def adj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
from pathlib import Path
from within_benchmark import read_section     # slim h5py reader: log1p expression (float32), MSI, m/z, coordinates
def _names(s):
    import h5py
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
    with h5py.File(f"{DATA}/{s}.h5ad","r") as f: return [str(v) for v in read_elem(f["var"]).index]
genes=sorted(set.intersection(*[set(_names(s)) for s in secs]))
def gm(s):
    names,x,raw,_,c=read_section(Path(f"{DATA}/{s}.h5ad")); pos={g:i for i,g in enumerate(names)}
    return x[:,[pos[g] for g in genes]],np.asarray(raw,np.float32),np.asarray(c,float)
# two-pass: pick top-2000 HVG among shared genes without holding all full matrices
s1=np.zeros(len(genes)); s2=np.zeros(len(genes)); N=0; Y={};co={};As={}
for s in secs:
    M,Yr,c=gm(s); s1+=M.sum(0); s2+=(M*M).sum(0); N+=M.shape[0]; Y[s]=np.log1p(Yr); co[s]=c; As[s]=adj(c); del M,Yr; gc.collect()
hv=np.argsort(s2/N-(s1/N)**2)[-2000:]
Xg={s:gm(s)[0][:,hv] for s in secs}; gc.collect()
pca=PCA(50,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s] for s in secs]))
emb={s:pca.transform(Xg[s]) for s in secs}; cent={s:emb[s].mean(0) for s in secs}
# (A) LOSO r + embedding shift per held section  (Xg already restricted to 2000 HVG)
def feat(P,s): return np.hstack([P,As[s]@P,As[s]@(As[s]@P)])
shift=[]; loso=[]; metshift=[]
for held in secs:
    tr=[s for s in secs if s!=held]
    pc=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s] for s in tr]))
    lab=np.concatenate([[s]*len(Y[s]) for s in tr])
    reg=make().fit(np.vstack([feat(pc.transform(Xg[s]).astype(np.float32),s) for s in tr]),np.vstack([Y[s] for s in tr]),lab,lab)
    pr=reg.predict(feat(pc.transform(Xg[held]).astype(np.float32),held)); Yh=Y[held]
    loso.append(float(np.nanmedian([spearmanr(pr[:,j],Yh[:,j])[0] for j in range(Yh.shape[1]) if Yh[:,j].std()>1e-9])))
    train_cent=np.mean([cent[s] for s in tr],0); shift.append(float(np.linalg.norm(cent[held]-train_cent)))
    metshift.append(float(np.linalg.norm(Y[held].mean(0)-np.mean([Y[s].mean(0) for s in tr],0))))
rt=spearmanr(shift,loso)[0]; rm=spearmanr(metshift,loso)[0]
print("(A) per-section: LOSO r vs distribution shift")
for i,s in enumerate(secs): print(f"  {s:18s} LOSO={loso[i]:+.3f}  embed-shift={shift[i]:.2f}  metab-shift={metshift[i]:.3f}")
print(f"  Spearman(embedding shift, LOSO r) = {rt:+.3f}   Spearman(metabolite shift, LOSO r) = {rm:+.3f}")

# (B) epistemic-stratified calibration — pool IN-DISTRIBUTION test spots + OOD (held-section) spots
sub=["BC_525","BC_823","LC_091","LC_170"]; held="LC_170"; tr=[s for s in sub if s!=held]
pc=PCA(100,svd_solver='randomized',random_state=0).fit(np.vstack([Xg[s] for s in tr]))
Ftr=np.vstack([feat(pc.transform(Xg[s]).astype(np.float32),s) for s in tr]); Ytr=np.vstack([Y[s] for s in tr])
Fh=feat(pc.transform(Xg[held]).astype(np.float32),held); Yh=Y[held]; Ltr=np.concatenate([[s]*len(Y[s]) for s in tr])
idx=rng.permutation(len(Ftr)); ncal=len(idx)//5; ntest=3000; cal=idx[:ncal]; test_id=idx[ncal:ncal+ntest]; fit=idx[ncal+ntest:]
K=10; ep_id=[]; ep_ood=[]
for k in range(K):
    bs=rng.randint(0,len(fit),len(fit)); m=make().fit(Ftr[fit][bs],Ytr[fit][bs],Ltr[fit][bs],Ltr[fit][bs])
    ep_id.append(m.predict(Ftr[test_id])); ep_ood.append(m.predict(Fh))
epi=np.concatenate([np.array(ep_id).std(0).mean(1),np.array(ep_ood).std(0).mean(1)])   # per-spot epistemic, pooled
reg=make().fit(Ftr[fit],Ytr[fit],Ltr[fit],Ltr[fit]); qhat=np.quantile(np.abs(Ytr[cal]-reg.predict(Ftr[cal])),0.9,axis=0)
cov=np.concatenate([(np.abs(Ytr[test_id]-reg.predict(Ftr[test_id]))<=qhat).mean(1),
                    (np.abs(Yh-reg.predict(Fh))<=qhat).mean(1)])                        # per-spot coverage, pooled
qs=np.quantile(epi,[0,.25,.5,.75,1.0]); strat=[]
for i in range(4):
    m=(epi>=qs[i])&(epi<=qs[i+1]); strat.append([float((qs[i]+qs[i+1])/2),float(cov[m].mean())])
print("\n(B) coverage stratified by epistemic uncertainty (in-dist + OOD pooled; quartile center, coverage):",[(round(a,3),round(b,2)) for a,b in strat])
json.dump(dict(shift_loso=[[secs[i],loso[i],shift[i],metshift[i]] for i in range(len(secs))],
    spearman_embed_loso=float(rt),spearman_metab_loso=float(rm),
    epistemic_calibration=strat),open(f"{SP}/rigor_shift.json","w"),indent=2)
print("saved rigor_shift.json")
