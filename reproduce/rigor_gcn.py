"""Learned graph convolution vs fixed diffusion prior, matched receptive field (2 hops).
A 2-layer GCN (learned weights, trained by gradient descent) is compared to our fixed 2-hop
diffusion ridge. If the learned model does not beat the fixed prior, the gain is from spatial
context, not architectural complexity.
Both arms use the same training targets: the fixed arm is the released estimator (reproduce/estimator.py)
and the GCN is trained on rank-Gaussian targets of the training spots."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import anndata as ad, numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import fit_predict
from metabocast.metabocast import _rank_gauss
warnings.filterwarnings('ignore'); rng=np.random.RandomState(0)
DATA=P.DESIUM; SP=P.WORK
secs=["BC_525","LC_091"]; NMET=150
def nadj(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    A=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); A=((A+A.T)>0).astype(np.float32)+eye(n,dtype=np.float32)
    return diags((1/np.sqrt(np.asarray(A.sum(1)).ravel())).astype(np.float32))@A
def medS(P,Y,te): return float(np.nanmedian([spearmanr(P[te,j],Y[te,j])[0] for j in range(Y.shape[1]) if Y[te,j].std()>1e-9]))
def load(s):
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Y=np.log1p(Yr); co=np.asarray(a.obsm['spatial'],float)
    det=(Yr>0).mean(0)>0.2; top=np.where(det)[0][np.argsort(Y[:,det].var(0))[-NMET:]]; hv=np.argsort(X.var(0))[-3000:]
    return X[:,hv],Y[:,top],co
class Adam:
    def __init__(s,ps,lr=5e-3): s.lr=lr; s.m=[np.zeros_like(p) for p in ps]; s.v=[np.zeros_like(p) for p in ps]; s.t=0
    def step(s,ps,gs):
        s.t+=1
        for i,(p,g) in enumerate(zip(ps,gs)):
            s.m[i]=.9*s.m[i]+.1*g; s.v[i]=.999*s.v[i]+.001*(g*g)
            mh=s.m[i]/(1-.9**s.t); vh=s.v[i]/(1-.999**s.t); p-=s.lr*mh/(np.sqrt(vh)+1e-8)
def gcn(X,Ahat,Y,tr,h=64,epochs=300,l2=1e-3):
    n,d=X.shape; m=Y.shape[1]; AX=Ahat@X
    W1=rng.randn(d,h).astype(np.float32)*np.sqrt(1/d); W2=rng.randn(h,m).astype(np.float32)*np.sqrt(1/h)
    opt=Adam([W1,W2],lr=5e-3); trm=np.zeros(n,bool); trm[tr]=True; ntr=trm.sum()
    for e in range(epochs):
        A1=AX@W1; H1=np.maximum(A1,0); Z2=Ahat@H1; Yh=Z2@W2
        dY=(Yh-Y)*trm[:,None]/ntr
        dW2=Z2.T@dY + l2*W2; dZ2=dY@W2.T; dH1=Ahat@dZ2; dA1=dH1*(A1>0); dW1=AX.T@dA1 + l2*W1
        opt.step([W1,W2],[dW1,dW2])
    A1=AX@W1; H1=np.maximum(A1,0); return (Ahat@H1)@W2
res={"fixed 2-hop diffusion (MetaboCast)":[],"learned GCN (2-layer)":[]}
for s in secs:
    Xh,Yd,co=load(s); A=nadj(co); blk=KMeans(3,n_init=3,random_state=0).fit_predict(co)
    for fb in range(3):
        tr=np.where(blk!=fb)[0]; te=blk==fb
        P=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32)
        T=np.hstack([P,A@P,A@(A@P)]); pr=np.zeros_like(Yd)
        pr[te]=fit_predict(T[tr],Yd[tr],T[te],KMeans(5,n_init=3,random_state=0).fit_predict(co[tr]))
        res["fixed 2-hop diffusion (MetaboCast)"].append(medS(pr,Yd,te))
        Yt=np.zeros_like(Yd); Yt[tr]=_rank_gauss(Yd[tr])                           # same training targets
        pr=gcn(P,np.asarray(A.todense(),np.float32),Yt,tr); res["learned GCN (2-layer)"].append(medS(pr,Yd,te))
    print(f"  {s} done",flush=True)
out={k:[float(np.mean(v)),float(np.std(v))] for k,v in res.items()}
print("\n=== LEARNED GCN vs FIXED DIFFUSION (matched 2-hop receptive field) ===")
for k in res: print(f"  {k:30s}: {out[k][0]:+.3f} ± {out[k][1]:.3f}")
json.dump(out,open(f"{SP}/rigor_gcn.json","w"),indent=2); print("saved rigor_gcn.json")
