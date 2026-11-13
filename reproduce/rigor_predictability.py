"""Determinants of predictability (Additional file 1: SuppFig_predictability a,b; robustness and
predictability_ranking tables).

Per-ion within-section accuracy (Spearman r of five-fold spatial-block cross-validated predictions with the released
estimator, reproduce/estimator.py) is regressed on measurable properties of the measured ion: mean log abundance,
coefficient of variation, spatial autocorrelation (Moran's I on the binary k-nearest-neighbour graph, k = 6),
detection rate and putative chemical group (lipid, polar or unannotated; single-class putative candidates from
reproduce/annotate_ions.py). The unit of analysis is an ion-section pair (ions detected in >20% of a section's spots).

Reported:
  univariate     Spearman correlation of accuracy with each property, with a 95% interval from 2,000 bootstrap
                 resamples of ions (all sections of an ion resampled together, because the same ion appears in
                 several sections);
  cv_r2          variance explained by a gradient-boosting regressor on held-out rows under two grouped designs:
                 leave-one-section-out (4 folds) and five folds of ions (GroupKFold by m/z, so no ion is in both
                 training and test rows);
  permutation    the ranking: decrease in held-out R² when one property (the two chemical-group indicators
                 together) is permuted within the held-out fold, 20 permutations per fold; mean and range over the
                 leave-one-section-out folds, and mean over the ion folds;
  impurity       gradient-boosting impurity importance of a model fitted to all rows (for reference only: it favours
                 continuous features and is not computed on held-out data).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import anndata as ad, numpy as np, pandas as pd, json, warnings
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr; from scipy.spatial import cKDTree; from scipy.sparse import csr_matrix,diags,eye
from estimator import fit_predict
warnings.filterwarnings('ignore')
DATA=P.DESIUM; SP=P.WORK
secs=["BC_525","BC_823","LC_091","LC_170"]
# chemical groups of single-class putative candidates: lipids are the hydrophobic classes; every other annotated class
# (amino and carboxylic acids, sugars and sugar phosphates, nucleotides, nucleosides and nucleobases, cofactors,
# carnitines, choline and its phosphodiesters, polyamines, vitamins) is polar
LIPID={'fatty acid','glycerolipid','phospholipid','lysophospholipid','sphingolipid','sterol'}
ann=pd.read_csv(os.environ.get("METABOCAST_ANNOTATION", os.path.join(P.REPO,"reproduce","results","desium_ion_annotation.csv")))
ann=ann[(ann["n_candidates"]>0)&(ann["class"]!="ambiguous")].copy(); ann["mz"]=ann["mz"].round(3); CL=dict(zip(ann.mz,ann['class']))
def clsof(m): v=CL.get(round(float(m),3),'unannotated'); return v if isinstance(v,str) else 'unannotated'
def group(c): return 'unannotated' if c=='unannotated' else ('lipid' if c in LIPID else 'polar')
def A_bin(c,k=6):
    n=len(c); d,idx=cKDTree(c).query(c,k=min(k+1,n)); r=np.repeat(np.arange(n),idx.shape[1]-1); cc=idx[:,1:].ravel()
    B=csr_matrix((np.ones(len(r)),(r,cc)),shape=(n,n)); B=((B+B.T)>0).astype(np.float32); return B
def norm(A): return diags((1/np.sqrt(np.asarray((A+eye(A.shape[0])).sum(1)).ravel())).astype(np.float32))@(A+eye(A.shape[0]))
rows=[]
for s in secs:
    a=ad.read_h5ad(f"{DATA}/{s}.h5ad"); X=a.layers['log1p']; X=(np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)).astype(np.float32)
    Yr=np.asarray(a.uns['msi'],np.float32); Y=np.log1p(Yr); co=np.asarray(a.obsm['spatial'],float); mz=np.round(np.asarray(a.uns['mz_features'],float),4)
    det=(Yr>0).mean(0); keep=np.where(det>0.2)[0]; Yd=Y[:,keep]; mzk=mz[keep]
    B=A_bin(co); An=norm(B); W=B.sum()
    hv=np.argsort(X.var(0))[-3000:]; Xh=X[:,hv]; blk=KMeans(5,n_init=3,random_state=0).fit_predict(co)
    pred=np.zeros_like(Yd)
    for fb in range(5):
        tr=blk!=fb; te=blk==fb
        Pc=PCA(100,svd_solver='randomized',random_state=0).fit(Xh[tr]).transform(Xh).astype(np.float32); T=np.hstack([Pc,An@Pc,An@(An@Pc)])
        pred[te]=fit_predict(T[tr],Yd[tr],T[te],blk[tr])
    r=np.array([spearmanr(pred[:,j],Yd[:,j])[0] if Yd[:,j].std()>1e-9 else np.nan for j in range(Yd.shape[1])])
    Yc=Yd-Yd.mean(0); Bc=B@Yc; moran=(Yd.shape[0]/W)*(np.sum(Yc*Bc,0)/(np.sum(Yc*Yc,0)+1e-9))
    abund=Y[:,keep].mean(0); cv=Yr[:,keep].std(0)/(Yr[:,keep].mean(0)+1e-9); detf=det[keep]
    for j in range(len(mzk)):
        c=clsof(mzk[j])
        rows.append(dict(sec=s,mz=float(mzk[j]),r=r[j],abundance=abund[j],cv=cv[j],moran=moran[j],detection=detf[j],cls=c,group=group(c)))
df=pd.DataFrame(rows).dropna(subset=['r']).reset_index(drop=True)
df['is_lipid']=(df['group']=='lipid').astype(float); df['is_polar']=(df['group']=='polar').astype(float)
PROPS=['abundance','cv','moran','detection']
FEATS=PROPS+['is_lipid','is_polar']
GROUPS={'abundance':['abundance'],'cv':['cv'],'moran':['moran'],'detection':['detection'],'chemical group':['is_lipid','is_polar']}
Xg=df[FEATS].values; yg=df['r'].values
ion=df['mz'].round(3).values; uion=np.unique(ion); pos={m:np.where(ion==m)[0] for m in uion}

# univariate Spearman with an ion-level bootstrap interval
rng=np.random.RandomState(0)
def rho(idx,p): return spearmanr(df['r'].values[idx],df[p].values[idx])[0]
uni={}
boots={p:[] for p in PROPS}
for _ in range(2000):
    pick=rng.choice(uion,len(uion),replace=True); idx=np.concatenate([pos[m] for m in pick])
    for p in PROPS: boots[p].append(rho(idx,p))
for p in PROPS:
    uni[p]=dict(rho=float(rho(np.arange(len(df)),p)),ci=[float(np.percentile(boots[p],2.5)),float(np.percentile(boots[p],97.5))])

def gbr(): return GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=0)
def r2(y,p): return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
def grouped(groups,n_splits):
    """Held-out R² of the pooled out-of-fold predictions and per-fold permutation importance."""
    yp=np.zeros_like(yg); imp={g:[] for g in GROUPS}; fold_r2=[]; names=[]
    for k,(tr,te) in enumerate(GroupKFold(n_splits=n_splits).split(Xg,yg,groups)):
        m=gbr().fit(Xg[tr],yg[tr]); yp[te]=m.predict(Xg[te]); base=r2(yg[te],yp[te]); fold_r2.append(base)
        names.append(str(groups[te][0])); prng=np.random.RandomState(k)
        for g,cols in GROUPS.items():
            ci=[FEATS.index(c) for c in cols]; drops=[]
            for _ in range(20):
                Xp=Xg[te].copy(); Xp[:,ci]=Xp[prng.permutation(len(te))][:,ci]
                drops.append(base-r2(yg[te],m.predict(Xp)))
            imp[g].append(float(np.mean(drops)))
    return r2(yg,yp),fold_r2,imp,names
r2_sec,fold_sec,imp_sec,fold_names=grouped(df['sec'].values,len(secs))
r2_ion,fold_ion,imp_ion,_=grouped(ion,5)
perm={g:dict(loso_mean=float(np.mean(imp_sec[g])),loso_min=float(np.min(imp_sec[g])),loso_max=float(np.max(imp_sec[g])),
             loso_per_section=dict(zip(fold_names,imp_sec[g])),ion_folds_mean=float(np.mean(imp_ion[g]))) for g in GROUPS}
ranking=sorted(GROUPS,key=lambda g:-perm[g]['loso_mean'])
full=gbr().fit(Xg,yg); impurity=dict(zip(FEATS,[float(x) for x in full.feature_importances_]))
print("Univariate Spearman (95% ion-bootstrap interval):")
for p,v in uni.items(): print(f"  {p:10s} {v['rho']:+.3f} ({v['ci'][0]:+.3f} to {v['ci'][1]:+.3f})")
print(f"held-out R²: leave-one-section-out {r2_sec:.3f} (folds {np.round(fold_sec,3)}), ion folds {r2_ion:.3f}")
print("permutation importance (ΔR², leave-one-section-out mean [min, max]):")
for g in ranking: print(f"  {g:15s} {perm[g]['loso_mean']:.3f} [{perm[g]['loso_min']:.3f}, {perm[g]['loso_max']:.3f}]  ion folds {perm[g]['ion_folds_mean']:.3f}")
json.dump(dict(univariate={p:v['rho'] for p,v in uni.items()},univariate_ci={p:v['ci'] for p,v in uni.items()},
               cv_r2=r2_sec,cv_r2_design="leave-one-section-out (4 sections), gradient boosting",
               cv_r2_ion_folds=r2_ion,fold_r2_sections=dict(zip(fold_names,fold_sec)),fold_r2_ions=fold_ion,
               permutation=perm,ranking=ranking,impurity=impurity,
               n=int(len(df)),n_ions=int(len(uion)),sections=secs,
               n_by_group={k:int(v) for k,v in df['group'].value_counts().items()},
               scatter={p:[df[p].tolist(),df['r'].tolist()] for p in ['moran','abundance']}),
          open(f"{SP}/rigor_predictability.json","w"),indent=2)
df.to_csv(f"{SP}/predictability_determinants.csv",index=False)
print("saved rigor_predictability.json")
