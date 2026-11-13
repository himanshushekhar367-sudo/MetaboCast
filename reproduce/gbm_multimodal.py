"""Is antibody-derived protein a non-redundant input? CytAssist GBM has RNA (18,085 genes) + 35 antibody
markers on the SAME spots but no paired MSI, so we cannot score a metabolite-accuracy gain. We can,
however, test whether protein carries spatial signal ORTHOGONAL to the transcriptome — i.e. whether it
is a non-redundant input a metabolite predictor could exploit. Metric: R^2 of RNA(PCA)->protein per marker;
low R^2 / large unexplained variance = orthogonal information."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths as P  # data and work locations; see reproduce/paths.py
import scanpy as sc, numpy as np, json, warnings
from sklearn.decomposition import PCA; from sklearn.linear_model import Ridge; from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
UP=P.ROOT; SP=P.WORK
ad=sc.read_10x_h5(f"{UP}/CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_filtered_feature_bc_matrix.h5",gex_only=False)
ad.var_names_make_unique()
ft=ad.var['feature_types'].astype(str).values
rna=ad[:,ft=='Gene Expression'].copy(); prot=ad[:,ft=='Antibody Capture'].copy()
pnames=list(prot.var_names)
print(f"GBM CytAssist: {ad.n_obs} spots | {rna.n_vars} genes | {prot.n_vars} protein markers")
# normalize
sc.pp.normalize_total(rna,target_sum=1e4); sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna,n_top_genes=3000); rna=rna[:,rna.var.highly_variable].copy()
X=np.asarray(rna.X.todense()) if hasattr(rna.X,'todense') else np.asarray(rna.X)
P=np.asarray(prot.X.todense()) if hasattr(prot.X,'todense') else np.asarray(prot.X)
P=np.log1p(P)                                   # CLR-lite: log1p on antibody counts
P=(P-P.mean(0))/(P.std(0)+1e-8)
# 5-fold CV R^2 of RNA-PCA -> each protein marker
kf=KFold(5,shuffle=True,random_state=0); r2=np.zeros(P.shape[1]);
for tr,te in kf.split(X):
    pca=PCA(50,svd_solver='randomized',random_state=0).fit(X[tr]); Ztr=pca.transform(X[tr]); Zte=pca.transform(X[te])
    for j in range(P.shape[1]):
        m=Ridge(alpha=10).fit(Ztr,P[tr,j]); pr=m.predict(Zte)
        ss_res=np.sum((P[te,j]-pr)**2); ss_tot=np.sum((P[te,j]-P[te,j].mean())**2)
        r2[j]+= (1-ss_res/ss_tot)/5
orth=1-r2
o=np.argsort(r2)
print("\nRNA(PCA-50) -> protein reconstruction, 5-fold CV R^2:")
print(f"  mean R^2 = {r2.mean():.2f} ; mean unexplained (orthogonal) variance = {orth.mean():.2f}")
print(f"  markers with R^2<0.3 (largely orthogonal to RNA): {int((r2<0.3).sum())}/{len(r2)}")
print("  most orthogonal markers (lowest R^2):")
for j in o[:8]: print(f"    {pnames[j]:16s} R^2={r2[j]:.2f}")
json.dump(dict(n_spots=int(ad.n_obs),n_genes=int(rna.n_vars),n_prot=int(prot.n_vars),
    mean_r2=float(r2.mean()),mean_orthogonal=float(orth.mean()),n_orthogonal=int((r2<0.3).sum()),
    per_marker={pnames[j]:float(r2[j]) for j in range(len(r2))}),open(f"{SP}/gbm_multimodal.json","w"),indent=2)
print("saved gbm_multimodal.json")
