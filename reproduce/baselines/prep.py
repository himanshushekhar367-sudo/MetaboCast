import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths as P  # data and work locations; see reproduce/paths.py
import anndata as ad, numpy as np, pandas as pd
D=P.DESIUM
mg=pd.read_csv(os.path.join(P.SCFEA_REPO, 'data/module_gene_m168.csv'),header=None,index_col=0)
genes=set(mg.values.ravel())-{np.nan}
for s in ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]:
    a=ad.read_h5ad(f"{D}/{s}.h5ad")
    keep=[g for g in a.var_names if g in genes]
    X=a[:,keep].layers['log1p']; X=np.asarray(X.todense()) if hasattr(X,'todense') else np.asarray(X)
    pd.DataFrame(X.T,index=keep,columns=a.obs_names).to_csv(f"input/{s}.csv")
    np.save(f"input/{s}_msi.npy",np.log1p(np.asarray(a.uns['msi'],np.float32))); np.save(f"input/{s}_coords.npy",np.asarray(a.obsm['spatial'],float))
    if s=="BC_823": np.save("input/mz.npy",np.asarray(a.uns['mz_features']).astype(str))
    print(s,a.shape,len(keep),flush=True)
