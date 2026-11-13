import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths as P  # data and work locations; see reproduce/paths.py
import sys, numpy as np, scanpy as sc, anndata as ad, scipy.sparse as sp, warnings, os
warnings.filterwarnings('ignore'); sys.path.insert(0,P.MGFEA_REPO)
os.chdir(os.path.join(P.BASELINES, 'mgfea_out'))
from MGFEA.multiGraphFEA import SpatialGraphFEA
from MGFEA import preprocess
from scipy.spatial import cKDTree
import torch; torch.manual_seed(0); np.random.seed(0)
D=P.DESIUM
graph=ad.read_h5ad(os.path.join(P.MGFEA_REPO, 'data/gsmmmodel/scfea_b_human.h5ad'))
for s in sys.argv[1:]:
    a=ad.read_h5ad(f"{D}/{s}.h5ad")
    a=ad.AnnData(X=sp.csr_matrix(np.asarray(a.layers['log1p'],np.float32)),obs=a.obs[[]],var=a.var[[]],obsm={'spatial':np.asarray(a.obsm['spatial'],float)})
    a.var['symbol']=a.var_names
    sc.pp.highly_variable_genes(a,flavor='seurat')
    g2=preprocess.define_genesffea(a,graph,num_hvg=2000)
    sub=a[:,a.var['genes4fea']].copy()
    c=sub.obsm['spatial']; _,idx=cKDTree(c).query(c,k=7)
    e=np.vstack([np.repeat(np.arange(len(c)),6),idx[:,1:].ravel()]); e=np.hstack([e,e[::-1]])
    m=SpatialGraphFEA(sub,e,g2,train_epoch_num=100,save_lossfig=f'{s}_loss.jpg',save_model=f'{s}.pth')
    x,flux=m.run()
    Xs=g2.X.toarray() if hasattr(g2.X,'toarray') else np.asarray(g2.X)
    bal=Xs@flux.T   # compounds x spots
    np.save(f'{s}_flux.npy',flux); np.save(f'{s}_balance.npy',bal.T)
    with open(f'{s}_compounds.txt','w') as f: f.write('\n'.join(g2.obs['c_id'].astype(str)))
    print(s,'done',flux.shape,bal.shape,flush=True)
