"""EXPERIMENTAL morphology-only ablation on DESIUM: characterises the H&E-only companion model
(metabocast.MorphologyMetabolitePredictor) with three controls, top-1000 well-measured panel.

  LOSO   : transcriptome vs transcriptome+H&E vs H&E-only
  SHUFFLE: H&E-only LOSO with permuted features (a signal from tissue structure collapses to about 0)
  WITHIN : spatial 5-fold per section, all three arms

Run: python reproduce/histology_only_ablation.py --data /path/to/DESIUM/.../data --test loso|shuffle|within
     python reproduce/histology_only_ablation.py --data <DESIUM_DIR> --test all --out reproduce/results/ablations/histology_only_desium.csv
"""
import argparse, os, sys, warnings, gc, numpy as np, anndata as ad, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from metabocast import MetaboCast, add_histology_features, MorphologyMetabolitePredictor

SAMPLES = ["BC_515_Section_1","BC_515_Section_2","BC_525","BC_823","LC_091","LC_170","LC_276"]

def top_ions(a, n=1000):
    raw=np.asarray(a.uns["msi"],float); y=np.log1p(raw); det=(raw>0).mean(0)>0.20
    k=np.where(det)[0]; return k[np.argsort(y[:,k].var(0))[-n:]] if k.size>n else k
def sc(true,pred,ions):
    rr=[spearmanr(true[:,k],pred[:,k])[0] for k in ions if true[:,k].std()>1e-9 and pred[:,k].std()>1e-9]
    return float(np.nanmedian(rr)) if rr else float("nan")
def subset(a, mask):
    s=a[mask].copy(); s.uns["msi"]=np.asarray(a.uns["msi"],float)[mask]; s.uns["mz_features"]=a.uns["mz_features"]; return s

def load(data):
    secs,names=[],[]
    for s in SAMPLES:
        p=os.path.join(data,s+".h5ad")
        if not os.path.exists(p): continue
        a=ad.read_h5ad(p)
        try: add_histology_features(a,key_added="histology")
        except Exception as e: print(f"skip {s}: {str(e)[:50]}");
        else:
            if "spatial" in a.uns: del a.uns["spatial"]
            secs.append(a); names.append(s)
    return secs,names

def run_loso(secs,names,shuffle=False,seed=0,he_only=False):
    rng=np.random.RandomState(seed); rows=[]
    for i,s in enumerate(names):
        tr=[secs[j] for j in range(len(secs)) if j!=i]; te=secs[i]
        true=np.log1p(np.asarray(te.uns["msi"],float)); ions=top_ions(te)
        if not (shuffle or he_only):
            for lab,ek in [("MetaboCast",None),("MetaboCast+H&E","histology")]:
                m=MetaboCast(n_hvg=2000,extra_key=ek).fit(tr); p=m.predict_metabolome(te.copy())
                rows.append((s,lab,sc(true,p,ions))); del m,p; gc.collect()
        mm=MorphologyMetabolitePredictor().fit(tr)
        teq=te.copy()
        if shuffle: teq.obsm["histology"]=np.asarray(teq.obsm["histology"])[rng.permutation(teq.n_obs)]
        rows.append((s,"H&E-only"+("(shuffled)" if shuffle else ""),sc(true,mm.predict_from_histology(teq),ions)))
        print("  ",s,rows[-1],flush=True); gc.collect()
    return pd.DataFrame(rows,columns=["sample","model","median_spearman"])

def run_within(secs,names,he_only=False):
    rows=[]
    for a,s in zip(secs,names):
        folds=KMeans(5,n_init=5,random_state=0).fit_predict(np.asarray(a.obsm["spatial"],float))
        ions=top_ions(a); true=np.log1p(np.asarray(a.uns["msi"],float))
        for lab,ek in ([] if he_only else [("MetaboCast",None),("MetaboCast+H&E","histology")]):
            ps=[sc(true[folds==f],MetaboCast(n_hvg=2000,extra_key=ek).fit([subset(a,folds!=f)]).predict_metabolome(subset(a,folds==f)),ions) for f in range(5)]
            rows.append((s,lab,float(np.nanmean(ps)))); gc.collect()
        ps=[sc(true[folds==f],MorphologyMetabolitePredictor().fit([subset(a,folds!=f)]).predict_from_histology(subset(a,folds==f)),ions) for f in range(5)]
        rows.append((s,"H&E-only",float(np.nanmean(ps)))); print("  ",s,"done",flush=True); gc.collect()
    return pd.DataFrame(rows,columns=["sample","model","median_spearman"])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default=os.environ.get("DESIUM_DIR","."))
    ap.add_argument("--test",choices=["loso","shuffle","within","all"],default="loso",
                    help="'all' runs the three tests for the H&E-only model and writes one table (histology table)")
    ap.add_argument("--out",default=None)
    ap.add_argument("--he-only",action="store_true",help="run only the H&E-only model (skip the transcriptome arms)")
    a=ap.parse_args()
    secs,names=load(a.data)
    if len(secs)<2: raise SystemExit("need >=2 sections with matched histology")
    if a.test=="all":
        parts=[("LOSO (cross-section)",run_loso(secs,names,he_only=True)),
               ("shuffle control (H&E-only)",run_loso(secs,names,shuffle=True)),
               ("within-sample 5-fold",run_within(secs,names,he_only=True))]
        df=pd.concat([d.assign(test=t) for t,d in parts],ignore_index=True).rename(columns={"sample":"section"})
        df=df[["test","section","model","median_spearman"]].round({"median_spearman":4})
    else:
        df={"loso":lambda:run_loso(secs,names,he_only=a.he_only),"shuffle":lambda:run_loso(secs,names,shuffle=True),
            "within":lambda:run_within(secs,names,he_only=a.he_only)}[a.test]()
    keys=["test","model"] if "test" in df else "model"
    print("\n=== mean over sections ===\n"+df.groupby(keys).median_spearman.mean().round(4).to_string())
    if a.out: os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True); df.to_csv(a.out,index=False); print("saved",a.out)
