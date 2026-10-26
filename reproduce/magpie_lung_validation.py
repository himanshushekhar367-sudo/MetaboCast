"""Evaluation of MetaboCast on the MAGPIE DESI-MSI + Visium lung cohort.

Builds paired MetaboCast AnnData from the public MAGPIE Zenodo inputs (10.5281/zenodo.17789448,
magpie_inputs.zip), co-registering MSI pixels with independently documented orientation.
The outcome-selected orientation search (maximising transcriptome-metabolite coupling) is available only
as an explicitly exploratory option (--allow-outcome-selected-alignment); it is not used in the manuscript, because
the MAGPIE registration cannot be recovered from the public files (data/README.md).
Then runs within-section spatial 5-fold CV and leave-one-section-out (LOSO), reported by metabolite
class. DESI-MSI negative mode, so all annotations are [M-H]-.

Usage (run in the MetaboCast environment):
    python reproduce/magpie_lung_validation.py COHORT --build-dir NEW_DIR --out NEW.csv --orientations orientations.json
It auto-discovers every <section>/msi + <section>/visium pair under that folder.
Outputs: user-selected new build directory and CSV; existing outputs are never overwritten.
"""
import sys, os, re, glob, json, gc, argparse, numpy as np, pandas as pd, anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
# make the repo's metabocast/ importable whether or not the package was pip-installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metabocast import MetaboCast, common_mz_axis
from metabocast.validation import subject_splits

PROT = 1.007276
# compact [M-H]- neutral-mass class dictionary (extend as needed)
NEUT = {
 "lipid": [256.2402,284.2715,282.2559,280.2402,278.2246,304.2402,302.2246,326.2559,328.2402,
           255.2324,283.2637,301.2172],  # fatty acids (FA16:0..FA22:6) etc.
 "redox": [307.0838,176.0321,168.0283],                   # glutathione, ascorbate, urate
 "nucleotide": [347.0631,268.0808,136.0385,152.0334,267.0967,323.0519],  # AMP, inosine, hypoxanthine, xanthine, adenosine, IMP-ish
 "polar": [90.0317,180.0634,192.0270,118.0270,134.0215,147.0532,133.0375,146.0579,131.0946,
           174.0528,166.0528,181.0739,105.0426,117.0426],  # lactate,glucose,citrate,succinate,malate,Glu,Asp,Gln,Leu,...
}

def classify(mz_arr, tol=0.01):
    cls=np.array(["unannotated"]*len(mz_arr), dtype=object)
    ions={k:[m-PROT for m in v] for k,v in NEUT.items()}
    for i,mz in enumerate(mz_arr):
        matches = [k for k, masses in ions.items() if any(abs(mz-m)<tol for m in masses)]
        if len(matches) == 1:
            cls[i] = matches[0]
        elif len(matches) > 1:
            cls[i] = "ambiguous"
    return cls

def load_section(sec):
    vis=sc.read_10x_h5(glob.glob(f"{sec}/visium/*filtered_feature_bc_matrix.h5")[0]); vis.var_names_make_unique()
    tpf=glob.glob(f"{sec}/visium/spatial/tissue_positions*.csv")[0]
    hdr=0 if "in_tissue" in open(tpf).readline() else None
    tp=pd.read_csv(tpf,header=hdr)
    if hdr is None: tp.columns=["barcode","in_tissue","row","col","pxl_row","pxl_col"]
    else: tp=tp.rename(columns={"pxl_row_in_fullres":"pxl_row","pxl_col_in_fullres":"pxl_col"})
    tp=tp.set_index(tp.columns[0]); tp=tp[tp["in_tissue"]==1]
    common=[b for b in vis.obs_names if b in set(tp.index)]
    vis=vis[common].copy(); tp=tp.loc[common]
    vis.obsm["spatial"]=tp[["pxl_col","pxl_row"]].values.astype(float)
    sc.pp.normalize_total(vis,target_sum=1e4); sc.pp.log1p(vis); vis.layers["log1p"]=vis.X.copy()
    mi=pd.read_csv(glob.glob(f"{sec}/msi/MSI_intensities.csv")[0],index_col=0)
    mm=pd.read_csv(glob.glob(f"{sec}/msi/MSI_metadata.csv")[0],index_col=0).loc[mi.index]
    mz=np.array([float(re.sub(r'^mz[._]','',c)) for c in mi.columns])
    return vis, mm[["x","y"]].values.astype(float), mi.values.astype(np.float32), mz

def n01(a):
    lo=np.percentile(a,1,0); hi=np.percentile(a,99,0); return (a-lo)/(hi-lo+1e-9)
def variant(M,k):
    x,y=M[:,0],M[:,1]
    return np.column_stack([(x,y),(1-x,y),(x,1-y),(1-x,1-y),(y,x),(1-y,x),(y,1-x),(1-y,1-x)][k])

def coregister(vis, msi_xy, msi_val, orientation=None, allow_outcome_selected=False):
    S=n01(vis.obsm["spatial"]); M=n01(msi_xy)
    if orientation is not None:
        if isinstance(orientation, bool) or not isinstance(orientation, int) or not 0 <= orientation < 8:
            raise ValueError("orientation must be an integer in [0, 7]")
        _, idx = cKDTree(variant(M, orientation)).query(S, k=min(3, len(M)))
        sm = msi_val[idx].mean(1) if idx.ndim == 2 else msi_val[idx]
        return sm, np.nan, np.nan, orientation
    if not allow_outcome_selected:
        raise ValueError("provide an independently established orientation; outcome-selected alignment is exploratory only")
    Xg=np.asarray(vis.layers["log1p"].todense() if hasattr(vis.layers["log1p"],"todense") else vis.layers["log1p"],float)
    P=PCA(30,random_state=0).fit_transform(Xg-Xg.mean(0))
    def couple(sm):
        Y=np.log1p(sm); det=(sm>0).mean(0)>0.2; tr=np.arange(len(P))%2==0; te=~tr
        pr=Ridge(alpha=100).fit(P[tr],Y[tr]).predict(P[te])
        rr=[spearmanr(Y[te,j],pr[:,j])[0] for j in np.where(det)[0] if Y[te,j].std()>1e-9 and pr[:,j].std()>1e-9]
        return float(np.nanmedian(rr))
    best=(-9,None,None)
    for k in range(8):
        tree=cKDTree(variant(M,k)); d,idx=tree.query(S,k=3); sm=msi_val[idx].mean(1)
        c=couple(sm)
        if c>best[0]: best=(c,k,sm)
    c,k,sm=best
    cn=couple(sm[np.random.RandomState(0).permutation(len(sm))])
    return sm, c, cn, k

def build(cohort, outdir, orientations=None, allow_outcome_selected=False):
    secs=sorted([d for d in glob.glob(f"{cohort}/*") if os.path.isdir(os.path.join(d,"msi"))])
    if not secs:
        raise ValueError("no MAGPIE section folders found")
    orientations = orientations or {}
    for sec in secs:
        entry = orientations.get(os.path.basename(sec))
        if entry is None and not allow_outcome_selected:
            raise ValueError(f"missing independent orientation for {os.path.basename(sec)}")
        if entry is not None and (not isinstance(entry, dict) or not entry.get('source')
                                   or type(entry.get('orientation')) is not int
                                   or not 0 <= entry['orientation'] < 8):
            raise ValueError("each orientation entry needs integer orientation 0..7 and a source description")
    os.makedirs(outdir, exist_ok=False)
    built=[]
    for sec in secs:
        name=os.path.basename(sec)
        vis,mxy,mval,mz=load_section(sec)
        entry = orientations.get(name)
        sm,c,cn,k=coregister(vis,mxy,mval, orientation=entry['orientation'] if entry else None,
                            allow_outcome_selected=allow_outcome_selected)
        vis.uns["msi"]=sm.astype(np.float32); vis.uns["mz_features"]=mz; vis.uns["align_coupling"]=c; vis.uns["align_null"]=cn
        vis.uns['alignment_method'] = 'independent_orientation' if entry else 'outcome_selected_exploratory'
        vis.uns['alignment_source'] = entry['source'] if entry else 'whole-section predictive agreement; not independent validation'
        vis.uns['alignment_orientation'] = k
        p=os.path.join(outdir,f"{name}.h5ad"); vis.write(p)
        flag="OK" if c>0.05 and c>3*max(cn,0.001) else "LOW-COUPLING(check)"
        print(f"  [{name}] spots={vis.n_obs} peaks={len(mz)} coupling={c:+.3f} null={cn:+.3f} orient={k}  {flag}",flush=True)
        built.append(p)
    return built

def _slim(a):
    """Keep only what validation uses (log1p expression once, MSI, m/z axis, coordinates, provenance): the built
    files also store a duplicate count matrix, which doubled memory when all eight sections were loaded."""
    from scipy.sparse import csr_matrix
    x = a.layers["log1p"] if "log1p" in a.layers else a.X
    b = ad.AnnData(X=csr_matrix(x, dtype=np.float32), obs=pd.DataFrame(index=a.obs_names))
    b.var_names = list(a.var_names); b.layers["log1p"] = b.X
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"], float)
    for k in ("msi", "mz_features", "alignment_method", "align_coupling", "align_null"):
        if k in a.uns:
            b.uns[k] = a.uns[k]
    return b

def subset(a,m):
    b=a[m].copy(); b.uns["msi"]=np.asarray(a.uns["msi"])[m]; b.uns["mz_features"]=a.uns["mz_features"]; return b

def perion(true,pred,det):
    r=np.full(true.shape[1],np.nan)
    for j in np.where(det)[0]:
        if true[:,j].std()>1e-9 and pred[:,j].std()>1e-9: r[j]=spearmanr(true[:,j],pred[:,j])[0]
    return r

def validate(built, output, subjects=None):
    if os.path.exists(output):
        raise ValueError("output exists; choose a new path")
    secs=[_slim(ad.read_h5ad(p)) for p in built]
    mz=common_mz_axis(secs); cls=classify(np.asarray(mz,float))
    names = [os.path.basename(p)[:-5] for p in built]
    if subjects is not None and any(not isinstance(subjects.get(n), str) or not subjects[n].strip() for n in names):
        raise ValueError("subject manifest must provide a nonempty string ID for every section")
    rows=[]
    # within-section spatial 5-fold CV
    for name,a in zip([os.path.basename(p)[:-5] for p in built],secs):
        sm=np.asarray(a.uns["msi"]); true=np.log1p(sm); det=(sm>0).mean(0)>0.2
        xy=a.obsm["spatial"]; folds=KMeans(5,n_init=5,random_state=0).fit_predict((xy-xy.mean(0))/(xy.std(0)+1e-9))
        pred=np.zeros_like(true)
        for f in range(5):
            trm=folds!=f
            m=MetaboCast(n_hvg=2000,use_kegg=False).fit([subset(a,trm)]); pred[~trm]=m.predict_metabolome(subset(a,~trm)); del m; gc.collect()
        r=perion(true,pred,det)
        rows.append(dict(section=name,mode="within_cv",overall=np.nanmedian(r[det]),
                         lipid=np.nanmean(r[(cls=='lipid')&det]),polar=np.nanmean(r[(cls=='polar')&det]),
                         nucleotide=np.nanmean(r[(cls=='nucleotide')&det]),redox=np.nanmean(r[(cls=='redox')&det]),n_det=int(det.sum())))
        print(f"  within {name}: median={rows[-1]['overall']:+.3f}",flush=True)
    # LOSO across sections (needs >=2)
    if len(secs)>=2:
        ids = [subjects[n] for n in names] if subjects is not None else names
        splits = list(subject_splits(ids))
        train_for_test = {int(i): train for train, test in splits for i in test}
        for i,name in enumerate(names):
            tr=[secs[j] for j in train_for_test[i]]; te=secs[i]
            sm=np.asarray(te.uns["msi"]); true=np.log1p(sm); det=(sm>0).mean(0)>0.2
            m=MetaboCast(n_hvg=2000,use_kegg=False).fit(tr); pred=m.predict_metabolome(subset(te,np.ones(te.n_obs,bool)))
            r=perion(true,pred,det); del m; gc.collect()
            rows.append(dict(section=name,mode="leave_subject_out" if subjects is not None else "leave_section_out",overall=np.nanmedian(r[det]),
                             lipid=np.nanmean(r[(cls=='lipid')&det]),polar=np.nanmean(r[(cls=='polar')&det]),
                             nucleotide=np.nanmean(r[(cls=='nucleotide')&det]),redox=np.nanmean(r[(cls=='redox')&det]),n_det=int(det.sum())))
            print(f"  LOSO  {name}: median={rows[-1]['overall']:+.3f}",flush=True)
    df=pd.DataFrame(rows)
    # sections built before alignment provenance was recorded were all coupling-selected (exploratory)
    df['alignment_method'] = df.section.map(dict(zip(names, [a.uns.get('alignment_method', 'outcome_selected_exploratory') for a in secs])))
    df['subject_id'] = df.section.map(subjects) if subjects is not None else 'unknown'
    for klass in ('lipid', 'polar', 'nucleotide', 'redox', 'ambiguous', 'unannotated'):
        counts = {name: int(((cls == klass) & ((np.asarray(a.uns['msi']) > 0).mean(0) > .2)).sum())
                  for name, a in zip(names, secs)}
        df['n_' + klass] = df.section.map(counts)
    df.to_csv(output,index=False)
    print("\n==== MAGPIE LUNG VALIDATION SUMMARY ====")
    for mode in df["mode"].unique():
        sub=df[df["mode"]==mode]
        print(f"{mode}: overall median={sub['overall'].mean():+.3f} | lipid={sub['lipid'].mean():+.3f} "
              f"polar={sub['polar'].mean():+.3f} nucleotide={sub['nucleotide'].mean():+.3f} redox={sub['redox'].mean():+.3f} (n={len(sub)} sections)")
    print(f"saved {output}")

if __name__=="__main__":
    ap = argparse.ArgumentParser(description='MAGPIE validation with explicit alignment provenance')
    ap.add_argument('cohort')
    ap.add_argument('--build-dir', required=True, help='new directory; existing data are never overwritten')
    ap.add_argument('--out', required=True, help='new CSV result path')
    ap.add_argument('--orientations', help='JSON: section -> {orientation: 0..7, source: description}')
    ap.add_argument('--subjects', help='JSON: section -> independent animal ID')
    ap.add_argument('--allow-outcome-selected-alignment', action='store_true', help='exploratory only; does not establish independent validation')
    args = ap.parse_args()
    if os.path.exists(args.out):
        ap.error('output already exists')
    orientations = json.load(open(args.orientations)) if args.orientations else None
    subjects = json.load(open(args.subjects)) if args.subjects else None
    built = build(args.cohort, args.build_dir, orientations, args.allow_outcome_selected_alignment)
    validate(built, args.out, subjects)
