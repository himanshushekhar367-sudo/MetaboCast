#!/usr/bin/env python3
"""
Train a MetaboCast model on YOUR paired (spatial transcriptome + spatial metabolome) sections
and save a deployable model bundle (model + m/z axis + optional conformal uncertainty widths).

Each training section must be a MetaboCast-format .h5ad:
    adata.layers['log1p'] (or adata.X)   genes, log1p(CP10k)   (n_spots x n_genes)
    adata.uns['msi']                     metabolite intensities (n_spots x n_metab)
    adata.uns['mz_features']             m/z values             (n_metab,)
    adata.obsm['spatial']                x, y coordinates       (n_spots x 2)

USAGE
    python examples/train_on_paired_data.py --data /path/to/folder_of_h5ad --out my_model.pkl
    python examples/train_on_paired_data.py --data ./sections --out m.pkl --kegg data/genesets/scMetab_KEGG.gmt

Then predict on any transcriptome-only data with examples/predict_on_your_data.py.
"""
import argparse, glob, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, anndata as ad

# make `import metabocast` work whether or not the package is pip-installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metabocast
from metabocast import MetaboCast


def trainable(a):
    return ("msi" in a.uns) and ("mz_features" in a.uns) and ("spatial" in a.obsm) \
        and (("log1p" in a.layers) or (a.X is not None))


def main():
    ap = argparse.ArgumentParser(description="Train MetaboCast on paired sections.")
    ap.add_argument("--data", required=True, help="folder containing MetaboCast-format .h5ad sections")
    ap.add_argument("--out", default="metabocast_model.pkl", help="output model bundle (.pkl)")
    ap.add_argument("--kegg", default=None, help="optional scMetabolism .gmt for the KEGG pathway prior")
    ap.add_argument("--alpha", default="auto",
                    help='ridge penalty: "auto" (default; chosen by leave-one-section-out CV on the training data) or a number')
    ap.add_argument("--estimator", choices=["ensemble", "ridge"], default="ensemble",
                    help="ensemble (default): ridge + ExtraTrees; ridge: the linear model alone")
    ap.add_argument("--no-conformal", action="store_true", help="train on every section without uncertainty")
    ap.add_argument("--calibration-data", help="separate folder of independent calibration subjects, excluded from fitting")
    ap.add_argument("--calibration-unit", choices=['section', 'spot'], default='section',
                    help="section maxima by default; spot mode assumes exchangeable spots")
    args = ap.parse_args()
    if os.path.exists(args.out):
        ap.error('output model exists; choose a new --out path')

    paths = [p for p in sorted(glob.glob(os.path.join(args.data, "**", "*.h5ad"), recursive=True))
             if "training_ready" not in p]
    ads, names = [], []
    for p in paths:
        try:
            a = ad.read_h5ad(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}"); continue
        if trainable(a):
            ads.append(a); names.append(os.path.basename(p))
            print(f"  [ok] {os.path.basename(p)}: {a.n_obs} spots x {a.n_vars} genes | "
                  f"{np.asarray(a.uns['msi']).shape[1]} metabolites")
    if not ads:
        sys.exit("No MetaboCast-format .h5ad found. See docs/USAGE.md for the required fields.")
    # all sections must share one metabolite (m/z) axis, compared by value, to be pooled
    try:
        metabocast.common_mz_axis(ads)
    except ValueError as e:
        sys.exit(f"{e}\nTrain each cohort separately (pipeline/train_and_evaluate.py groups them for you).")

    use_kegg = args.kegg is not None
    if args.no_conformal and args.calibration_data:
        ap.error('--no-conformal and --calibration-data cannot be combined')
    calibration = []
    if args.calibration_data:
        cal_paths = sorted(glob.glob(os.path.join(args.calibration_data, '**', '*.h5ad'), recursive=True))
        if set(map(os.path.realpath, cal_paths)) & set(map(os.path.realpath, paths)):
            ap.error('calibration files overlap training files')
        calibration = [ad.read_h5ad(p) for p in cal_paths]
        if not calibration:
            ap.error('no calibration sections found')
    elif not args.no_conformal:
        print('No independent calibration folder supplied: point predictions only.')

    alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
    kw = dict(alpha=alpha, estimator=args.estimator)
    print(f"\nTraining final model on all {len(ads)} sections ({kw}, KEGG={use_kegg}) ...")
    model = MetaboCast(use_kegg=use_kegg, kegg_gmt=args.kegg, **kw).fit(ads)
    print(f"  ridge penalty {model.alpha_:g} ({model.alpha_selection_})")
    if calibration:
        model.calibrate(calibration, unit=args.calibration_unit)
        print('Calibration uses the fixed fitted model; small section counts can produce infinite widths.')
    conf_width = model.conf_width_

    bundle = dict(model=model, mz=np.asarray(model.mz_, float), conf_width=conf_width,
                  genes=list(model.genes_), n_train_sections=len(ads), train_sections=names,
                  estimator=model.estimator, target_transform=model.target_transform, alpha=model.alpha_,
                  version=f"metabocast-{metabocast.__version__}")
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nsaved model bundle -> {args.out}")
    print(f"  metabolites (m/z): {len(bundle['mz'])} | training genes: {len(bundle['genes'])} | "
          f"conformal: {'yes' if conf_width is not None else 'no'}")
    print("  predict on any transcriptome-only data:")
    print(f"    python examples/predict_on_your_data.py --model {args.out} --query your_data.h5ad --out results/")


if __name__ == "__main__":
    main()
