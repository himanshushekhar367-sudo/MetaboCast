"""Leave-one-section-out transfer with the released MetaboCast class (Figs 5a and 6; transfer, sma_transfer
and kegg_prior tables). For each held-out section the model is fitted on the remaining sections of the same cohort
(same m/z axis and species) and scored on ions detected in >20% of the held-out spots (detection-matched
median Spearman). HVGs come from the training sections only.

With --hold-out subject every section of the held-out section's animal or donor (uns["sma"]["subject"], written by
pipeline/sma_to_metabocast.py) is left out of training, so no biological unit contributes to both sides.

Usage:
    python reproduce/loso_benchmark.py --data <dir> --sections S1 S2 ... --out results/loso_desium.json
        [--estimator ridge] [--use-kegg] [--maps NAME:MZ ...] [--hold-out subject]
Defaults are the package defaults (rank targets, cross-validated penalty, ridge + ExtraTrees ensemble);
--estimator ridge drops the ExtraTrees half and --use-kegg adds the KEGG prior (kegg_prior table).
Writes the JSON summary and <out>_per_ion.npz (per-ion held-out Spearman per section, NaN = not detected).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metabocast import MetaboCast   # noqa: E402


def slim(a):
    """Keep only what fitting and scoring need (log1p expression as sparse float32, MSI, m/z, coordinates)."""
    from scipy.sparse import csr_matrix
    x = a.layers['log1p'] if 'log1p' in a.layers else a.X
    b = ad.AnnData(X=csr_matrix(x, dtype=np.float32))
    b.var_names = list(a.var_names); b.obs_names = list(a.obs_names)
    b.layers['log1p'] = b.X
    b.uns['msi'] = np.asarray(a.uns['msi'], np.float32)
    b.uns['mz_features'] = np.asarray(a.uns['mz_features'], float)
    b.obsm['spatial'] = np.asarray(a.obsm['spatial'], float)
    return b


def col_spearman(P, Y):
    R1 = rankdata(P, axis=0); R2 = rankdata(Y, axis=0)
    R1 -= R1.mean(0); R2 -= R2.mean(0)
    den = np.sqrt((R1 * R1).sum(0) * (R2 * R2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (R1 * R2).sum(0) / den
    r[~(den > 1e-12)] = np.nan
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--sections', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-hvg', type=int, default=3000)
    ap.add_argument('--estimator', choices=['ensemble', 'ridge'], default='ensemble')
    ap.add_argument('--use-kegg', action='store_true')
    ap.add_argument('--maps', nargs='*', default=[], help='name:m/z of held-out predicted maps to save')
    ap.add_argument('--upper-genes', action='store_true',
                    help='upper-case gene symbols (lets the human KEGG sets match mouse symbols case-insensitively)')
    ap.add_argument('--hold-out', choices=['section', 'subject'], default='section',
                    help="unit left out of training: the held-out section only, or every section of its subject")
    args = ap.parse_args()
    secs, subject = {}, {}
    for s in args.sections:
        a = ad.read_h5ad(args.data / f'{s}.h5ad')
        subject[s] = str(dict(a.uns.get('sma', {})).get('subject', '') or s)
        secs[s] = slim(a)
    if args.hold_out == 'subject' and len(set(subject.values())) < 2:
        sys.exit('--hold-out subject needs sections from at least two subjects')
    if args.upper_genes:
        for a in secs.values():
            a.var_names = [g.upper() for g in a.var_names]
            a.var_names_make_unique()
    kw = dict(n_hvg=args.n_hvg, use_kegg=args.use_kegg, estimator=args.estimator)
    res, per_ion, maps = {}, {}, {}
    with threadpool_limits(limits=1):
        for held in args.sections:
            t0 = time.time()
            tr = [secs[s] for s in args.sections
                  if (subject[s] != subject[held] if args.hold_out == 'subject' else s != held)]
            m = MetaboCast(**kw).fit(tr)
            q = secs[held].copy()
            pred = m.predict_metabolome(q)
            raw = np.asarray(secs[held].uns['msi'], float)
            y = np.log1p(raw); det = (raw > 0).mean(0) > 0.2
            ok = det & (y.std(0) > 1e-9) & (pred.std(0) > 1e-9)
            r = np.full(y.shape[1], np.nan, np.float32); r[ok] = col_spearman(pred[:, ok], y[:, ok])
            per_ion[held] = r
            res[held] = dict(median=float(np.nanmedian(r[det])), n_detected=int(det.sum()), subject=subject[held],
                             n_train_sections=len(tr), alpha=float(m.alpha_), alpha_selection=m.alpha_selection_,
                             seconds=round(time.time() - t0))
            mz = np.asarray(secs[held].uns['mz_features'], float)
            for spec in args.maps:
                nm, target = spec.split(':'); j = int(np.argmin(np.abs(mz - float(target))))
                if abs(mz[j] - float(target)) <= 0.05:
                    maps[f'{held}__{nm}__measured'] = y[:, j]; maps[f'{held}__{nm}__predicted'] = pred[:, j]
                    maps[f'{held}__{nm}__coords'] = np.asarray(secs[held].obsm['spatial'], float)
                    maps[f'{held}__{nm}__r'] = np.array([r[j]])
            print(held, res[held], flush=True)
    vals = [v['median'] for v in res.values()]
    summary = dict(per_section=res, mean=float(np.mean(vals)), sd=float(np.std(vals)),
                   config=dict(kw, upper_genes=args.upper_genes, hold_out=args.hold_out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    np.savez_compressed(str(args.out).replace('.json', '_per_ion.npz'), **per_ion)
    if maps:
        np.savez_compressed(str(args.out).replace('.json', '_maps.npz'), **maps)
    print(f'mean {summary["mean"]:.4f} +- {summary["sd"]:.4f}; saved {args.out}')


if __name__ == '__main__':
    main()
