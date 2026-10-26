# Model selection (the model_selection table and panel e of SuppFig_robustness; numbers in `reproduce/manifest.py`)

These scripts screen the design choices of the MetaboCast estimator. Every variant is scored with the manuscript
protocols, and every setting (penalty, target transform, ensemble weights) is chosen on training data only; held-out
labels are used for scoring and nothing else. Choices were made on DESIUM and confirmed on the three SMA cohorts,
which use the published spot-to-pixel pairing (`pipeline/sma_published_pairs/`).
Each run writes `<out>/loso_summary.json` (or `within_summary.json`); the summaries behind the manuscript are
stored in `reproduce/results/model_selection/` as `<run>.json`.

Data locations come from `reproduce/paths.py` (`DESIUM_DIR`, `METABOCAST_DATA_ROOT`); the SMA folder is
`$METABOCAST_DATA_ROOT/SMA/sma_h5ad` unless `SMA_DIR` is set. The readers accept h5ad string columns written by any
AnnData version (plain, nullable-string and categorical encodings).

| Script | What it does | Result |
|---|---|---|
| `screen_lib.py` | shared leakage-safe helpers: slim h5ad reader, diffusion features, exact ridge from per-section sufficient statistics, rank-Gaussian targets, ExtraTrees on target principal components | |
| `loso_screen2.py` | leave-one-section-out screen of representations (PCA with 50, 100 or 200 components, NMF, enzyme-gene block, per-section gene scaling, KEGG features, 0-2 diffusion hops) and target transforms | `loso2_desium.json` |
| `loso_round3.py` | rank targets, cross-validated penalty, jackknife bagging, ExtraTrees and the ridge + ExtraTrees ensemble | `r3_loso_*.json` |
| `within_round3.py` | the same variants under within-section spatial-block cross-validation | `r3_within_desium.json` |
| `loso_tic.py` | total-ion-count-normalised targets | `loso_tic_desium.json` |
| `combination_rule.py` | how the ridge and tree halves are combined, and which ions select the penalty, with the package code | `combo_*.json` |
| `screen_alpha.py` | penalty rule for single-section training (inner blocks scored one at a time or pooled), within-section protocol | `screen_alpha_*.json` |

Variant codes: `a200` fixed penalty α = 200; `aG` one penalty chosen by inner cross-validation; `aI` one penalty per
ion; `bag` jackknife-bagged ridge; `y` per-ion standardised log1p targets; `yR` rank-Gaussian targets per training
section; `yS` neighbour-smoothed targets; `wc` within-section centring; `ET` ExtraTrees on 50 target principal
components; `ens(ridge aG,ET)` the equal-spread ensemble used by MetaboCast; `secz` / `secrank` per-section z-scored
or ranked genes; `pca+enz` PCA plus a KEGG metabolic-gene block; `pca50` / `pca200` PCA size; `pca_h0` / `pca_h1`
diffusion hops. In `screen_alpha_*.json`, `fixed200_logtargets` is a ridge with α = 200 on standardised log1p targets
without trees, and score keys end in `|log` (log1p-intensity scale) or `|tgt` (training-target scale).

## Commands

```bash
cd reproduce/model_selection
# DESIUM leave-one-section-out (2,000 highly variable genes pooled over the sections)
python loso_screen2.py --out out/loso2_desium
python loso_round3.py  --out out/r3_loso_desium --sets pca,pca+enz,pca200,secz,secrank \
       --target-sets pca,pca+enz,pca200,secz,secrank --targets y,yR --et-targets yR --et-feats pca,secz
# SMA confirmation cohorts (highly variable genes from the training sections, n_hvg = 3,000 as in the package)
python loso_round3.py  --out out/r3_loso_sma_dhb --hvg train --n-hvg 3000 --sets pca,pca+enz,secz,secrank \
       --target-sets pca,pca+enz,secz,secrank --targets y,yR --et-targets yR --et-feats pca,secz \
       --sections SMA_DHB_V11L12-038_A1 SMA_DHB_V11L12-038_B1
#   (same flags with the six mouse or four human FMP sections for r3_loso_sma_fmp_mouse and r3_loso_sma_fmp_human)
# within-section, DESIUM
python within_round3.py --out out/r3_within_desium --sets pca,pca+enz,secz --targets y,yR
# total-ion-count-normalised targets
python loso_tic.py --out out/loso_tic_desium --sets pca --target-sets pca,nmf --targets ytic,yRtic --et-sets ""
# combination rule and penalty-selection ions, with the package estimator
python combination_rule.py loso "$DESIUM_DIR" BC_515_Section_1,BC_515_Section_2,BC_525,BC_823,LC_091,LC_170,LC_276 out/combo_loso_desium.json
# penalty rule for single-section training (within-section protocol)
python screen_alpha.py "$SMA_DIR" SMA_9AA_V11L12-038_D1,SMA_DHB_V11L12-038_A1,SMA_DHB_V11L12-038_B1,SMA_FMP_V11T16-085_A1,SMA_FMP_V11L12-109_A1,SMA_FMP_V11T17-102_A1 out/screen_alpha_sma.json
python screen_alpha.py "$DESIUM_DIR" BC_515_Section_2,BC_823,LC_091,LC_276 out/screen_alpha_desium.json
```

Within-section rows of the screen are scored on each variant's training-target scale (the rank scale for rank
targets) and are not directly comparable with intensity-scale rows; the benchmark table holds the like-for-like
comparison from the released package (`reproduce/within_benchmark.py`).
