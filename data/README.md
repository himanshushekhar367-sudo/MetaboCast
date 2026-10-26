# Datasets

MetaboCast is trained and evaluated on public paired spatial transcriptomics and mass-spectrometry-imaging (MSI)
data. Only the gene-set files in `genesets/` are bundled here; download the datasets from their archives.

The reproduction scripts find the data through `reproduce/paths.py`: put the datasets under one folder and set
`METABOCAST_DATA_ROOT` to it (default `data/raw`), or set `DESIUM_DIR` for the DESIUM sections directly.

## DESIUM: DESI-MSI and Visium, human breast and lung cancer

- **Archive:** Harvard Dataverse, [10.7910/DVN/GZFCWC](https://doi.org/10.7910/DVN/GZFCWC).
- **Reference:** Godfrey T. M. et al. *Angew. Chem. Int. Ed.* **64**, e202502028 (2025).
- **Contents:** seven paired sections (breast BC_515 sections 1 and 2, BC_525, BC_823; lung LC_091, LC_170,
  LC_276), 14,816 spots and one shared axis of 2,086 negative-mode m/z channels. The files in
  `Correlation_NMF_analysis/data/*.h5ad` are already in MetaboCast format (`uns["msi"]`, `uns["mz_features"]`,
  `obsm["spatial"]`).
- **Train and benchmark:** `python pipeline/train_and_evaluate.py /path/to/DESIUM/Correlation_NMF_analysis/data`

## SMA: MALDI-MSI and Visium, mouse and human brain

- **Archive:** Mendeley Data, [10.17632/w7nw4km7xd.1](https://doi.org/10.17632/w7nw4km7xd.1)
  ("Spatial Multimodal Analysis of Transcriptomes and Metabolomes in Tissues": `metadata.csv`, `sma.zip`,
  `misc.zip` and other archives).
- **Reference:** Vicari M. et al. *Nat. Biotechnol.* **42**, 1046-1050 (2024),
  [10.1038/s41587-023-01937-y](https://doi.org/10.1038/s41587-023-01937-y).
- **Contents:** 13 sections across three MALDI chemistries: DHB (lipids; 2 mouse sections), 9-AA (small
  metabolites; 1 mouse section) and FMP-10 (neurotransmitters; 6 mouse and 4 human sections) in striatum and
  substantia nigra. Each section has a Space Ranger `outs/` folder (`sma.zip`) and a MALDI raster (`*_smamsi.csv`).
- **Subjects** (from `metadata.csv`; also in `pipeline/sma_published_pairs/sections.csv`): the DHB and 9-AA
  sections come from one mouse (mPD3); the six FMP-10 mouse sections come from three mice (mPD1, mPD3, mPD4), each
  with one striatum (slide V11L12-109) and one substantia nigra section (V11T16-085); the four human sections
  (V11T17-102) come from one donor (hPD1).
- **Registration:** every analysis uses the spot-to-pixel pairing published by the SMA authors, who aligned the MALDI
  and Visium images by hand in STUtility and paired each spot with its nearest pixel. It is stored in
  `pipeline/sma_published_pairs/<section>.csv` (barcode, raster x, raster y) and was exported from `misc.zip`
  (`misc/knn_spatial_df_filtered_list` for the mouse sections; `misc/hPDStr.multi.list`, matched to the raster by
  identical spectra, for the human sections) with
  ```bash
  Rscript pipeline/sma_published_pairs.R /path/to/misc <export dir>            # needs SeuratObject
  python pipeline/sma_published_pairs.py --sma /path/to/SMA --export <export dir> --metadata /path/to/metadata.csv
  ```
  The pairing was fixed before any prediction was evaluated. The converter can also register a section by tissue
  geometry alone (`--pairs none`); it reports how well that agrees with the published pairing.
- **Convert and train:**
  ```bash
  python pipeline/sma_to_metabocast.py /path/to/SMA          # writes /path/to/SMA/sma_h5ad/*.h5ad and a QC report
  python pipeline/train_and_evaluate.py /path/to/SMA/sma_h5ad
  ```
  `reproduce/sma_registration_audit.py` compares the published pairing with geometry registration and with an
  orientation chosen by maximising transcriptome-metabolite coupling (the sma_registration table and SuppFig_SMA_registration; numbers in `reproduce/manifest.py`). Coupling
  does not identify the right orientation and is not used for registration.

## MAGPIE: DESI-MSI and Visium, mouse lung injury (not used)

- **Archive:** Zenodo, [10.5281/zenodo.17789448](https://doi.org/10.5281/zenodo.17789448) (MAGPIE inputs).
- **Reference:** Williams E. C. et al. *Nat. Commun.* **17**, 205 (2026),
  [10.1038/s41467-025-68003-w](https://doi.org/10.1038/s41467-025-68003-w).
- The published MSI-to-Visium registration cannot be recovered from the public inputs, and on SMA, where the
  published pairing is known, choosing the orientation by transcriptome-metabolite coupling fails. These data are
  therefore not used in the manuscript. `reproduce/magpie_lung_validation.py` is kept for users who have the
  published registration (`--orientations`).

## Transcriptome-only applications

Files are read from `METABOCAST_DATA_ROOT` by `reproduce/crosscancer/regenerate_crosscancer.py`.

- **Hepatocellular carcinoma:** five Visium samples from Wu R. et al. *Sci. Adv.* **7**, eabg3750 (2021), GSA
  accession HRA000437, as `HCC-<n>-expr_counts.mtx.gz`, `HCC-<n>-expr_features.txt`, `HCC-<n>-expr_barcodes.txt`
  and `HCC-<n>-expr_meta.csv` (n = 1-5; the metadata carry the region labels and spot coordinates).
- **Glioblastoma:** two 10x Genomics public datasets (CC BY 4.0):
  - "Visium CytAssist Gene and Protein Expression Library of Human Glioblastoma, IF, 11mm (FFPE)" (Space Ranger
    2.1.0), [10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-glioblastoma-cytassist-ffpe-2-standard](https://www.10xgenomics.com/datasets/gene-and-protein-expression-library-of-human-glioblastoma-cytassist-ffpe-2-standard):
    `CytAssist_FFPE_Protein_Expression_Human_Glioblastoma_filtered_feature_bc_matrix.h5` and `..._spatial.tar.gz`;
  - "Human Glioblastoma: Whole Transcriptome Analysis" (Space Ranger 1.2.0, fresh frozen),
    [10xgenomics.com/datasets/human-glioblastoma-whole-transcriptome-analysis-1-standard-1-2-0](https://www.10xgenomics.com/datasets/human-glioblastoma-whole-transcriptome-analysis-1-standard-1-2-0):
    `Parent_Visium_Human_Glioblastoma_filtered_feature_bc_matrix.h5` and `..._spatial.tar.gz`.

## Bundled gene sets (`genesets/`)

| File | Source |
|---|---|
| `scMetab_KEGG.gmt` | KEGG metabolic gene sets as distributed with scMetabolism (85) |
| `scMetab_REACT.gmt` | Reactome metabolic gene sets as distributed with scMetabolism |
| `scFEA_mod.csv`, `scFEA_cmpd.csv` | scFEA metabolic module and compound tables |

`MetaboCast(use_kegg=True)` uses the KEGG sets shipped inside the package; pass `kegg_gmt=` to use another file.
