cd ${SCFEA_REPO:-external/scFEA}/src
for s in BC_823 LC_170 BC_515_Section_2 LC_276 LC_091 BC_525 BC_515_Section_1; do
  python3 scFEA.py --data_dir ../data --input_dir ${METABOCAST_BASELINE_DIR:-$PWD/work/baselines}/input --res_dir out --test_file $s.csv \
    --moduleGene_file module_gene_m168.csv --stoichiometry_matrix cmMat_c70_m168.csv --cName_file cName_c70_m168.csv \
    --sc_imputation False --output_flux_file ${METABOCAST_BASELINE_DIR:-$PWD/work/baselines}/output/${s}_flux.csv --output_balance_file ${METABOCAST_BASELINE_DIR:-$PWD/work/baselines}/output/${s}_balance.csv --train_epoch 100 > ${METABOCAST_BASELINE_DIR:-$PWD/work/baselines}/output/${s}.log 2>&1
  echo "$s exit $?" >> ${METABOCAST_BASELINE_DIR:-$PWD/work/baselines}/output/status.txt
done
echo ALLDONE >> ${METABOCAST_BASELINE_DIR:-$PWD/work/baselines}/output/status.txt
