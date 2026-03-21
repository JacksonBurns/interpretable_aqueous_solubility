chemprop train \
    --output-dir chemeleon_results \
    --logfile chemeleon_results/log.txt \
    --data-path data/chemeleon_aqsoldbc_filtered_train.csv \
    --split random \
    --split-sizes 0.80 0.20 0.00 \
    --from-foundation CheMeleon \
    --pytorch-seed 42 \
    --smiles-columns SMILES \
    --target-columns logS \
    --task-type regression \
    --patience 5 \
    --loss mse \
    --metrics rmse r2 mse mae \
    --show-individual-scores \
    --ffn-num-layers 1 \
    --ffn-hidden-dim 2048 \
    --batch-size 32 \
    --epochs 50

chemprop predict \
    --model-path chemeleon_results/model_0/best.pt \
    --preds-path chemeleon_results/biogen_pred.csv \
    --test-path data/biogen.csv \
    --smiles-column SMILES

chemprop predict \
    --model-path chemeleon_results/model_0/best.pt \
    --preds-path chemeleon_results/ochem_pred.csv \
    --test-path data/ochem.csv \
    --smiles-column SMILES

chemprop predict \
    --model-path chemeleon_results/model_0/best.pt \
    --preds-path fatty_acid_demo/chemeleon_pred.csv \
    --test-path fatty_acid_demo/ralston_hoerr_joc_1942.csv \
    --smiles-column SMILES

chemprop predict \
    --model-path chemeleon_results/model_0/best.pt \
    --preds-path ancenes_demo/chemeleon_pred.csv \
    --test-path ancenes_demo/data.csv \
    --smiles-column SMILES
