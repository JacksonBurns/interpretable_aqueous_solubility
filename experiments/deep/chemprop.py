import subprocess
from pathlib import Path

import pandas as pd


def fit_chemprop(df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS"):
    tempfile = Path("temp.csv")
    df[[smiles_col, target_col]].to_csv(tempfile, index=False)
    subprocess.run(
        [
            "chemprop", "train",
            "--output-dir", "chemprop_training",
            "--logfile", "chemprop_training/log.txt",
            "--data-path", str(tempfile),
            "--split", "random",
            "--split-sizes", "0.80", "0.20", "0.00",
            "--num-replicates", "3",
            "--pytorch-seed", "42",
            "--smiles-columns", smiles_col,
            "--target-columns", target_col,
            "--task-type", "regression",
            "--patience", "5",
            "--loss", "mse",
            "--metrics", "rmse", "r2", "mse", "mae",
            "--show-individual-scores",
            "--epochs", "50",
            "--remove-checkpoints",
        ],
        check=True
    )
    tempfile.unlink()

    def predictor(df_new: pd.DataFrame):
        df_new[[smiles_col, target_col]].to_csv(tempfile, index=False)
        output_tempfile = Path("temp_preds.csv")
        subprocess.run(
            [
                "chemprop", "predict", "--model-path", "chemprop_training", "--preds-path", str(output_tempfile), "--test-path", str(tempfile), "--smiles-column", smiles_col,
            ]
        )
        tempfile.unlink()
        y_hat = pd.read_csv(output_tempfile)[target_col].values
        output_tempfile.unlink()
        Path("temp_preds_individual.csv").unlink()
        return pd.Series(y_hat, index=df_new.index)

    return predictor, None
