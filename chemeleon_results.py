import pandas as pd
from rdkit.Chem import CanonSmiles

from main import parity_plot

if __name__ == "__main__":
    train_smiles = [
        CanonSmiles(smi) for smi in pd.read_csv("chemeleon_aqsoldbc_filtered_train.csv")["SMILES"]
    ]
    for f in ("biogen", "ochem"):
        true_df = pd.read_csv(f"{f}.csv")
        true_df = true_df[~true_df["SMILES"].isin(train_smiles)]
        pred_df = pd.read_csv(f"chemeleon_results/{f}_pred.csv")
        pred_df = pred_df[~pred_df["SMILES"].isin(train_smiles)]
        # drop overlap
        pred_arr = pred_df["logS"].to_numpy()
        true_arr = true_df["logS"].to_numpy()
        parity_plot(true_arr, pred_arr, f"results/chemeleon_{f}_full_parity")
        if f == "biogen":
            continue
        mask = (true_arr > -7) & (true_arr < -3)
        parity_plot(true_arr[mask], pred_arr[mask], f"results/chemeleon_{f}_subset_parity")
