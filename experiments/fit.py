from itertools import chain

import numpy as np
import pandas as pd
from rdkit.Chem import CanonSmiles

from interpretable.symantic import fit_symantic
from interpretable.pysr import fit_pysr
from baseline.esol import fit_esol
from baseline.rf import fit_rf
from deep.chemeleon import fit_chemeleon


if __name__ == "__main__":
    # data loading
    _train_df = pd.read_csv("../data/aqsoldbc.csv")
    _train_df["SMILES"] = _train_df["SMILES"].map(CanonSmiles)
    # remove overlap with test sets
    _biogen_df = pd.read_csv("../data/biogen.csv")
    _biogen_df["SMILES"] = _biogen_df["SMILES"].map(CanonSmiles)
    _ochem_df = pd.read_csv("../data/ochem.csv")
    _ochem_df["SMILES"] = _ochem_df["SMILES"].map(CanonSmiles)
    # remove overlap for demos as well
    _ancenes_df = pd.read_csv("../data/ancenes.csv")
    _ancenes_df["SMILES"] = _ancenes_df["SMILES"].map(CanonSmiles)
    _fatty_acids_df = pd.read_csv("../data/ralston_hoerr_joc_1942.csv")
    _fatty_acids_df["SMILES"] = _fatty_acids_df["SMILES"].map(CanonSmiles)

    test_smiles = set(chain(_biogen_df["SMILES"], _ochem_df["SMILES"], _ancenes_df["SMILES"], _fatty_acids_df["SMILES"]))
    _train_df = _train_df[~_train_df["SMILES"].isin(test_smiles)].reset_index(drop=True)


    # training and inference
    downsample_sizes = np.logspace(
        1,  # i.e. 10**1 = 10
        np.log10(_train_df.shape[0]),  # i.e. the full dataset size
        num=8,
        base=10,
        dtype=int,
    )
    repetitions = 5
    logfile = "fit_results.txt"
    for n in downsample_sizes:
        print(f"\n\n=== Training models on downsampled dataset of size {n} ===")
        for rep in range(repetitions):
            print(f"\n--- Repetition {rep+1}/{repetitions} for downsample size {n} ---")
            train_df = _train_df.sample(n=n, random_state=rep).reset_index(drop=True)
            ochem_df = _ochem_df.copy()
            biogen_df = _biogen_df.copy()

            pred_str = "_pred" if n == _train_df.shape[0] else f"_{n}_{rep}_pred"

            # pysr
            (f_pysr_utopia, f_pysr_greedy), (pysr_utopia_eqn, pysr_greedy_eqn) = fit_pysr(train_df.copy())
            with open(logfile, "a") as f:
                f.write(f"Downsample size: {n}, Repetition: {rep+1}\n")
                f.write(f"PySR utopia equation: {pysr_utopia_eqn}\n")
                f.write(f"PySR greedy equation: {pysr_greedy_eqn}\n")
                f.write("\n")
            print("PySR utopia equation:", pysr_utopia_eqn)
            print("PySR greedy equation:", pysr_greedy_eqn)
            biogen_df["pysr_utopia" + pred_str], _ = f_pysr_utopia(biogen_df)
            ochem_df["pysr_utopia" + pred_str], _ = f_pysr_utopia(ochem_df)
            biogen_df["pysr_greedy" + pred_str], _ = f_pysr_greedy(biogen_df)
            ochem_df["pysr_greedy" + pred_str], _ = f_pysr_greedy(ochem_df)

            # symantic
            (f_symantic_utopia, f_symantic_greedy), (symantic_utopia_eqn, symantic_greedy_eqn) = fit_symantic(train_df.copy())
            with open(logfile, "a") as f:
                f.write(f"Downsample size: {n}, Repetition: {rep+1}\n")
                f.write(f"SyMANTIC utopia equation: {symantic_utopia_eqn}\n")
                f.write(f"SyMANTIC greedy equation: {symantic_greedy_eqn}\n")
                f.write("\n")
            print("SyMANTIC utopia equation:", symantic_utopia_eqn)
            print("SyMANTIC greedy equation:", symantic_greedy_eqn)
            biogen_df["symantic_utopia" + pred_str], _ = f_symantic_utopia(biogen_df)
            ochem_df["symantic_utopia" + pred_str], _ = f_symantic_utopia(ochem_df)
            biogen_df["symantic_greedy" + pred_str], _ = f_symantic_greedy(biogen_df)
            ochem_df["symantic_greedy" + pred_str], _ = f_symantic_greedy(ochem_df)

            # esol
            f_esol, esol_eqn = fit_esol(train_df.copy())
            with open(logfile, "a") as f:
                f.write(f"Downsample size: {n}, Repetition: {rep+1}\n")
                f.write(f"ESOL refitted equation: {esol_eqn}\n")
                f.write("\n")
            print("ESOL refitted equation:", esol_eqn)
            biogen_df["esol" + pred_str], esol_biogen_features = f_esol(biogen_df)
            ochem_df["esol" + pred_str], esol_ochem_features = f_esol(ochem_df)
            
            # rf
            f_rf, _ = fit_rf(train_df.copy())
            biogen_df["rf" + pred_str], _ = f_rf(biogen_df)
            ochem_df["rf" + pred_str], _ = f_rf(ochem_df)
            
            # chemeleon
            f_chemeleon, _ = fit_chemeleon(train_df.copy())
            biogen_df["chemeleon" + pred_str], _ = f_chemeleon(biogen_df)
            ochem_df["chemeleon" + pred_str], _ = f_chemeleon(ochem_df)

            print("debugging run, exiting"); exit(1)
    
    # print and save results
    print(ochem_df)
    print(biogen_df)
    ochem_df.to_csv("ochem_pred.csv", index=False)
    biogen_df.to_csv("biogen_pred.csv", index=False)
