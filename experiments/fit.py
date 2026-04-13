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
    downsample_sizes[-1] = _train_df.shape[0]
    repetitions = 5
    logfile = "fit_results.txt"
    for n in downsample_sizes:
        print(f"\n\n=== Training models on downsampled dataset of size {n} ===")
        for rep in range(repetitions):
            print(f"\n--- Repetition {rep+1}/{repetitions} for downsample size {n} ---")
            train_df = _train_df.sample(n=n, random_state=rep).reset_index(drop=True)
            ochem_df = _ochem_df.copy()
            biogen_df = _biogen_df.copy()

            pred_str = "_pred" if n == downsample_sizes[-1] else f"_{n}_{rep}_pred"

            # pysr
            (f_pysr_utopia, f_pysr_greedy), (pysr_utopia_eqn, pysr_greedy_eqn) = fit_pysr(train_df.copy())
            with open(logfile, "a") as f:
                f.write(f"Downsample size: {n}, Repetition: {rep+1}\n")
                f.write(f"PySR utopia equation: {pysr_utopia_eqn}\n")
                f.write(f"PySR greedy equation: {pysr_greedy_eqn}\n")
                f.write("\n")
            print("PySR utopia equation:", pysr_utopia_eqn)
            print("PySR greedy equation:", pysr_greedy_eqn)
            _biogen_df["pysr_utopia" + pred_str] = f_pysr_utopia(biogen_df)
            _ochem_df["pysr_utopia" + pred_str] = f_pysr_utopia(ochem_df)
            _biogen_df["pysr_greedy" + pred_str] = f_pysr_greedy(biogen_df)
            _ochem_df["pysr_greedy" + pred_str] = f_pysr_greedy(ochem_df)

            # symantic
            (f_symantic_utopia, f_symantic_greedy), (symantic_utopia_eqn, symantic_greedy_eqn) = fit_symantic(train_df.copy())
            with open(logfile, "a") as f:
                f.write(f"Downsample size: {n}, Repetition: {rep+1}\n")
                f.write(f"SyMANTIC utopia equation: {symantic_utopia_eqn}\n")
                f.write(f"SyMANTIC greedy equation: {symantic_greedy_eqn}\n")
                f.write("\n")
            print("SyMANTIC utopia equation:", symantic_utopia_eqn)
            print("SyMANTIC greedy equation:", symantic_greedy_eqn)
            _biogen_df["symantic_utopia" + pred_str] = f_symantic_utopia(biogen_df)
            _ochem_df["symantic_utopia" + pred_str] = f_symantic_utopia(ochem_df)
            _biogen_df["symantic_greedy" + pred_str] = f_symantic_greedy(biogen_df)
            _ochem_df["symantic_greedy" + pred_str] = f_symantic_greedy(ochem_df)

            # esol
            f_esol, esol_eqn = fit_esol(train_df.copy())
            with open(logfile, "a") as f:
                f.write(f"Downsample size: {n}, Repetition: {rep+1}\n")
                f.write(f"ESOL refitted equation: {esol_eqn}\n")
                f.write("\n")
            print("ESOL refitted equation:", esol_eqn)
            _biogen_df["esol" + pred_str] = f_esol(biogen_df)
            _ochem_df["esol" + pred_str] = f_esol(ochem_df)
            
            # rf
            f_rf, _ = fit_rf(train_df.copy())
            _biogen_df["rf" + pred_str] = f_rf(biogen_df)
            _ochem_df["rf" + pred_str] = f_rf(ochem_df)
            
            # chemeleon
            f_chemeleon, _ = fit_chemeleon(train_df.copy())
            _biogen_df["chemeleon" + pred_str] = f_chemeleon(biogen_df)
            _ochem_df["chemeleon" + pred_str] = f_chemeleon(ochem_df)

            if n == downsample_sizes[-1]:
                break  # only run the full dataset once
    
    # print and save results
    print(_ochem_df)
    print(_biogen_df)
    _ochem_df.to_csv("ochem_pred.csv", index=False)
    _biogen_df.to_csv("biogen_pred.csv", index=False)
