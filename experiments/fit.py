import numpy as np
import pandas as pd

from interpretable.symantic import fit_symantic
from baseline.esol import fit_esol
from baseline.rf import fit_rf
from deep.chemeleon import fit_chemeleon



if __name__ == "__main__":
    from itertools import chain
    from rdkit.Chem import CanonSmiles

    # data loading
    train_df = pd.read_csv("../data/aqsoldbc.csv")
    train_df["SMILES"] = train_df["SMILES"].map(CanonSmiles)
    biogen_df = pd.read_csv("../data/biogen.csv")
    biogen_df["SMILES"] = biogen_df["SMILES"].map(CanonSmiles)
    ochem_df = pd.read_csv("../data/ochem.csv")
    ochem_df["SMILES"] = ochem_df["SMILES"].map(CanonSmiles)
    
    test_smiles = set(chain(biogen_df["SMILES"], ochem_df["SMILES"]))
    train_df = train_df[~train_df["SMILES"].isin(test_smiles)].reset_index()

    # TODO: debugging only
    # train_df = train_df.sample(n=100).reset_index()

    # TODO: statistical comparison
    f_esol, esol_eqn = fit_esol(train_df.copy())  # copy, just to be safe
    print("ESOL refitted equation:", esol_eqn)
    f_rf, _ = fit_rf(train_df.copy())
    f_chemeleon, _ = fit_chemeleon(train_df.copy())
    f_symantic, symantic_eqn = fit_symantic(train_df.copy())
    print("SyMANTIC equation:", symantic_eqn)
    # pass symantic eqn to or predictor to the GP model?... could fit a residual GP on all of these approaches...

    # now run inference
    biogen_df["esol_pred"] = f_esol(biogen_df)
    biogen_df["rf_pred"] = f_rf(biogen_df)
    biogen_df["chemeleon_pred"] = f_chemeleon(biogen_df)
    biogen_df["symantic_pred"] = f_symantic(biogen_df)
    ochem_df["esol_pred"] = f_esol(ochem_df)
    ochem_df["rf_pred"] = f_rf(ochem_df)
    ochem_df["chemeleon_pred"] = f_chemeleon(ochem_df)
    ochem_df["symantic_pred"] = f_symantic(ochem_df)
    print(ochem_df)
    print(biogen_df)
    ochem_df.to_csv("ochem_pred.csv", index=False)
    biogen_df.to_csv("biogen_pred.csv", index=False)
