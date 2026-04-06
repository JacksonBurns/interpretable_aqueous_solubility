import pandas as pd

from interpretable.symantic import fit_symantic
from interpretable.pysr import fit_pysr
from baseline.esol import fit_esol
from baseline.rf import fit_rf
from deep.chemeleon import fit_chemeleon
from residual.gp import fit_residual_gp


if __name__ == "__main__":
    from itertools import chain
    from rdkit.Chem import CanonSmiles

    # data loading
    train_df = pd.read_csv("../data/aqsoldbc.csv")
    train_df["SMILES"] = train_df["SMILES"].map(CanonSmiles)
    # remove overlap with test sets
    biogen_df = pd.read_csv("../data/biogen.csv")
    biogen_df["SMILES"] = biogen_df["SMILES"].map(CanonSmiles)
    ochem_df = pd.read_csv("../data/ochem.csv")
    ochem_df["SMILES"] = ochem_df["SMILES"].map(CanonSmiles)
    # remove overlap for demos as well
    ancenes_df = pd.read_csv("../data/ancenes.csv")
    ancenes_df["SMILES"] = ancenes_df["SMILES"].map(CanonSmiles)
    fatty_acids_df = pd.read_csv("../data/ralston_hoerr_joc_1942.csv")
    fatty_acids_df["SMILES"] = fatty_acids_df["SMILES"].map(CanonSmiles)
    
    test_smiles = set(chain(biogen_df["SMILES"], ochem_df["SMILES"], ancenes_df["SMILES"], fatty_acids_df["SMILES"]))
    train_df = train_df[~train_df["SMILES"].isin(test_smiles)].reset_index(drop=True)

    # training and inference

    # symantic
    # known equation provided from run on machine with ~256 GB of memory - to run from scratch on a machine with less memory, use downsample_size and remove known_equation argument
    f_symantic, symantic_eqn = fit_symantic(
        train_df.copy(),
        known_equation="0.0367*((MolLogP*LabuteASA)/(MolLogP-Chi0v)) + 499.497*((Chi0v+Chi0n)/(LabuteASA)**2) -2.863",
    )
    print("SyMANTIC equation:", symantic_eqn)
    biogen_df["symantic_pred"], symantic_biogen_features = f_symantic(biogen_df)
    ochem_df["symantic_pred"], symantic_ochem_features = f_symantic(ochem_df)
    
    # symanticgp
    symantic_aqsoldbc_pred, symantic_aqsoldbc_features = f_symantic(train_df.copy())
    f_symantic_gp, _ = fit_residual_gp(symantic_aqsoldbc_features, train_df["logS"], symantic_aqsoldbc_pred)
    biogen_df["symanticgp_pred"], _ = f_symantic_gp(symantic_biogen_features, biogen_df["symantic_pred"])
    ochem_df["symanticgp_pred"], _ = f_symantic_gp(symantic_ochem_features, ochem_df["symantic_pred"])

    # pysr
    f_pysr, pysr_eqn = fit_pysr(train_df.copy())
    print("PySR equation:", pysr_eqn)
    biogen_df["pysr_pred"], _ = f_pysr(biogen_df)
    ochem_df["pysr_pred"], _ = f_pysr(ochem_df)
    
    # esol
    f_esol, esol_eqn = fit_esol(train_df.copy())
    print("ESOL refitted equation:", esol_eqn)
    biogen_df["esol_pred"], esol_biogen_features = f_esol(biogen_df)
    ochem_df["esol_pred"], esol_ochem_features = f_esol(ochem_df)
    
    # esolgp
    esol_aqsoldbc_pred, esol_aqsoldbc_features = f_esol(train_df.copy())
    f_esol_gp, _ = fit_residual_gp(esol_aqsoldbc_features, train_df["logS"], esol_aqsoldbc_pred)
    biogen_df["esolgp_pred"], _ = f_esol_gp(esol_biogen_features, biogen_df["esol_pred"])
    ochem_df["esolgp_pred"], _ = f_esol_gp(esol_ochem_features, ochem_df["esol_pred"])
    
    # rf
    f_rf, _ = fit_rf(train_df.copy())
    biogen_df["rf_pred"], _ = f_rf(biogen_df)
    ochem_df["rf_pred"], _ = f_rf(ochem_df)
    
    # chemeleon
    f_chemeleon, _ = fit_chemeleon(train_df.copy())
    biogen_df["chemeleon_pred"], _ = f_chemeleon(biogen_df)
    ochem_df["chemeleon_pred"], _ = f_chemeleon(ochem_df)
    
    # print and save results
    print(ochem_df)
    print(biogen_df)
    ochem_df.to_csv("ochem_pred.csv", index=False)
    biogen_df.to_csv("biogen_pred.csv", index=False)
