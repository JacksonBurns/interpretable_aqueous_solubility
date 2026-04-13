import re
from typing import Literal

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from mordred import Calculator, descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles, Descriptors


def _add_features(df: pd.DataFrame, smiles_col: str = "SMILES", feature_set: Literal["rdkit", "mordred"] = "rdkit", means: np.ndarray | None = None):
    if feature_set == "mordred":
        calc = Calculator(descriptors, ignore_3D=True)
        descs = calc.pandas(mols=[MolFromSmiles(s) for s in df[smiles_col]]).fill_missing()
        # retain only Only alphanumeric characters, numbers, and underscores in column names for compatibility with PySR
        descs.columns = [re.sub(r'[^\w]+', '_', col) for col in descs.columns]
        # suffix with _mordred to avoid conflicts with julia vars
        descs.columns = [col + "_mordred" for col in descs.columns]
    else:
        names = [x[0] for x in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        data = [calc.CalcDescriptors(MolFromSmiles(smiles)) for smiles in df[smiles_col]]
        descs = pd.DataFrame(columns=calc.GetDescriptorNames(), data=data)
    # imputation
    descs = descs.astype(float)
    descs = descs.replace([np.inf, -np.inf], np.nan)
    if means is None:
        # drop columns with zero variance (do this only when scaling for the first time)
        descs = descs.loc[:, descs.nunique() > 1]
        means = descs.mean(axis=0, skipna=True)
    descs = descs.fillna(means)
    return descs, means


def fit_pysr(df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS"):
    """
    Fit a PySR (Symbolic Regression) model using RDKit/Mordred descriptors.
    """
    input_df, means = _add_features(df, smiles_col)

    model = PySRRegressor(
        niterations=50,  
        populations=30,  
        population_size=100,  
        binary_operators=["+", "-", "*", "/"],
        loss="L2DistLoss()",  
        parsimony=0.002,  
        maxsize=20,  
        maxdepth=10,  
        turbo=True,  
        bumper=True,  
        precision=64,  
        random_state=42,
        parallelism='serial',
        deterministic=True,
        progress=True,
        verbosity=1,
        temp_equation_file=True,
    )
    model.fit(input_df, df[target_col])
    utopia_eqn = str(model.sympy())
    # extract the absolute lowest error, least interpretable

    model = PySRRegressor(
        model_selection='accuracy',  # damn the interpretability, full steam ahead
        niterations=50,  
        populations=30,  
        population_size=100,  
        binary_operators=["+", "-", "*", "/"],
        loss="L2DistLoss()",  
        parsimony=0.002,  
        maxsize=20,  
        maxdepth=10,  
        turbo=True,  
        bumper=True,  
        precision=64,  
        random_state=42,
        parallelism='serial',
        deterministic=True,
        progress=True,
        verbosity=1,
        temp_equation_file=True,
    )
    model.fit(input_df, df[target_col])
    greedy_eqn = str(model.sympy())

    def utopia_predictor(df_new: pd.DataFrame):
        df_new_features, _ = _add_features(df_new, smiles_col, means=means)        
        return df_new_features.eval(utopia_eqn)

    def greedy_predictor(df_new: pd.DataFrame):
        df_new_features, _ = _add_features(df_new, smiles_col, means=means)
        return df_new_features.eval(greedy_eqn)

    return (utopia_predictor, greedy_predictor), (utopia_eqn, greedy_eqn)
