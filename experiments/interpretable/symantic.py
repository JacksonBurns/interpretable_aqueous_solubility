from typing import Literal

from .src.model import SymanticModel
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles, Descriptors

import re

OPERATORS = ['+', '-']


def _add_features(df: pd.DataFrame, smiles_col: str = "SMILES", feature_set: Literal["rdkit", "mordred"] = "rdkit", means: np.ndarray | None = None):
    if feature_set == "mordred":
        calc = Calculator(descriptors, ignore_3D=True)
        descs = calc.pandas(mols=[MolFromSmiles(s) for s in df[smiles_col]]).fill_missing()
    else:
        names = [x[0] for x in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        data = [calc.CalcDescriptors(MolFromSmiles(smiles)) for smiles in df[smiles_col]]
        descs = pd.DataFrame(columns=calc.GetDescriptorNames(), data=data)
    # imputation
    descs = descs.astype(float)
    descs = descs.replace([np.inf, -np.inf], np.nan)
    if means is None:
        means = descs.mean(axis=0, skipna=True)
    descs = descs.fillna(means)
    return pd.concat((df, descs), axis=1), means

def fit_symantic(df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS", downsample_size: int = None):
    input_df, means = _add_features(df, smiles_col)
    if downsample_size is not None and downsample_size > df.shape[0]:
        print(f"Downsample size {downsample_size} is larger than dataset size {df.shape[0]}, using full dataset instead.")
    elif downsample_size is None:
        downsample_size = df.shape[0]
    input_df = input_df.sample(n=downsample_size, random_state=42).reset_index(drop=True)
    # SISSO treats first column as target, remainder as features
    input_df.drop(columns=smiles_col, inplace=True)
    input_df = input_df[[target_col] + [c for c in input_df.columns if c != target_col]]
    symantic = SymanticModel(
        input_df,
        operators=OPERATORS,
        disp=True,
        metrics=[0.05, 0.99],
        initial_screening=['spearman', 0.95],
        n_term=4,
        sis_features=5,
    )
    res, _ = symantic.fit()
    eqn = res['utopia']['expression'].strip()
    # Only keep columns used in the equation
    features_in_eqn = list(set(re.findall(r'[a-zA-Z_]\w*', eqn)))
    
    def predictor(df_new: pd.DataFrame):
        df_new, _ = _add_features(df_new, smiles_col, means=means)
        return df_new.eval(eqn), df_new[features_in_eqn]

    return predictor, eqn
