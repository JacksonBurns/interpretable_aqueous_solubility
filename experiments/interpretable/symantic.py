from typing import Literal

from .src.model import SymanticModel
import pandas as pd
from mordred import Calculator, descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles, Descriptors

OPERATORS = ['+', '-', '*', '/']


def _add_features(df: pd.DataFrame, smiles_col: str = "SMILES", feature_set: Literal["rdkit", "mordred"] = "rdkit"):
    if feature_set == "mordred":
        calc = Calculator(descriptors, ignore_3D=True)
        descs = calc.pandas(mols=[MolFromSmiles(s) for s in df[smiles_col]]).fill_missing(-1)
    else:
        names = [x[0] for x in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
        data = [calc.CalcDescriptors(MolFromSmiles(smiles)) for smiles in df[smiles_col]]
        descs = pd.DataFrame(columns=calc.GetDescriptorNames(), data=data)
    return pd.concat((df, descs), axis=1)

def fit_symantic(df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS"):
    input_df = _add_features(df, smiles_col)
    # SISSO treats first column as target, remainder as features
    input_df.drop(columns=smiles_col, inplace=True)
    input_df = input_df[[target_col] + [c for c in input_df.columns if c != target_col]]
    symantic = SymanticModel(
        input_df,
        operators=OPERATORS,
        disp=True,
        metrics=[0.05,0.99],
        initial_screening=['spearman',0.80],
        n_term = 2,
        sis_features=5,
    )
    res, _ = symantic.fit()
    eqn = res['utopia']['expression'].strip()
    
    def predictor(df_new: pd.DataFrame):
        df_new = _add_features(df_new, smiles_col)
        return df_new.eval(eqn)

    return predictor, eqn


# def fit_symantic_gp(df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS"):
# fit and get the symantic model, then train an sklearn gp regressor on the training residuals