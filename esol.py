from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from dataclasses import dataclass


@dataclass
class ESOLFitResult:
    intercept: float
    coefficients: dict
    rmse: float
    mae: float
    r: float
    feature_names: list


def fit_esol(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "logS",
):
    """
    Fit an ESOL-style linear model using RDKit descriptors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain SMILES and experimental solubility
    smiles_col : str
        Column containing SMILES
    target_col : str
        Column containing experimental logS

    Returns
    -------
    predictor : callable
        Function mapping a dataframe -> predicted logS
    result : ESOLFitResult
        Coefficients, intercept, and training metrics
    """

    aromatic_query = Chem.MolFromSmarts("a")

    def calc_ap(mol):
        return len(mol.GetSubstructMatches(aromatic_query)) / mol.GetNumAtoms()

    def calc_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "rotors": Lipinski.NumRotatableBonds(mol),
            "ap": calc_ap(mol),
        }

    # Compute descriptors
    desc_df = df[smiles_col].map(calc_descriptors).dropna().apply(pd.Series)

    # Align target
    y = df.loc[desc_df.index, target_col].values
    X = desc_df.values
    feature_names = desc_df.columns.tolist()

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Training predictions + metrics
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r, _ = pearsonr(y, y_pred)

    coef_dict = dict(zip(feature_names, model.coef_))

    result = ESOLFitResult(
        intercept=float(model.intercept_),
        coefficients=coef_dict,
        rmse=rmse,
        mae=mae,
        r=r,
        feature_names=feature_names,
    )

    def predictor(df_new: pd.DataFrame):
        desc_new = (
            df_new[smiles_col]
            .map(calc_descriptors)
            .dropna()
            .apply(pd.Series)
        )
        X_new = desc_new[feature_names].values
        y_hat = model.intercept_ + X_new @ model.coef_
        return pd.Series(y_hat, index=desc_new.index)

    return predictor, result
