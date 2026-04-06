import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolFromSmiles, Descriptors

def _featurize(smiles: list[str]):
    fpgen = GetMorganGenerator(radius=2, fpSize=2_048)
    names = [x[0] for x in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    mols = list(map(MolFromSmiles, smiles))
    fps = np.array(list(map(fpgen.GetCountFingerprintAsNumPy, mols)))
    descs = np.array(list(map(desc_calc.CalcDescriptors, mols)))
    return np.concat((fps, descs), axis=1, dtype=np.float32)  # cast now to avoid overflow later


def _impute(X: np.ndarray, means: np.ndarray | None = None):
    X = np.where(np.isfinite(X), X, np.nan)  # treat inf as missing
    if means is None:
        means = np.nanmean(X, axis=0)
        means = np.where(np.isfinite(means), means, 0)
    return np.where(np.isfinite(X), X, means), means


def fit_rf(
    df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS",
):
    X = _featurize(df[smiles_col])
    X, means = _impute(X)
    y = df[target_col].values.reshape(-1, 1)
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    rf.fit(X, y)

    def predictor(df_new: pd.DataFrame):
        X_new = _featurize(df_new[smiles_col])
        X_new, _ = _impute(X_new, means)
        y_hat = rf.predict(X_new)
        return pd.Series(y_hat, index=df_new.index), X_new

    return predictor, rf
