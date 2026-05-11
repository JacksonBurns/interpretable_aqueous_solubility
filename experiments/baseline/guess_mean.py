import numpy as np
import pandas as pd


def fit_guess_mean(
    df: pd.DataFrame, smiles_col: str = "SMILES", target_col: str = "logS",
):
    y_mean = df[target_col].values.mean()

    def predictor(df_new: pd.DataFrame):
        return pd.Series(np.full(df_new.shape[0], y_mean), index=df_new.index)

    return predictor, y_mean
