from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import pandas as pd
import numpy as np

def fit_residual_gp(X_train: pd.DataFrame, y_true: pd.Series, y_pred_base: pd.Series):
    """
    Fits a Gaussian Process to the residuals (true - predicted) of a base model.
    Returns a predictor function that adds the GP residual prediction to the base prediction.
    """
    # drop any rows with nan predictions, warn
    mask = y_pred_base.notna()
    if not mask.all():
        print(f"Warning: Dropping {len(mask) - mask.sum()} rows with NaN predictions for GP fitting.")
    X_train = X_train[mask]
    y_true = y_true[mask]
    y_pred_base = y_pred_base[mask]

    # residuals
    residuals = y_true - y_pred_base
    
    # Standard kernel setup for regression on residuals
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
    
    # Fit the GP
    gp.fit(X_train.values, residuals.values)
    
    def predictor(X_new: pd.DataFrame, y_pred_new_base: pd.Series) -> np.ndarray:
        """
        Takes the features for the new data and the base model's predictions,
        and applies the residual correction.
        """
        residual_preds = gp.predict(X_new.values)
        return y_pred_new_base.values + residual_preds, X_new

    return predictor, gp
