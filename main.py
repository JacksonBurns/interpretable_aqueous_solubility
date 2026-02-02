import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr

import sympy as sp

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from rdkit.Chem import MolFromSmiles
from rdkit.ML.Descriptors import MoleculeDescriptors


DESCRIPTORS = [
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "TPSA",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAmideBonds",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumAtomStereoCenters",
    "NumBridgeheadAtoms",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumHeterocycles",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumSpiroAtoms",
    "NumUnspecifiedAtomStereoCenters",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]
CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTORS)


def _f(smiles):
    try:
        return CALCULATOR.CalcDescriptors(MolFromSmiles(smiles))
    except Exception as e:
        print(f"Skipped molecule `{smiles}` because of exception {e}")
        return pd.NA


def parity_plot(y_true, y_pred, outname):
    """Create a parity plot with regression statistics."""
    # Compute regression statistics
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", facecolor="C0", s=40)

    # 1:1 line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, color="red", linestyle="--", linewidth=1.5)

    # Labels and title
    plt.xlabel("Measured log(solubility) [mol/L]", fontsize=12)
    plt.ylabel("Predicted log(solubility) [mol/L]", fontsize=12)
    plt.title("Parity Plot", fontsize=14)

    # Annotation with statistics
    stats_text = f"$r$ = {r:.3f}\n" f"RMSE = {rmse:.3f}\n" f"MAE = {mae:.3f}"
    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"),
    )

    # Aesthetics
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outname}.png", dpi=300)


def downsample_to_uniform(df, target_col, random_state=0):
    rng = np.random.default_rng(random_state)

    x = df[target_col].to_numpy()

    # Estimate density of the target distribution
    kde = gaussian_kde(x)
    p = kde(x)

    # Inverse-density acceptance probabilities
    # Normalize so max acceptance prob = 1
    accept_prob = 1.0 / p
    accept_prob /= accept_prob.max()

    # Rejection sampling
    keep = rng.random(len(df)) < accept_prob

    return df.loc[keep].reset_index(drop=True)


def sympy_df_function(expr_str: str):
    # Parse expression
    expr = sp.sympify(expr_str, evaluate=False)

    # Extract symbols (i.e., column names)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    symbol_names = [s.name for s in symbols]

    # Compile to a NumPy-aware callable
    f_np = sp.lambdify(symbols, expr, modules="numpy")

    def f(df):
        # Ensure required columns exist
        missing = set(symbol_names) - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        # Pass columns in correct order
        args = [df[name].values for name in symbol_names]
        return f_np(*args)

    return f


if __name__ == "__main__":
    import pandas as pd
    from rdkit.Chem import CanonSmiles

    from TorchSisso import SissoModel
    import numpy as np
    import pandas as pd

    # data loading

    df = pd.read_csv("aqsoldbc.csv")
    print(f"Original dataframe has {df.shape[0]} entries")
    # bound to -7 and -3 for applications to drug discovery
    df = df[(df["logS"] < -3) & (df["logS"] > -7)].reset_index(drop=True)
    print(f"Dropping samples outside of bounds reduces to {df.shape[0]} entries")
    # optional downsample to a uniform distribution
    # df = downsample_to_uniform(df, "logS")
    # print(f"Uniform downsampling further reduces to {df.shape[0]} entries")

    # feature calculation

    feature_df = pd.DataFrame(columns=DESCRIPTORS, data=df["SMILES"].map(_f).to_list())
    feature_df = feature_df.dropna(axis=1)
    df = df.loc[feature_df.index]
    df = pd.concat((df, feature_df), axis=1)
    sisso_df = df.drop(
        columns="SMILES"
    )  # SISSO treats first column as target, remainder as features

    # model training
    # see: https://github.com/PaulsonLab/TorchSISSO/blob/main/README.md#installation

    sm = SissoModel(
        sisso_df,
        use_gpu=True,
        operators=["+", "-", "*", "/", "log", "pow(2)"],
        n_term=2,  # terms in final equation - could do a scree plot to determine this
        initial_screening = ["mi", 0.95],  # use mutual information to eliminate unnecessary features  --> needed if many features used
        n_expansion=3,  # hyperparameter, default 3
        k=20,  # hyperparameter, default 20
    )

    # Run the SISSO algorithm to get the interpretable model with the highest accuracy
    rmse, equation, r2, eqn_terms = sm.fit()
    equation = sympy_df_function(equation)

    y_true = df["logS"].values
    y_pred = equation(df)
    parity_plot(y_true, y_pred, "train_parity")

    test_df = pd.read_csv("biogen.csv")  # <-- already bounded

    # test_df = pd.read_csv("ochem.csv")
    # test_df = test_df[(test_df["logS"] < -3) & (test_df["logS"] > -7)].reset_index(drop=True)

    test_df = test_df[test_df["SMILES"].map(lambda s: MolFromSmiles(s) is not None).to_list()]
    train_smiles = set(map(CanonSmiles, df["SMILES"]))
    _original_length = test_df.shape[0]
    test_df = test_df[~test_df["SMILES"].map(CanonSmiles).isin(train_smiles)]
    print(f"Filtered {(_original_length - test_df.shape[0])} overlapping molecules from test set.")

    test_features = pd.DataFrame(columns=DESCRIPTORS, data=test_df["SMILES"].map(_f).to_list())
    test_pred = equation(test_features)
    test_true = test_df["logS"].values
    parity_plot(test_true, test_pred, "test_parity")
