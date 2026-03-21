import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from rdkit.Chem import MolFromSmiles
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from esol import fit_esol

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
    # "MolMR",  <-- hard to interpret
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
COUNT_DESCRIPTORS = [c for c in DESCRIPTORS if c.startswith("fr_") or c.startswith("Num")] + [
    "TPSA"
]
CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTORS)

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_and_downsample(df, max_k=10, samples_per_cluster=50):
    """
    Identifies optimal clusters based on features (all columns except the first)
    and downsamples the dataframe while keeping the target column intact.
    """
    # 1. Separate target and features
    # We assume the first column is the SMILES and the second is target
    features = df.iloc[:, 2:]

    # 2. Preprocessing: Scale only the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    best_k = 2
    best_score = -1

    # 3. Identify ideal number of clusters (Silhouette Score)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)

        if score > best_score:
            best_score = score
            best_k = k

    print(f"Optimal clusters found: {best_k} (Silhouette Score: {best_score:.3f})")

    # 4. Apply optimal clustering and add label back to the original full DF
    final_kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    df["_cluster_label"] = final_kmeans.fit_predict(scaled_features)

    # 5. Downsample the original dataframe (containing the target)
    downsampled_df = df.groupby("_cluster_label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_cluster), random_state=42)
    )
    return downsampled_df.reset_index(drop=True)


def add_ap(df):
    aromatic_query = Chem.MolFromSmarts("a")

    def calc_ap(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return len(mol.GetSubstructMatches(aromatic_query)) / mol.GetNumAtoms()

    df["AromaticProportion"] = df["SMILES"].map(calc_ap)
    return df


def _f(smiles):
    try:
        return CALCULATOR.CalcDescriptors(MolFromSmiles(smiles))
    except Exception as e:
        print(f"Skipped molecule `{smiles}` because of exception {e}")
        return pd.NA


def parity_plot(y_true, y_pred, outname):
    """Create a parity plot with hexbin-based density."""
    # Compute regression statistics
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(6, 6))

    hb = plt.hexbin(y_true, y_pred, gridsize=50, mincnt=1,)
    cb = plt.colorbar(
        hb,
        fraction=0.04,  # width relative to axes
        pad=0.02,  # gap between plot and colorbar
        shrink=0.85,  # height scaling
    )
    cb.set_label("# of Compounds", fontsize=11)

    # 1:1 line
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
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


from sklearn.ensemble import RandomForestRegressor


def robust_screen_features(df, target_col, force_include=None, n_keep=20):
    """
    Screens features using Random Forest importance (captures interactions),
    while bypassing screening for 'force_include' features.
    """
    if force_include is None:
        force_include = []

    # 1. Separate "forced" features from "candidate" features
    # Ensure forced features actually exist in the dataframe
    valid_force = [f for f in force_include if f in df.columns]
    missing = set(force_include) - set(valid_force)
    if missing:
        print(f"Warning: Forced features not found in DF: {missing}")

    # Candidates are everything else (excluding target and forced)
    candidates = [c for c in df.columns if c != target_col and c not in valid_force]

    # 2. If we already have few enough candidates, just return
    if len(candidates) + len(valid_force) <= n_keep:
        print(
            f"Feature count ({len(candidates) + len(valid_force)}) is below limit ({n_keep}). No screening needed."
        )
        return df[valid_force + candidates + [target_col]]

    print(f"Screening {len(candidates)} candidate features with Random Forest...")

    # 3. Train Random Forest to judge "candidate" usefulness
    # We use a lightweight forest (100 trees) to gauge importance
    X = df[candidates]
    y = df[target_col]

    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, verbose=1)
    rf.fit(X, y)

    # 4. Select top features to fill the remaining slots
    slots_remaining = n_keep - len(valid_force)
    if slots_remaining <= 0:
        print("Warning: 'force_include' list filled all available slots. Dropping all candidates.")
        selected_candidates = []
    else:
        # Get feature importances and sort indices
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]  # Descending order
        top_indices = indices[:slots_remaining]
        selected_candidates = [candidates[i] for i in top_indices]

        print(f"Selected top {len(selected_candidates)} candidates via RF Importance:")
        for i in range(slots_remaining):
            print(f" - {selected_candidates[i]} (Imp: {importances[indices[i]]:.4f})")

    # 5. Reconstruct DataFrame with Forced + Selected features (target goes first for SISSO)
    final_cols = [target_col] + valid_force + selected_candidates
    return df[final_cols]


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
    from rdkit.Chem import CanonSmiles
    # from TorchSisso import SissoModel

    # data loading
    df = pd.read_csv("data/aqsoldbc.csv")
    print(f"Original dataframe has {df.shape[0]} entries")

    # bound to -7 and -3 for applications to drug discovery
    df = df[(df["logS"] < -3) & (df["logS"] > -7)].reset_index(drop=True)
    print(f"Dropping samples outside of bounds reduces to {df.shape[0]} entries")

    # feature calculation
    feature_df = pd.DataFrame(columns=DESCRIPTORS, data=df["SMILES"].map(_f).to_list())
    feature_df = feature_df.dropna(axis=1)
    df = df.loc[feature_df.index]
    df = pd.concat((df, feature_df), axis=1)

    # filter to drug-like
    lipinski = (
        (df["MolWt"] <= 500).astype(int)
        + (df["NumHAcceptors"] <= 10).astype(int)
        + (df["NumHDonors"] <= 5).astype(int)
        + (df["MolLogP"] <= 5).astype(int)
    ).ge(3)
    df = df[lipinski].reset_index(drop=True)
    print(f"Removing Rule of 5 Violators Reduced to {df.shape[0]} entries")

    df[["SMILES", "logS"]].to_csv("data/chemeleon_aqsoldbc_filtered_train.csv", index=False)

    # feature engineering - convert discrete into density, add AP
    for col in COUNT_DESCRIPTORS:
        df[f"{col}_norm"] = df[col] / df["HeavyAtomCount"]
    df = add_ap(df)

    # baseline: ESOL
    esol_predict, esol_result = fit_esol(df, smiles_col="SMILES", target_col="logS",)
    print(esol_result)
    y_pred = esol_predict(df)
    parity_plot(df.loc[y_pred.index, "logS"], y_pred, "esol_train_parity")

    # our implementation of feature initial_screening
    df = robust_screen_features(
        df,
        target_col="logS",
        force_include=[
            "SMILES",
            "MolLogP",
            "MolMR",
            "TPSA_norm",
            "MolWt",
            "HeavyAtomCount",
            "AromaticProportion",
        ],
        n_keep=15 + 1,  # +1 for SMILES
    )

    # clustering-based downsampling
    df = df[["SMILES"] + [col for col in df.columns if col != "SMILES"]]  # ensure SMILES is first
    df = cluster_and_downsample(df, max_k=10, samples_per_cluster=50)
    print(f"After cluster-based downsampling, dataset has {df.shape[0]} rows")

    # SISSO treats first column as target, remainder as features
    sisso_df = df.drop(columns="SMILES")

    # model training
    # see: https://github.com/PaulsonLab/TorchSISSO/blob/main/README.md#installation
    sm = SissoModel(
        sisso_df,
        use_gpu=True,
        operators=["+", "-"],
        n_term=2,  # terms in final equation - could do a scree plot to determine this
        initial_screening=None,  # done manually
        n_expansion=2,  # hyperparameter, default 3
        k=30,  # hyperparameter, default 20
    )

    # Run the SISSO algorithm to get the interpretable model with the highest accuracy
    rmse, equation, r2, eqn_terms = sm.fit()
    equation = sympy_df_function(equation)
    y_true = df["logS"].values
    y_pred = equation(df)
    parity_plot(y_true, y_pred, "sisso_train_parity")

    # inference
    for f in ("biogen", "ochem"):
        for full in (True, False):
            if not full and f == "biogen":  # biogen already bounded
                continue
            test_df = pd.read_csv(f"data/{f}.csv")
            if not full:
                test_df = test_df[(test_df["logS"] < -3) & (test_df["logS"] > -7)].reset_index(
                    drop=True
                )
            test_df = test_df[
                test_df["SMILES"].map(lambda s: MolFromSmiles(s) is not None).to_list()
            ]
            train_smiles = set(map(CanonSmiles, df["SMILES"]))
            _original_length = test_df.shape[0]
            test_df = test_df[~test_df["SMILES"].map(CanonSmiles).isin(train_smiles)]
            print(
                f"Filtered {(_original_length - test_df.shape[0])} overlapping molecules from {f} test set."
            )

            # esol
            y_pred = esol_predict(test_df)
            parity_plot(
                test_df.loc[y_pred.index, "logS"],
                y_pred,
                f"results/esol_{f}{'_full' if full else ''}_parity",
            )

            # sisso
            smiles = test_df["SMILES"].to_list()
            logs = test_df["logS"].to_list()
            test_features = pd.DataFrame(
                columns=DESCRIPTORS, data=test_df["SMILES"].map(_f).to_list()
            )
            test_features.insert(0, "SMILES", smiles)
            test_features.insert(1, "logS", logs)
            for col in COUNT_DESCRIPTORS:
                test_features[f"{col}_norm"] = test_features[col] / test_features["HeavyAtomCount"]
            test_features = add_ap(test_features)
            # sisso
            test_pred = equation(test_features)
            # nans
            nan_guess = np.nanmean(test_pred)
            for i in range(len(smiles)):
                if np.isnan(test_pred[i]):
                    print("nan prediction found, replacing with nan mean")
                    print(test_features.iloc[i, :])
                    test_pred[i] = nan_guess
            test_true = test_df["logS"].values
            parity_plot(
                test_true, test_pred, f"results/sisso_{f}{'_full' if full else ''}_parity",
            )
