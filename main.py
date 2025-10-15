#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install rdkit pysr ipykernel pandas matplotlib numpy scikit-learn


# In[2]:


import pandas as pd


# In[3]:


# df = pd.read_csv("ci034243xsi20040112_053635.txt")
df = pd.read_csv("AqSolDBc.csv").sample(1_024, random_state=42).reset_index(drop=True)  # author suggests >1000 not needed


# In[4]:


df


# In[5]:


from rdkit.Chem import Descriptors, MolFromSmiles


# In[6]:


def _f(smiles):
    try:
        return Descriptors.CalcMolDescriptors(MolFromSmiles(smiles))
    except Exception as e:
        print(f"Skipped molecule `{smiles}` because of exception {e}")
        return pd.NA


# In[7]:


# feature_df = pd.DataFrame.from_records(df["SMILES"].map(_f).to_list())
feature_df = pd.DataFrame.from_records(df["SmilesCurated"].map(_f).to_list())
feature_df


# In[8]:


from pysr import PySRRegressor


# In[ ]:


# model = PySRRegressor(
#     'best',
#     populations=96,
#     niterations=1_000_000,  # run until i interrupt
#     maxsize=32,
#     maxdepth=16,
#     batching=True,
#     batch_size=512,
#     binary_operators=["*", "+", "-", "/"],
#     unary_operators=["exp", "log", "abs"],
#     progress=True,
#     # denoise=True,  <-- takes forever
#     # select_k_features=32,  <-- while tempting, the author says this often doesn't help
#     #
#     #
#     # random_state=42,  <-- 
#     # deterministic=True,  <-- can't enable this unless willing to run in serial
#     # parallelism="serial", <-- 
# )
model = PySRRegressor.from_file(run_directory="outputs/20250927_130113_o3wIvl")


# In[10]:


# drop nan features and their corresponding rows in df
feature_df = feature_df.dropna(axis=1)
df = df.loc[feature_df.index]


# In[11]:


# model.fit(feature_df, df[["measured log(solubility:mol/L)"]])
# model.fit(feature_df, df[["ExperimentalLogS"]])  <-- uncomment to run training if not loading form file
# could also do a warm start


# In[12]:


model


# In[13]:

model
# best = model.get_best()
best = model.get_best(15)

pysr_eq = best["lambda_format"]  # <-- manual choice if not training now
pysr_eq

print("Selected model:")
print(best["equation"])


# In[14]:


train_pred = pysr_eq(feature_df[model.feature_names_in_])


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def parity_plot(y_true, y_pred, outname):
    """Create a parity plot with regression statistics."""
    # Compute regression statistics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", facecolor="C0", s=40)

    # 1:1 line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, color="red", linestyle="--", linewidth=1.5)

    # Labels and title
    plt.xlabel("Measured log(solubility) [mol/L]", fontsize=12)
    plt.ylabel("Predicted log(solubility) [mol/L]", fontsize=12)
    plt.title("Parity Plot", fontsize=14)

    # Annotation with statistics
    stats_text = (
        f"$R^2$ = {r2:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}"
    )
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3")
    )

    # Aesthetics
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outname}.png", dpi=300)

# y_true = df["measured log(solubility:mol/L)"].values
y_true = df["ExperimentalLogS"].values
y_pred = train_pred
parity_plot(y_true, y_pred, "train_parity")


# In[16]:


# test_df = pd.read_csv("biogen_solubility.csv")
test_df = pd.read_csv("OChemUnseen.csv")

# filter any invalid SMILES
test_df = test_df[test_df["SMILES"].map(lambda s: MolFromSmiles(s) is not None).to_list()]


# In[17]:


# filter out overlapping molecules based on their canonical SMILES
from rdkit.Chem import CanonSmiles

train_smiles = set(map(CanonSmiles, df["SMILES"]))
_original_length = test_df.shape[0]
test_df = test_df[~test_df["SMILES"].map(CanonSmiles).isin(train_smiles)]
print(f"Filtered {(_original_length - test_df.shape[0])} overlapping molecules from test set.")


# In[18]:


test_df


# In[19]:


test_features = pd.DataFrame.from_records(test_df["SMILES"].map(_f).to_list())
test_features


# In[20]:


test_pred = pysr_eq(test_features[model.feature_names_in_])
test_pred


# In[21]:


test_df


# In[22]:


test_true = test_df["LogS"].values


# In[23]:


# drop any NaN or inf predictions and print how many were dropped
nans = np.isnan(test_pred)
if nans.sum() > 0:
    print(f"Dropped {nans.sum()} NaN predictions")
    test_pred = test_pred[~nans]
    test_true = test_true[~nans]
infs = np.isinf(test_pred)
if infs.sum() > 0:
    print(f"Dropped {infs.sum()} inf predictions")
    test_pred = test_pred[~infs]
    test_true = test_true[~infs]


# In[24]:


parity_plot(test_true, test_pred, "test_parity")

