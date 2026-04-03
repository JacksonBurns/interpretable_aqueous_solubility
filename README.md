# Interpretable Aqueous Solubility

The goal of this repository is to find new aqueous solubility prediction formulae which are inherently interpretable.
The primary approach to doing so is `SyMANTIC`, which can automatically discover interpretable equations from descriptors.

For comparison, we test all of the below models:

 - ESOL (refitted)
 - CheMeleon
 - Random Forest with Morgan Count and RDKit Descriptors
 - SyMANTIC
 - SyMANTIC with Gaussian Process

## Usage

### Datasets

All of the data needed is already in `data`, retrieved from:

 - `biogen`: https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/refs/heads/main/solubility/biogen_solubility.csv
 - `ochem` and `aqsoldbc`: https://github.com/JacksonBurns/fastprop_llompart/tree/main/data
 - `esol.csv`: https://www.kaggle.com/datasets/yeonseokcho/delaney

### Dependencies

With Python 3.13 (or really any modern version of Python) use `pip` to install the following:

```
numpy
pandas
torch
torchsisso
scikit-learn
rdkit
sympy
scipy
gpytorch
mordredcommunity[full]
matplotlib
chemprop
```

### Execution



## Results

```
SyMANTIC equation: -0.5801802012520476*MolLogP + -5.94040410617455*(Chi4n/Chi0n) - 0.5215417871293462
```

```
ESOL refitted equation: logS = -6.751e-05*mw - 6.271e-01*logp + 9.861e-03*rotors - 1.089e+00*ap - 1.036e+00
```
