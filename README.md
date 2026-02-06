# Interpretable Aqueous Solubility

The goal of this repository is to find new aqueous solubility prediction formulae which are inherently interpretable.
The primary approach to doing so is `TorchSISSO`.

## Usage

With Python 3.14 (or really any modern version of Python) use `pip` to install the following:

```
numpy
pandas
torch
torchsisso
scikit-learn
rdkit
sympy
scipy
```

This will allow you to run `main.py`, which actually fits the SISSO model.

To fit the comparison Chemprop-based `CheMeleon` model, you need a separate environment with `chemprop>=2.2.2` installed, which supports Python 3.12 or 3.11 - you could also use these and combine with the above environment, if you are so included.
Running `chemeleon.sh` and then `python chemeleon_results.py` will fit the model and run inference, then plot parity plots into the `results` directory.

All of the data needed is already in `data`, retrieved from:

 - `biogen`: https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/refs/heads/main/solubility/biogen_solubility.csv
 - `ochem` and `aqsoldbc`: https://github.com/JacksonBurns/fastprop_llompart/tree/main/data
 - `esol.csv`: https://www.kaggle.com/datasets/yeonseokcho/delaney

## Results

We fit `CheMeleon` and `TorchSISSO`, as well as re-fit ESOL, on the AqSolDBc dataset (except that we clip it to only -7 to -3 logS), and then test the fitted models on `ochem` (also clipped) and `biogen` (naturally clipped).

Under the current configuration, `TorchSISSO` discovers the below equation:

```
-0.2978142347*(MolLogP+NumHeteroatoms_norm)   + 0.0040525405*(NumValenceElectrons-HeavyAtomMolWt)  -2.777767233245860
```

The results for Pearson r are summarized here:

| Model | `ochem` | `biogen` |
|---|---|---|
| ESOL | 0.557 | 0.449 |
| `TorchSISSO` | 0.546 | 0.447 |
| `CheMeleon` | 0.798 | 0.470 |

Further data are available in `results`.

`TorchSISSO` can be improved by setting aside a validation set, and then identifying better values for:

 - the number of terms in the equation
 - the size of the expansion
 - which features are included
 - which samples are included
