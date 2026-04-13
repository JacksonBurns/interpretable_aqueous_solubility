# Interpretable Aqueous Solubility

The goal of this repository is to explore the interpretability-performance tradeoff in aqueous solubility modeling.
To do so we use `SyMANTIC` to automatically discover interpretable equations from descriptors and machinelearning to push for absolute performance.

For comparison, we test all of the below models:

 - [ESOL](https://doi.org/10.1021/ci034243x)
 - [`CheMeleon`](https://doi.org/10.48550/arXiv.2506.15792)
 - Random Forest with Morgan Count and RDKit Descriptors, as in [MolPipeline](https://doi.org/10.1021/acs.jcim.4c00863)
 - [`SyMANTIC`](https://doi.org/10.48550/arXiv.2502.03367)

[Gaussian Process](https://scikit-learn.org/stable/modules/gaussian_process.html) modeling of the residuals is also applied to the ESOL and `SyMANTIC` linear models to demonstrate how giving up _some_ interpretability can change the performance.

## Usage

### Datasets

All of the data needed is already in `data`, retrieved from:

 - `biogen`: [this repository](https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/refs/heads/main/solubility/biogen_solubility.csv), originally curated [here](https://doi.org/10.1021/acs.jcim.3c00160)
 - `ochem` and `aqsoldbc`: [this demo repository](https://github.com/JacksonBurns/fastprop_llompart/tree/main/data), originally curated in [this article](https://doi.org/10.1038/s41597-024-03105-6)
 - `esol.csv`: (unused) [kaggle](https://www.kaggle.com/datasets/yeonseokcho/delaney), originally curated in the ESOL paper
 - `ancenes.csv`: curated from Wikipedia entries as part of this demo
 - `ralston_hoerr_joc_1942.csv`: fatty acid solubility data digitized from [THE SOLUBILITIES OF THE NORMAL SATURATED FATTY ACIDS](https://doi.org/10.1021/jo01200a013)

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
pysr
ipykernel
seaborn
```

### Execution

First run `experiments/fit.py` to actually fit the models.
With a consumer grade GPU for `CheMeleon` and sufficient memory for `SyMANTIC`, this should take 5-10 minutes using the RDKit descriptors or ~30 minutes with the `mordred` descriptors.
At the end, `biogen_pred.csv` and `ochem_pred.csv`, containing the predictions for the models, will be written to disk.

After that, execute `experiments/analyze.py` to generate parity plots and the final comparison statistics for all of the models.

You can also open and run `comparison.ipynb` to compare the datasets to one another, though this notebook doesn't change anything about the data or modeling process.

Finally, after running `fit.py` you can also run `demo/demo.ipynb` to compare how the various models perform on human-interpretable predictions.
Note that the linear equations are hard-coded into this notebook.
If you change anything about the data preparation, training, etc. you will need to copy the updated equations in to the notebook.

## Results

These are the final interpretable equations learned by `SyMANTIC`, ESOL, and PySR:

```
SyMANTIC equation: 0.0367*((MolLogP*LabuteASA)/(MolLogP-Chi0v)) + 499.497*((Chi0v+Chi0n)/(LabuteASA)**2) -2.863
PySR equation: (BCUT2D_CHGLO - MolLogP)/1.5725921487066359
ESOL refitted equation: logS = -6.640e-05*mw - 6.265e-01*logp + 9.637e-03*rotors - 1.087e+00*ap - 1.037e+00
```

Results are included in the `results` directory, including a simultaneous model comparison and parity plots for individual models.
