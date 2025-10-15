Pat Walters' Prepared Biogen Dataset

`wget https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/refs/heads/main/solubility/biogen_solubility.csv`

Download the training data from the ESOL paper SI.

Download OChemUnseen from https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CZVZIA

This is perhaps the final equation:

```
FractionCSP3 - (exp((((MolLogP * 1.9045392) + -8.334743) - abs(VSA_EState9 * log(qed))) / Chi0v) * 5.45409)
```
