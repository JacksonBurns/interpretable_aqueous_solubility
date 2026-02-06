Pat Walters' Prepared Biogen Dataset

`wget https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_posts/refs/heads/main/solubility/biogen_solubility.csv`

Download the training data from the ESOL paper SI.

Download OChemUnseen from https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CZVZIA

Current version has this equation: -0.2978142347*(MolLogP+NumHeteroatoms_norm)   + 0.0040525405*(NumValenceElectrons-HeavyAtomMolWt)  -2.777767233245860

python3.14 then `pip install torchsisso`

`pip install numpy pandas matplotlib rdkit scipy scikit-learn`
