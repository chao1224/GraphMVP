# Datasets

## Geometric Ensemble Of Molecules (GEOM)

```bash
mkdir -p GEOM/raw
mkdir -p GEOM/processed
```

+ GEOM: [Paper](https://arxiv.org/pdf/2006.05531v3.pdf), [GitHub](https://github.com/learningmatter-mit/geom)
+ Data Download:
    + [Not Used] [Drug Crude](https://dataverse.harvard.edu/api/access/datafile/4360331),
      [Drug Featurized](https://dataverse.harvard.edu/api/access/datafile/4327295),
      [QM9 Crude](https://dataverse.harvard.edu/api/access/datafile/4327190),
      [QM9 Featurized](https://dataverse.harvard.edu/api/access/datafile/4327191)

    + [Mainly Used] [RdKit Folder](https://dataverse.harvard.edu/api/access/datafile/4327252)
    ```bash
    wget https://dataverse.harvard.edu/api/access/datafile/4327252
    mv 4327252 rdkit_folder.tar.gz
    tar -xvf rdkit_folder.tar.gz
    ```
    or do the following if you are using slurm system
    ```
    cp rdkit_folder.tar.gz $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    tar -xvf rdkit_folder.tar.gz
    ```
+ over 33m conformations
+ over 430k molecules
    + 304,466 species contain experimental data for the inhibition of various pathogens
    + 133,258 are species from the QM9

## Chem Dataset

```bash
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset molecule_datasets
```

## Other Chem Datasets

- delaney/esol (already included)
- lipophilicity (already included)
- malaria
- cep

```
wget -O malaria-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv
mkdir -p ./molecule_datasets/malaria/raw
mv malaria-processed.csv ./molecule_datasets/malaria/raw/malaria.csv

wget -O cep-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv
mkdir -p ./molecule_datasets/cep/raw
mv cep-processed.csv ./molecule_datasets/cep/raw/cep.csv
```

Then we copy them for the regression (more atom features).
```
mkdir -p ./molecule_datasets_regression/esol
cp -r ./molecule_datasets/esol/raw               ./molecule_datasets_regression/esol/

mkdir -p ./molecule_datasets_regression/lipophilicity
cp -r ./molecule_datasets/lipophilicity/raw      ./molecule_datasets_regression/lipophilicity/

mkdir -p ./molecule_datasets_regression/malaria
cp -r ./molecule_datasets/malaria/raw            ./molecule_datasets_regression/malaria/

mkdir -p ./molecule_datasets_regression/cep
cp -r ./molecule_datasets/cep/raw                ./molecule_datasets_regression/cep/
```

## Drug-Target Interaction

- Davis
- Kiba

```
mkdir -p dti_datasets
cd dti_datasets
```

Then we can follow [DeepDTA](https://github.com/hkmztrk/DeepDTA).
```
git clone git@github.com:hkmztrk/DeepDTA.git
cp -r DeepDTA/data/davis davis/
cp -r DeepDTA/data/kiba kiba/

cd davis
python preprocess.py > preprocess.out
cd ..

cd kiba
python preprocess.py > preprocess.out
cd ..
```
